# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Quantag IT Solutions GmbH
"""Compile C for AArch64, and load one function from an object file together with the data it reads.

A small linker: the function's code is placed at CODE, every data section it refers to (constant
tables, initialised or zeroed globals) at its own page from DATA on, and the relocations of the code
are applied, so that the symbolic execution and the CPU emulator see the same program at the same
addresses.
"""
import os, shutil, struct, subprocess, tempfile
from dataclasses import dataclass, field
import lief

CFLAGS = ["--target=aarch64-linux-gnu", "-ffreestanding", "-fno-stack-protector", "-fno-asynchronous-unwind-tables",
          "-fno-pic", "-ffunction-sections"]
CODE, DATA = 0x100000, 0x400000
R = lief.ELF.Relocation.TYPE


class FrontendError(Exception):
    pass


@dataclass
class Image:
    code: bytes                 # the function's machine code, relocated
    entry: int                  # its address
    section: bytes              # the whole code section, relocated (the CPU emulator maps it)
    section_addr: int
    data: list = field(default_factory=list)      # [(address, bytes, section name)]


def compile_c(src, opt="1", cc="clang", extra=()):
    if not shutil.which(cc):
        raise FrontendError(f"{cc} not found (clang with the AArch64 target is needed)")
    fd, out = tempfile.mkstemp(suffix=".o", prefix="qoracle_")
    os.close(fd)
    cmd = [cc] + CFLAGS + [f"-O{opt}", "-c", src, "-o", out] + list(extra)
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise FrontendError("compiling failed:\n" + " ".join(cmd) + "\n" + p.stderr)
    return out, " ".join(cmd)


def _patch(code, off, value, bits):
    """Replace bits (list of (lo, width)) of the instruction at off with consecutive slices of value."""
    insn = struct.unpack_from("<I", code, off)[0]
    shift = 0
    for lo, width in bits:
        m = (1 << width) - 1
        insn = (insn & ~(m << lo)) | (((value >> shift) & m) << lo)
        shift += width
    struct.pack_into("<I", code, off, insn & 0xFFFFFFFF)


def load(obj, name):
    """The function `name` of an AArch64 ELF object, linked at fixed addresses (see the module text)."""
    b = lief.parse(obj)
    if b is None or not isinstance(b, lief.ELF.Binary):
        raise FrontendError(f"{obj} is not an ELF object file")
    if b.header.machine_type != lief.ELF.ARCH.AARCH64:
        raise FrontendError(f"{obj} is not AArch64 code (other architectures come later)")
    syms = [s for s in b.symbols if s.name == name and s.type == lief.ELF.Symbol.TYPE.FUNC]
    if not syms:
        funcs = sorted(s.name for s in b.symbols if s.type == lief.ELF.Symbol.TYPE.FUNC and s.name)
        raise FrontendError(f"no function '{name}' in {obj} (functions: {', '.join(funcs) or 'none'})")
    fsym = syms[0]
    sections = list(b.sections)
    csec = sections[fsym.shndx]
    code = bytearray(csec.content)
    lo, hi = fsym.value, fsym.value + fsym.size
    if hi > len(code) or not fsym.size:
        raise FrontendError(f"cannot read the code of '{name}'")
    place, data = {}, []
    state = {"next": DATA}

    def address_of(shndx):
        if shndx not in place:
            sec = sections[shndx]
            if sec.name.startswith(".text"):
                raise FrontendError(f"'{name}' refers to other code ({sec.name})")
            content = bytes(sec.size) if sec.type == lief.ELF.Section.TYPE.NOBITS else bytes(sec.content)
            for r in b.relocations:                       # tables of addresses would need relocating too
                if r.section is not None and r.section.name == sec.name:
                    raise FrontendError(f"'{name}' reads {sec.name}, which holds addresses (pointers or a jump "
                                        "table); not supported yet")
            place[shndx] = state["next"]
            data.append((state["next"], content, sec.name))
            state["next"] += (max(len(content), 1) + 0xFFF) & ~0xFFF
        return place[shndx]

    for r in b.relocations:
        if r.section is None or r.section.name != csec.name or not lo <= r.address < hi:
            continue
        s = r.symbol
        if s.type == lief.ELF.Symbol.TYPE.FUNC or s.shndx == 0:
            raise FrontendError(f"'{name}' calls or refers to '{s.name}'; only self-contained functions are supported yet")
        S = address_of(s.shndx) + (s.value if s.type != lief.ELF.Symbol.TYPE.SECTION else 0)
        X = S + r.addend
        P = CODE + r.address
        off = r.address
        t = r.type
        if t == R.AARCH64_ADR_PREL_PG_HI21:
            _patch(code, off, ((X & ~0xFFF) - (P & ~0xFFF)) >> 12, [(29, 2), (5, 19)])
        elif t == R.AARCH64_ADR_PREL_LO21:
            _patch(code, off, X - P, [(29, 2), (5, 19)])
        elif t == R.AARCH64_ADD_ABS_LO12_NC:
            _patch(code, off, X & 0xFFF, [(10, 12)])
        elif t in (R.AARCH64_LDST8_ABS_LO12_NC, R.AARCH64_LDST16_ABS_LO12_NC, R.AARCH64_LDST32_ABS_LO12_NC,
                   R.AARCH64_LDST64_ABS_LO12_NC, R.AARCH64_LDST128_ABS_LO12_NC):
            scale = {R.AARCH64_LDST8_ABS_LO12_NC: 0, R.AARCH64_LDST16_ABS_LO12_NC: 1, R.AARCH64_LDST32_ABS_LO12_NC: 2,
                     R.AARCH64_LDST64_ABS_LO12_NC: 3, R.AARCH64_LDST128_ABS_LO12_NC: 4}[t]
            _patch(code, off, (X & 0xFFF) >> scale, [(10, 12)])
        elif t in (R.AARCH64_CALL26, R.AARCH64_JUMP26):
            raise FrontendError(f"'{name}' calls '{s.name}'; only self-contained functions are supported yet")
        else:
            raise FrontendError(f"'{name}': relocation {t} is not supported yet")
    code = bytes(code)
    return Image(code=code[lo:hi], entry=CODE + lo, section=code, section_addr=CODE, data=data)
