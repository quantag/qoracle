# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Quantag IT Solutions GmbH
"""Compile C for AArch64 and extract one function's machine code from the object file."""
import os, shutil, subprocess, tempfile
import lief

CFLAGS = ["--target=aarch64-linux-gnu", "-ffreestanding", "-fno-stack-protector", "-fno-asynchronous-unwind-tables",
          "-fno-pic", "-ffunction-sections"]


class FrontendError(Exception):
    pass


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


def extract(obj, name):
    """(code bytes, address of the function in its section) of function `name` in an ELF object."""
    b = lief.parse(obj)
    if b is None or not isinstance(b, lief.ELF.Binary):
        raise FrontendError(f"{obj} is not an ELF object file")
    if b.header.machine_type != lief.ELF.ARCH.AARCH64:
        raise FrontendError(f"{obj} is not AArch64 code (other architectures come later)")
    syms = [s for s in b.symbols if s.name == name and s.type == lief.ELF.Symbol.TYPE.FUNC]
    if not syms:
        funcs = sorted(s.name for s in b.symbols if s.type == lief.ELF.Symbol.TYPE.FUNC and s.name)
        raise FrontendError(f"no function '{name}' in {obj} (functions: {', '.join(funcs) or 'none'})")
    s = syms[0]
    sec = b.sections[s.shndx]
    data = bytes(sec.content)
    code = data[s.value:s.value + s.size]
    if len(code) != s.size or not s.size:
        raise FrontendError(f"cannot read the code of '{name}'")
    for r in b.relocations:
        if r.section is not None and r.section.name == sec.name and s.value <= r.address < s.value + s.size:
            sym = r.symbol.name if r.has_symbol else ""
            if not sym or sym.startswith((".rodata", ".L")):
                raise FrontendError(f"'{name}' reads a constant table (the compiler may have turned a switch or "
                                    "a small loop into one); lookup tables are not supported yet: try -O1")
            raise FrontendError(f"'{name}' refers to '{sym}' (a call or global data); only self-contained "
                                "functions are supported yet")
    return code, s.value
