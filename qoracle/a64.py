# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Quantag IT Solutions GmbH
"""AArch64 front end: decode machine code (capstone) and execute it symbolically on XAG words.

The state is the 31 general registers (64 bits each, as lists of XAG literals) and the NZCV flags.
Each instruction's effect is written down exactly as the architecture defines it, on symbolic bits;
the XAG folds constants and shares equal subexpressions as it goes. Integer instructions only:
Loads read the data the function refers to (constant tables, globals as initialised); with a
symbolic address a load is a quantum ROM (see Exec.load). Stores, the stack, calls, floating point and
system instructions are rejected with a clear message.
"""
import capstone
from capstone import arm64 as A
from .xag import FALSE, TRUE

R, I, MEM = A.ARM64_OP_REG, A.ARM64_OP_IMM, A.ARM64_OP_MEM
LOADS = {"ldr", "ldrb", "ldrh", "ldrsb", "ldrsh", "ldrsw", "ldur", "ldurb", "ldurh", "ldursb", "ldursh", "ldursw"}
SHIFT = {A.ARM64_SFT_LSL: "lsl", A.ARM64_SFT_LSR: "lsr", A.ARM64_SFT_ASR: "asr", A.ARM64_SFT_ROR: "ror"}
EXT = {A.ARM64_EXT_UXTB: (8, False), A.ARM64_EXT_UXTH: (16, False), A.ARM64_EXT_UXTW: (32, False), A.ARM64_EXT_UXTX: (64, False),
       A.ARM64_EXT_SXTB: (8, True), A.ARM64_EXT_SXTH: (16, True), A.ARM64_EXT_SXTW: (32, True), A.ARM64_EXT_SXTX: (64, True)}
CC = {A.ARM64_CC_EQ: "eq", A.ARM64_CC_NE: "ne", A.ARM64_CC_HS: "hs", A.ARM64_CC_LO: "lo", A.ARM64_CC_MI: "mi",
      A.ARM64_CC_PL: "pl", A.ARM64_CC_VS: "vs", A.ARM64_CC_VC: "vc", A.ARM64_CC_HI: "hi", A.ARM64_CC_LS: "ls",
      A.ARM64_CC_GE: "ge", A.ARM64_CC_LT: "lt", A.ARM64_CC_GT: "gt", A.ARM64_CC_LE: "le", A.ARM64_CC_AL: "al",
      A.ARM64_CC_NV: "al"}
BRANCHES = {"b", "cbz", "cbnz", "tbz", "tbnz", "ret"}
REJECT = {"bl": "a function call", "blr": "a call through a register", "br": "a jump through a register",
          "svc": "a system call", "brk": "a breakpoint", "hlt": "a halt"}


ARGS, RET_REG, NREGS, NFLAGS = list(range(8)), 0, 31, 4     # x0..x7 in, x0 out; NZCV


class Unsupported(Exception):
    pass


class State:
    """31 registers of 64 literals, and the flags. Words are never changed in place, so states can share them."""
    __slots__ = ("x", "nzcv")

    def __init__(self, x, nzcv):
        self.x, self.nzcv = x, nzcv

    def copy(self):
        return State(list(self.x), self.nzcv)


class Insn:
    def __init__(self, ci):
        self.addr, self.size = ci.address, ci.size
        self.mnemonic, self.op_str = ci.mnemonic, ci.op_str
        self.ops = list(ci.operands)
        self.writeback = getattr(ci, "writeback", False)
        self.cc = CC.get(ci.cc)
        self.reg_name = ci.reg_name
        m = self.mnemonic
        self.is_branch = m in BRANCHES or m.startswith("b.")
        self.is_ret, self.uncond = m == "ret", m == "b"
        self.targets = [o.imm for o in self.ops if o.type == I] if self.is_branch and m != "ret" else []
        if m in ("tbz", "tbnz"):
            self.targets = [self.ops[2].imm]

    def __repr__(self):
        return f"{self.addr:x}: {self.mnemonic} {self.op_str}"

    def where(self):
        return f"at 0x{self.addr:x} '{self.mnemonic} {self.op_str}'"


def check(ins):
    """Reject what the symbolic execution cannot handle, before anything runs."""
    m = ins.mnemonic
    if m in REJECT:
        raise Unsupported(f"{REJECT[m]} {ins.where()}")
    if m.startswith(("st", "prfm")) or (m.startswith("ld") and m not in LOADS):
        raise Unsupported(f"a memory write or an unsupported load {ins.where()} (functions may read tables, not write memory)")
    if m in LOADS:
        mem = [o for o in ins.ops if o.type == MEM]
        literal = len(ins.ops) == 2 and ins.ops[1].type == I         # ldr x0, label (a literal pool)
        if len(ins.ops) != 2 or (len(mem) != 1 and not literal) or ins.writeback:
            raise Unsupported(f"a load with write-back or an unusual form {ins.where()}")
        if mem and mem[0].mem.base and ins.reg_name(mem[0].mem.base) in ("sp", "wsp"):
            raise Unsupported(f"the stack {ins.where()} (the function keeps values in memory; try -O1 or higher)")
    for o in ins.ops:
        if o.type == R and ins.reg_name(o.reg) in ("sp", "wsp"):
            raise Unsupported(f"the stack pointer {ins.where()} (the function uses memory; try -O1 or higher)")
        if o.type not in (R, I) and not ins.is_branch and m not in LOADS:
            raise Unsupported(f"an operand of type {o.type} {ins.where()}")


def decode(code, base=0):
    md = capstone.Cs(capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM)
    md.detail = True
    out, pos = [], 0
    for ci in md.disasm(code, base):
        out.append(Insn(ci))
        pos = ci.address + ci.size - base
    if pos != len(code):
        raise Unsupported(f"cannot decode the instruction at offset 0x{base + pos:x}")
    return out


class Exec:
    """Symbolic execution of one instruction at a time."""

    def __init__(self, g, memory=()):
        """memory: [(address, bytes)] the function may read."""
        self.g = g
        self.memory = [(a, bytes(d)) for a, d in memory]
        self.active = TRUE              # the activity predicate of the block being executed (set by symexec)
        self.oob = FALSE                # some active load read outside the known data
        self.rom_bits = 0               # table bits read with a symbolic address (for the report)

    # ------------------------------------------------------------------ memory
    def load(self, addr, nbytes):
        """Read nbytes (little endian) at the symbolic 64-bit address addr.

        A quantum ROM: for every address the load could read (inside the known data, agreeing with the
        address bits that are constant), a decoder line eq = (addr == that address), built as an AND
        chain from the top variable bit down, so that lines share their prefixes. Lines are mutually
        exclusive, so each result bit is the XOR of the lines whose data has that bit set: CNOTs only.
        An active load matching no line reads outside the data: flagged in self.oob."""
        g = self.g
        const_mask = const_val = 0
        var = []
        for i, b in enumerate(addr):
            if b in (FALSE, TRUE):
                const_mask |= 1 << i
                const_val |= b << i
            else:
                var.append(i)
        cands = [(a + off, d[off:off + nbytes]) for a, d in self.memory for off in range(0, len(d) - nbytes + 1)
                 if ((a + off) & const_mask) == const_val]
        if not var:
            hit = [v for a, v in cands if a == const_val]
            if not hit:
                self.oob = g.OR(self.oob, self.active)
                return [FALSE] * (8 * nbytes)
            v = int.from_bytes(hit[0], "little")
            return g.const(v, 8 * nbytes)
        var.sort(reverse=True)
        bits = [FALSE] * (8 * nbytes)
        lines = []
        for a, v in cands:
            eq = TRUE
            for i in var:
                eq = g.AND(eq, addr[i] if (a >> i) & 1 else addr[i] ^ 1)
            lines.append(eq)
            val = int.from_bytes(v, "little")
            for k in range(8 * nbytes):
                if (val >> k) & 1:
                    bits[k] = g.XOR(bits[k], eq)
        self.rom_bits += 8 * nbytes * len(cands)
        inside = g.XOR_all(lines)
        self.oob = g.OR(self.oob, g.AND(self.active, inside ^ 1))
        return bits

    def address(self, ins, st, o):
        """The address of a memory operand [base, index (extended, shifted), #disp], as a 64-bit word."""
        g, m = self.g, o.mem
        a = self.reg(st, ins.reg_name(m.base).replace("w", "x", 1)) if m.base else g.const(0, 64)
        if m.index:
            idx = self.reg(st, ins.reg_name(m.index))
            if o.ext:
                bits, signed = EXT[o.ext]
                idx = idx[:bits] + [idx[:bits][-1] if signed else FALSE] * (64 - bits)
            idx = (idx + [FALSE] * 64)[:64]
            if o.shift.type == A.ARM64_SFT_LSL:
                idx = g.shift_const(idx, o.shift.value, "lsl")
            a = g.add(a, idx)[0]
        if m.disp:
            a = g.add(a, g.const(m.disp & ((1 << 64) - 1), 64))[0]
        return a

    # ------------------------------------------------------------------ operands
    def reg(self, st, name):
        name = name.lower()
        if name in ("xzr", "wzr"):
            return [FALSE] * (64 if name[0] == "x" else 32)
        if name in ("sp", "wsp"):
            raise Unsupported("the stack pointer (the function uses memory)")
        w = 64 if name[0] == "x" else 32
        return st.x[int(name[1:])][:w]

    def width(self, ins, k=0):
        return 64 if ins.reg_name(ins.ops[k].reg)[0] in "xs" else 32

    def write(self, st, name, word):
        name = name.lower()
        if name in ("xzr", "wzr"):
            return
        if name in ("sp", "wsp"):
            raise Unsupported("the stack pointer (the function uses memory)")
        n = int(name[1:])
        st.x[n] = (list(word[:64]) + [FALSE] * 64)[:64] if name[0] == "x" else list(word[:32]) + [FALSE] * 32

    def val(self, ins, st, k, w):
        g, o = self.g, ins.ops[k]
        if o.type == I:
            v = o.imm
            if o.shift.type == A.ARM64_SFT_LSL:
                v <<= o.shift.value
            return g.const(v & ((1 << w) - 1), w)
        if o.type != R:
            raise Unsupported(f"operand type {o.type} {ins.where()}")
        word = self.reg(st, ins.reg_name(o.reg))
        if o.ext:
            bits, signed = EXT[o.ext]
            word = word[:bits]
            word = word + [word[-1] if signed else FALSE] * (w - len(word))
            if o.shift.type == A.ARM64_SFT_LSL:
                word = g.shift_const(word[:w], o.shift.value, "lsl")
        else:
            word = (word + [FALSE] * w)[:w]
            if o.shift.type:
                word = g.shift_const(word, o.shift.value, SHIFT[o.shift.type])
        return word[:w]

    def cond(self, st, cc):
        g = self.g
        n, z, c, v = st.nzcv
        if cc in (None, "al"):
            return TRUE
        ge = g.XOR(n, v) ^ 1
        hi = g.AND(c, z ^ 1)
        gt = g.AND(z ^ 1, ge)
        return {"eq": z, "ne": z ^ 1, "hs": c, "lo": c ^ 1, "mi": n, "pl": n ^ 1, "vs": v, "vc": v ^ 1,
                "hi": hi, "ls": hi ^ 1, "ge": ge, "lt": ge ^ 1, "gt": gt, "le": gt ^ 1}[cc]

    def flags_add(self, res, carry, c_top):
        g = self.g
        return (res[-1], g.eq_zero(res), carry, g.XOR(carry, c_top))

    def flags_logic(self, res):
        return (res[-1], self.g.eq_zero(res), FALSE, FALSE)

    # ------------------------------------------------------------------ one instruction
    def step(self, ins, st):
        """Execute a non-branch instruction on st (changed in place)."""
        g, m, ops = self.g, ins.mnemonic, ins.ops
        rn = lambda k: ins.reg_name(ops[k].reg)
        if m in REJECT:
            raise Unsupported(f"{REJECT[m]} {ins.where()}")
        if m.startswith(("st", "prfm")) or (m.startswith("ld") and m not in LOADS):
            raise Unsupported(f"a memory write or an unsupported load {ins.where()}")
        if m in ("nop", "hint", "bti", "paciasp", "autiasp"):
            return
        w = self.width(ins) if ops and ops[0].type == R else 64

        if m in ("adrp", "adr"):
            self.write(st, rn(0), g.const(ops[1].imm & ((1 << 64) - 1), 64)); return
        if m in LOADS:
            size = {"b": 1, "h": 2}.get(m[-1], 4 if m.endswith("sw") else w // 8)
            signed = m.startswith(("ldrs", "ldurs"))
            a = (g.const(ops[1].imm & ((1 << 64) - 1), 64) if ops[1].type == I
                 else self.address(ins, st, ops[1]))
            v = self.load(a, size)
            self.write(st, rn(0), (v + [v[-1] if signed else FALSE] * w)[:w]); return

        if m == "mov":
            self.write(st, rn(0), self.val(ins, st, 1, w)); return
        if m in ("movz", "movn"):
            v = self.val(ins, st, 1, w)
            self.write(st, rn(0), [b ^ 1 for b in v] if m == "movn" else v); return
        if m == "movk":
            o = ops[1]
            sh = o.shift.value if o.shift.type else 0
            old = self.reg(st, rn(0))
            new = old[:sh] + g.const(o.imm & 0xFFFF, 16) + old[sh + 16:]
            self.write(st, rn(0), new[:w]); return
        if m in ("mvn",):
            self.write(st, rn(0), [b ^ 1 for b in self.val(ins, st, 1, w)]); return

        # add / subtract (and their aliases)
        if m in ("add", "adds", "sub", "subs", "cmp", "cmn", "neg", "negs", "adc", "adcs", "sbc", "sbcs", "ngc", "ngcs"):
            if m in ("cmp", "cmn"):
                a, b, dst = self.val(ins, st, 0, w), self.val(ins, st, 1, w), None
            elif m in ("neg", "negs", "ngc", "ngcs"):
                a, b, dst = g.const(0, w), self.val(ins, st, 1, w), rn(0)
            else:
                a, b, dst = self.val(ins, st, 1, w), self.val(ins, st, 2, w), rn(0)
            subtract = m in ("sub", "subs", "cmp", "neg", "negs", "sbc", "sbcs", "ngc", "ngcs")
            cin = st.nzcv[2] if m in ("adc", "adcs", "sbc", "sbcs", "ngc", "ngcs") else (TRUE if subtract else FALSE)
            res, carry, c_top = g.add(a, [x ^ 1 for x in b] if subtract else b, cin)
            if dst:
                self.write(st, dst, res)
            if m.endswith("s") or m in ("cmp", "cmn"):
                st.nzcv = self.flags_add(res, carry, c_top)
            return

        # logical
        if m in ("and", "ands", "tst", "orr", "eor", "eon", "bic", "bics", "orn"):
            if m == "tst":
                a, b, dst = self.val(ins, st, 0, w), self.val(ins, st, 1, w), None
            else:
                a, b, dst = self.val(ins, st, 1, w), self.val(ins, st, 2, w), rn(0)
            if m in ("eon", "bic", "bics", "orn"):
                b = [x ^ 1 for x in b]
            op = {"and": g.AND, "ands": g.AND, "tst": g.AND, "bic": g.AND, "bics": g.AND,
                  "orr": g.OR, "orn": g.OR, "eor": g.XOR, "eon": g.XOR}[m]
            res = [op(x, y) for x, y in zip(a, b)]
            if dst:
                self.write(st, dst, res)
            if m in ("ands", "tst", "bics"):
                st.nzcv = self.flags_logic(res)
            return

        # multiply
        if m in ("mul", "madd", "msub", "mneg"):
            a, b = self.val(ins, st, 1, w), self.val(ins, st, 2, w)
            p = g.mul(a, b, w)
            if m == "mul":
                r = p
            elif m == "mneg":
                r = g.neg(p)
            else:
                c = self.val(ins, st, 3, w)
                r = g.add(c, p)[0] if m == "madd" else g.sub(c, p)[0]
            self.write(st, rn(0), r); return
        if m in ("umull", "smull", "umaddl", "smaddl", "umsubl", "smsubl", "umnegl", "smnegl"):
            signed = m[0] == "s"
            a, b = self.reg(st, rn(1))[:32], self.reg(st, rn(2))[:32]
            ext = lambda v: v + [v[-1] if signed else FALSE] * 32
            p = g.mul(ext(a), ext(b), 64)
            if m.endswith("addl"):
                p = g.add(self.reg(st, rn(3)), p)[0]
            elif m.endswith("subl"):
                p = g.sub(self.reg(st, rn(3)), p)[0]
            elif m.endswith("negl"):
                p = g.neg(p)
            self.write(st, rn(0), p); return
        if m in ("umulh", "smulh"):
            signed = m[0] == "s"
            a, b = self.reg(st, rn(1)), self.reg(st, rn(2))
            ext = lambda v: v + [v[-1] if signed else FALSE] * 64
            self.write(st, rn(0), g.mul(ext(a), ext(b), 128)[64:]); return

        # divide
        if m == "udiv":
            self.write(st, rn(0), g.udivmod(self.val(ins, st, 1, w), self.val(ins, st, 2, w))[0]); return
        if m == "sdiv":
            self.write(st, rn(0), g.sdiv(self.val(ins, st, 1, w), self.val(ins, st, 2, w))); return

        # shifts and rotates
        if m in ("lsl", "lsr", "asr", "ror", "lslv", "lsrv", "asrv", "rorv"):
            a = self.val(ins, st, 1, w)
            kind = m[:3]
            if ops[2].type == I:
                r = g.shift_const(a, ops[2].imm, kind)
            else:
                r = g.shift_var(a, self.val(ins, st, 2, w), kind)
            self.write(st, rn(0), r); return

        # bit fields and extensions
        if m in ("sxtb", "sxth", "sxtw", "uxtb", "uxth"):
            bits = {"b": 8, "h": 16, "w": 32}[m[-1]]
            a = self.reg(st, rn(1))[:bits]
            self.write(st, rn(0), a + [a[-1] if m[0] == "s" else FALSE] * (w - bits)); return
        if m in ("ubfx", "sbfx", "ubfiz", "sbfiz", "bfi", "bfxil", "bfc"):
            if m == "bfc":
                lsb, width = ops[1].imm, ops[2].imm
                old = self.reg(st, rn(0))[:w]
                self.write(st, rn(0), old[:lsb] + [FALSE] * width + old[lsb + width:]); return
            a = self.val(ins, st, 1, w)
            lsb, width = ops[2].imm, ops[3].imm
            if m in ("ubfx", "sbfx"):
                f = a[lsb:lsb + width]
                r = f + [f[-1] if m == "sbfx" else FALSE] * (w - width)
            elif m in ("ubfiz", "sbfiz"):
                f = a[:width]
                r = ([FALSE] * lsb + f + [f[-1] if m == "sbfiz" else FALSE] * w)[:w]
            elif m == "bfi":
                old = self.reg(st, rn(0))[:w]
                r = old[:lsb] + a[:width] + old[lsb + width:]
            else:                                                       # bfxil
                old = self.reg(st, rn(0))[:w]
                r = a[lsb:lsb + width] + old[width:]
            self.write(st, rn(0), r[:w]); return
        if m == "extr":
            a, b, lsb = self.val(ins, st, 1, w), self.val(ins, st, 2, w), ops[3].imm
            self.write(st, rn(0), (b + a)[lsb:lsb + w]); return
        if m in ("rbit", "rev", "rev16", "rev32"):
            a = self.val(ins, st, 1, w)
            if m == "rbit":
                r = a[::-1]
            else:
                unit = {"rev": w, "rev16": 16, "rev32": 32}[m]
                r = []
                for base in range(0, w, unit):
                    chunk = a[base:base + unit]
                    r += [b for byte in reversed([chunk[i:i + 8] for i in range(0, unit, 8)]) for b in byte]
            self.write(st, rn(0), r); return
        if m in ("clz", "cls"):
            a = self.val(ins, st, 1, w)
            if m == "cls":
                a = [g.XOR(a[i], a[i + 1]) for i in range(w - 1)] + [FALSE]
            # count = number of leading zeros: scan from the top, keep "all zero so far"
            cnt, allz = g.const(0, w), TRUE
            for i in range(w - 1, -1, -1):
                allz = g.AND(allz, a[i] ^ 1)
                cnt = g.add(cnt, [allz] + [FALSE] * (w - 1))[0]
            self.write(st, rn(0), cnt if m == "clz" else g.sub(cnt, g.const(1, w))[0]); return

        # conditional select and compare
        if m in ("csel", "csinc", "csinv", "csneg"):
            a, b = self.val(ins, st, 1, w), self.val(ins, st, 2, w)
            if m == "csinc":
                b = g.add(b, g.const(1, w))[0]
            elif m == "csinv":
                b = [x ^ 1 for x in b]
            elif m == "csneg":
                b = g.neg(b)
            c = self.cond(st, ins.cc)
            self.write(st, rn(0), [g.MUX(c, x, y) for x, y in zip(a, b)]); return
        if m in ("cset", "csetm"):
            c = self.cond(st, ins.cc)
            self.write(st, rn(0), [c] + [c if m == "csetm" else FALSE] * (w - 1)); return
        if m in ("cinc", "cinv", "cneg"):
            a = self.val(ins, st, 1, w)
            b = {"cinc": lambda: g.add(a, g.const(1, w))[0], "cinv": lambda: [x ^ 1 for x in a], "cneg": lambda: g.neg(a)}[m]()
            c = self.cond(st, ins.cc)
            self.write(st, rn(0), [g.MUX(c, x, y) for x, y in zip(b, a)]); return
        if m in ("ccmp", "ccmn"):
            a, b = self.val(ins, st, 0, w), self.val(ins, st, 1, w)
            nzcv = ops[2].imm
            res, carry, c_top = g.add(a, [x ^ 1 for x in b] if m == "ccmp" else b, TRUE if m == "ccmp" else FALSE)
            new = self.flags_add(res, carry, c_top)
            alt = tuple(TRUE if (nzcv >> (3 - i)) & 1 else FALSE for i in range(4))
            c = self.cond(st, ins.cc)
            st.nzcv = tuple(g.MUX(c, x, y) for x, y in zip(new, alt)); return

        raise Unsupported(f"instruction not supported yet {ins.where()}")

    def branch(self, ins, st):
        """The predicate under which a branch is taken (relative to reaching it), or 'ret'."""
        g, m, ops = self.g, ins.mnemonic, ins.ops
        if m == "ret":
            return "ret"
        if m == "b":
            return TRUE
        if m.startswith("b."):
            return self.cond(st, ins.cc)
        if m in ("cbz", "cbnz"):
            z = g.eq_zero(self.reg(st, ins.reg_name(ops[0].reg)))
            return z if m == "cbz" else z ^ 1
        if m in ("tbz", "tbnz"):
            bit = self.reg(st, ins.reg_name(ops[0].reg).replace("w", "x"))[ops[1].imm]
            return bit ^ 1 if m == "tbz" else bit
        raise Unsupported(f"branch {ins.where()}")
