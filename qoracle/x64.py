# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Quantag IT Solutions GmbH
"""x86-64 front end: decode machine code (capstone) and execute it symbolically on XAG words.

The same model as a64.py: the state is the 16 general registers (64 bits each, as lists of XAG literals) and
the flags SF, ZF, CF, OF and PF (kept in State.nzcv, in that order, so that symexec merges them alike). The System V
calling convention: arguments in RDI, RSI, RDX, RCX, R8, R9, the result in RAX.

Integer instructions only, each written down as the architecture defines it on symbolic bits: MOV and its
extensions, LEA, ADD/SUB/ADC/SBB/CMP/NEG/INC/DEC, AND/OR/XOR/TEST/NOT, IMUL and MUL, DIV and IDIV, the shifts
and rotates, SETcc, CMOVcc, Jcc/JMP/RET, CDQE/CDQ/CQO, BSWAP, POPCNT, LZCNT, TZCNT, XCHG. Loads read the data the
function refers to (RIP-relative tables); a symbolic address is a quantum ROM (a64.Exec.load). Stores, the stack
(PUSH, POP, RSP or RBP addressing), calls, SIMD and floating point are rejected with a clear
message. Flags an instruction leaves undefined are computed as on current Intel and AMD processors where that is
well defined, else as 0; the verification against the real machine code on a CPU emulator catches any use of them.
"""
import capstone
from capstone import x86 as X
from .xag import FALSE, TRUE
from .a64 import Unsupported, Exec as _A64Exec

R, I, MEM = X.X86_OP_REG, X.X86_OP_IMM, X.X86_OP_MEM
NAMES = ["rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi"] + [f"r{i}" for i in range(8, 16)]
ARGS = [7, 6, 2, 1, 8, 9]                          # rdi rsi rdx rcx r8 r9
RET_REG, NREGS, NFLAGS = 0, 16, 5                # flags: SF ZF CF OF PF
SUB = {}                                           # register name -> (index, low bit, width)
for k, n in enumerate(NAMES[:8]):
    b = n[1:]                                      # ax cx dx bx sp bp si di
    SUB[n] = (k, 0, 64)
    SUB["e" + b] = (k, 0, 32)
    SUB[b] = (k, 0, 16)
    if b in ("ax", "cx", "dx", "bx"):
        SUB[b[0] + "l"] = (k, 0, 8)
        SUB[b[0] + "h"] = (k, 8, 8)
    else:
        SUB[b + "l"] = (k, 0, 8)                   # spl bpl sil dil
for k in range(8, 16):
    SUB[f"r{k}"], SUB[f"r{k}d"], SUB[f"r{k}w"], SUB[f"r{k}b"] = (k, 0, 64), (k, 0, 32), (k, 0, 16), (k, 0, 8)

CC = {"o": "o", "no": "no", "b": "b", "c": "b", "nae": "b", "ae": "ae", "nb": "ae", "nc": "ae", "e": "e", "z": "e",
      "ne": "ne", "nz": "ne", "be": "be", "na": "be", "a": "a", "nbe": "a", "s": "s", "ns": "ns", "l": "l",
      "nge": "l", "ge": "ge", "nl": "ge", "le": "le", "ng": "le", "g": "g", "nle": "g", "p": "p", "pe": "p",
      "np": "np", "po": "np"}
REJECT = {"call": "a function call", "syscall": "a system call", "int": "an interrupt", "int3": "a breakpoint",
          "hlt": "a halt", "push": "the stack (push)", "pop": "the stack (pop)", "leave": "the stack (leave)",
          "enter": "the stack (enter)", "pushfq": "the stack", "popfq": "the stack", "ud2": "an undefined instruction",
          "rep stosb": "a memory write", "rep movsb": "a memory write"}
NOPS = {"nop", "endbr64", "endbr32", "pause", "lfence", "mfence", "sfence"}


class State:
    """16 registers of 64 literals, and the flags (SF, ZF, CF, OF, PF). Words are never changed in place."""
    __slots__ = ("x", "nzcv")

    def __init__(self, x, nzcv):
        self.x, self.nzcv = x, nzcv

    def copy(self):
        return State(list(self.x), self.nzcv)


def _cc(m, prefix):
    c = m[len(prefix):]
    return CC.get(c)


class Insn:
    def __init__(self, ci):
        self.addr, self.size = ci.address, ci.size
        self.mnemonic, self.op_str = ci.mnemonic, ci.op_str
        self.ops = list(ci.operands)
        self.reg_name = ci.reg_name
        self.prefix = [p for p in ci.prefix if p]
        m = self.mnemonic
        self.is_ret = m in ("ret", "retq")
        self.uncond = m == "jmp"
        self.cc = _cc(m, "j") if m.startswith("j") and m != "jmp" else None
        self.is_branch = self.is_ret or self.uncond or self.cc is not None
        self.targets = [o.imm for o in self.ops if o.type == I] if self.is_branch and not self.is_ret else []

    def __repr__(self):
        return f"{self.addr:x}: {self.mnemonic} {self.op_str}"

    def where(self):
        return f"at 0x{self.addr:x} '{self.mnemonic} {self.op_str}'"


def _regname(ins, r):
    return ins.reg_name(r).lower() if r else None


def check(ins):
    """Reject what the symbolic execution cannot handle, before anything runs."""
    m = ins.mnemonic
    if m in NOPS or m.startswith("nop"):                 # alignment padding (nop word ptr cs:[rax + rax])
        return
    if m in REJECT or m.split()[0] in ("rep", "repne", "repe"):
        raise Unsupported(f"{REJECT.get(m, 'a string instruction')} {ins.where()}")
    if m.startswith("j") and not ins.is_branch:
        raise Unsupported(f"a jump through a register {ins.where()}")
    if ins.is_branch and not ins.is_ret and not ins.targets:
        raise Unsupported(f"a jump through a register or memory (a jump table?) {ins.where()}")
    for k, o in enumerate(ins.ops):
        if o.type == R:
            n = _regname(ins, o.reg)
            if n not in SUB:
                raise Unsupported(f"the register {n} (SIMD, floating point or a system register) {ins.where()}")
            if SUB[n][0] in (4, 5) and m not in NOPS:
                raise Unsupported(f"the stack pointer or frame pointer {n} {ins.where()} (the function uses memory; "
                                  "try -O1 or higher, -fomit-frame-pointer)")
        elif o.type == MEM:
            base, index = _regname(ins, o.mem.base), _regname(ins, o.mem.index)
            if o.mem.segment:
                raise Unsupported(f"a segment override (thread-local data) {ins.where()}")
            if base in ("rsp", "esp", "rbp", "ebp") or index in ("rsp", "rbp"):
                raise Unsupported(f"the stack {ins.where()} (the function keeps values in memory; try -O1 or higher)")
            if k == 0 and m not in ("cmp", "test", "nop", "lea") and len(ins.ops) >= 1 and not m.startswith("nop"):
                raise Unsupported(f"a memory write {ins.where()} (functions may read tables, not write memory)")
        elif o.type != I:
            raise Unsupported(f"an operand of type {o.type} {ins.where()}")


def decode(code, base=0):
    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    md.detail = True
    out, pos = [], 0
    for ci in md.disasm(code, base):
        out.append(Insn(ci))
        pos = ci.address + ci.size - base
    if pos != len(code):
        raise Unsupported(f"cannot decode the instruction at offset 0x{base + pos:x}")
    return out


class Exec(_A64Exec):
    """Symbolic execution of one x86-64 instruction at a time (the loads are a64's quantum ROM)."""

    # ------------------------------------------------------------------ operands
    def reg(self, st, name):
        k, lo, w = SUB[name]
        return st.x[k][lo:lo + w]

    def write(self, st, name, word):
        k, lo, w = SUB[name]
        word = list(word[:w]) + [FALSE] * max(0, w - len(word))
        if w == 64:
            st.x[k] = word
        elif w == 32:                              # a 32-bit write clears the upper half
            st.x[k] = word + [FALSE] * 32
        else:                                      # 16- and 8-bit writes keep the rest
            old = st.x[k]
            st.x[k] = old[:lo] + word + old[lo + w:]

    def address(self, ins, st, o):
        g, m = self.g, o.mem
        base = _regname(ins, m.base)
        if base in ("rip", "eip"):
            a = g.const((ins.addr + ins.size) & ((1 << 64) - 1), 64)
        elif base:
            a = (self.reg(st, base) + [FALSE] * 64)[:64]
        else:
            a = g.const(0, 64)
        if m.index:
            idx = (self.reg(st, _regname(ins, m.index)) + [FALSE] * 64)[:64]
            if m.scale > 1:
                idx = g.shift_const(idx, m.scale.bit_length() - 1, "lsl")
            a = g.add(a, idx)[0]
        if m.disp:
            a = g.add(a, g.const(m.disp & ((1 << 64) - 1), 64))[0]
        return a

    def val(self, ins, st, k, w=None):
        """Operand k as a word of w bits (its own size by default); immediates sign-extended."""
        g, o = self.g, ins.ops[k]
        w = w or 8 * o.size
        if o.type == I:
            return g.const(o.imm & ((1 << w) - 1), w)
        if o.type == R:
            word = self.reg(st, _regname(ins, o.reg))
        else:
            word = self.load(self.address(ins, st, o), o.size)
        return (word + [FALSE] * w)[:w]

    def size(self, ins, k=0):
        return 8 * ins.ops[k].size

    def dst(self, ins):
        return _regname(ins, ins.ops[0].reg)

    def cond(self, st, cc):
        g = self.g
        sf, zf, cf, of, pf = st.nzcv
        lt = g.XOR(sf, of)
        be = g.OR(cf, zf)
        le = g.OR(zf, lt)
        return {"o": of, "no": of ^ 1, "b": cf, "ae": cf ^ 1, "e": zf, "ne": zf ^ 1, "be": be, "a": be ^ 1,
                "s": sf, "ns": sf ^ 1, "l": lt, "ge": lt ^ 1, "le": le, "g": le ^ 1, "p": pf, "np": pf ^ 1}[cc]

    def flags_res(self, res, cf=FALSE, of=FALSE):
        return (res[-1], self.g.eq_zero(res), cf, of, self.parity(res))

    def parity(self, res):
        """PF: 1 when the low byte has an even number of ones."""
        return self.g.XOR_all(res[:8]) ^ 1

    @staticmethod
    def keep(st, **new):
        """The flags with some replaced: sf zf cf of pf."""
        f = dict(zip(("sf", "zf", "cf", "of", "pf"), st.nzcv))
        f.update(new)
        return (f["sf"], f["zf"], f["cf"], f["of"], f["pf"])

    # ------------------------------------------------------------------ one instruction
    def step(self, ins, st):
        g, m, ops = self.g, ins.mnemonic, ins.ops
        if m in NOPS or m.startswith("nop"):
            return
        if m in REJECT:
            raise Unsupported(f"{REJECT[m]} {ins.where()}")
        w = self.size(ins) if ops else 64

        if m in ("mov", "movabs"):
            self.write(st, self.dst(ins), self.val(ins, st, 1, w)); return
        if m in ("movzx", "movsx", "movsxd"):
            v = self.val(ins, st, 1)
            ext = v[-1] if m != "movzx" else FALSE
            self.write(st, self.dst(ins), v + [ext] * (w - len(v))); return
        if m == "lea":
            self.write(st, self.dst(ins), self.address(ins, st, ops[1])[:w]); return
        if m in ("cdqe", "cwde", "cbw"):
            half = {"cdqe": 32, "cwde": 16, "cbw": 8}[m]
            v = st.x[0][:half]
            self.write(st, {"cdqe": "rax", "cwde": "eax", "cbw": "ax"}[m], v + [v[-1]] * half); return
        if m in ("cdq", "cqo", "cwd"):
            half = {"cdq": 32, "cqo": 64, "cwd": 16}[m]
            s = st.x[0][half - 1]
            self.write(st, {"cdq": "edx", "cqo": "rdx", "cwd": "dx"}[m], [s] * half); return
        if m == "xchg":
            a, b = self.val(ins, st, 0), self.val(ins, st, 1)
            self.write(st, self.dst(ins), b)
            self.write(st, _regname(ins, ops[1].reg), a); return
        if m == "bswap":
            a = self.val(ins, st, 0)
            self.write(st, self.dst(ins), [b for byte in reversed([a[i:i + 8] for i in range(0, w, 8)]) for b in byte])
            return

        if m in ("add", "adc", "sub", "sbb", "cmp", "neg", "inc", "dec"):
            if m == "neg":
                a, b = g.const(0, w), self.val(ins, st, 0)
            elif m in ("inc", "dec"):
                a, b = self.val(ins, st, 0), g.const(1, w)
            else:
                a, b = self.val(ins, st, 0), self.val(ins, st, 1, w)
            sub = m in ("sub", "sbb", "cmp", "neg", "dec")
            if m in ("adc", "sbb"):
                cin = st.nzcv[2] ^ 1 if m == "sbb" else st.nzcv[2]
            else:
                cin = TRUE if sub else FALSE
            res, carry, c_top = g.add(a, [x ^ 1 for x in b] if sub else b, cin)
            cf = carry ^ 1 if sub else carry               # x86: CF is the borrow of a subtraction
            of = g.XOR(carry, c_top)
            if m in ("inc", "dec"):
                cf = st.nzcv[2]                            # INC and DEC keep CF
            if m != "cmp":
                self.write(st, self.dst(ins), res)
            st.nzcv = self.flags_res(res, cf, of); return

        if m in ("and", "or", "xor", "test", "andn"):
            if m == "andn":
                a, b = [x ^ 1 for x in self.val(ins, st, 1)], self.val(ins, st, 2)
            else:
                a, b = self.val(ins, st, 0), self.val(ins, st, 1, w)
            op = {"and": g.AND, "andn": g.AND, "test": g.AND, "or": g.OR, "xor": g.XOR}[m]
            res = [op(x, y) for x, y in zip(a, b)]
            if m != "test":
                self.write(st, self.dst(ins), res)
            st.nzcv = self.flags_res(res); return
        if m == "not":
            self.write(st, self.dst(ins), [x ^ 1 for x in self.val(ins, st, 0)]); return

        if m in ("imul", "mul") and len(ops) == 1:      # RDX:RAX = RAX * src (and the 8-bit form: AX = AL * src)
            b = self.val(ins, st, 0)
            a = st.x[0][:w]
            signed = m == "imul"
            ext = lambda v: v + [v[-1] if signed else FALSE] * w
            p = g.mul(ext(a), ext(b), 2 * w)
            lo, hi = p[:w], p[w:]
            if w == 8:
                self.write(st, "ax", p)
            else:
                self.write(st, {16: "ax", 32: "eax", 64: "rax"}[w], lo)
                self.write(st, {16: "dx", 32: "edx", 64: "rdx"}[w], hi)
            top = [lo[-1]] * w if signed else [FALSE] * w
            over = g.OR_all([g.XOR(x, y) for x, y in zip(hi, top)])
            st.nzcv = (lo[-1], g.eq_zero(lo), over, over, self.parity(lo)); return
        if m == "imul":
            if len(ops) == 2:
                a, b = self.val(ins, st, 0), self.val(ins, st, 1, w)
            else:
                a, b = self.val(ins, st, 1), self.val(ins, st, 2, w)
            ext = lambda v: v + [v[-1]] * w
            p = g.mul(ext(a), ext(b), 2 * w)
            lo = p[:w]
            over = g.OR_all([g.XOR(x, lo[-1]) for x in p[w:]])
            self.write(st, self.dst(ins), lo)
            st.nzcv = (lo[-1], g.eq_zero(lo), over, over, self.parity(lo)); return
        if m in ("div", "idiv"):                        # RDX:RAX / src: quotient in RAX, remainder in RDX
            b = self.val(ins, st, 0)
            if w == 8:
                num = st.x[0][:16]
            else:
                num = st.x[0][:w] + st.x[2][:w]
            signed = m == "idiv"
            bb = b + [b[-1] if signed else FALSE] * (len(num) - w)
            if signed:
                sa, sb = num[-1], bb[-1]
                ua = [g.MUX(sa, x, y) for x, y in zip(g.neg(num), num)]
                ub = [g.MUX(sb, x, y) for x, y in zip(g.neg(bb), bb)]
                q, r = g.udivmod(ua, ub)
                q = [g.MUX(g.XOR(sa, sb), x, y) for x, y in zip(g.neg(q), q)]
                r = [g.MUX(sa, x, y) for x, y in zip(g.neg(r), r)]
            else:
                q, r = g.udivmod(num, bb)
            if w == 8:
                self.write(st, "al", q[:8])
                self.write(st, "ah", r[:8])
            else:
                self.write(st, {16: "ax", 32: "eax", 64: "rax"}[w], q[:w])
                self.write(st, {16: "dx", 32: "edx", 64: "rdx"}[w], r[:w])
            return                                      # a quotient too large is #DE on the CPU: verification sees it

        if m in ("shl", "sal", "shr", "sar", "rol", "ror"):
            a = self.val(ins, st, 0)
            kind = {"shl": "lsl", "sal": "lsl", "shr": "lsr", "sar": "asr", "ror": "ror", "rol": "rol"}[m]
            mask = 63 if w == 64 else 31
            if len(ops) == 1:
                count = g.const(1, 8)
            else:
                count = self.val(ins, st, 1, 8)
            n = g.value(count)
            if n is not None:
                n &= mask
                if n == 0:
                    return                               # flags unchanged
                r = self._shift(a, n, kind)
                if kind == "lsl":
                    cf = a[w - n] if n <= w else FALSE
                elif kind in ("lsr", "asr"):
                    cf = a[n - 1] if n <= w else (a[-1] if kind == "asr" else FALSE)
                else:
                    cf = r[0] if kind == "rol" else r[-1]
                of = {"lsl": g.XOR(r[-1], cf), "lsr": a[-1], "asr": FALSE}.get(kind, FALSE) if n == 1 else FALSE
                self.write(st, self.dst(ins), r)
                if kind in ("rol", "ror"):
                    st.nzcv = self.keep(st, cf=cf, of=of)
                else:
                    st.nzcv = self.flags_res(r, cf, of)
                return
            amt = count[:mask.bit_length()]
            if kind == "rol":
                r = g.shift_var(a, g.neg(amt + [FALSE] * (w - len(amt)))[:w], "ror")
            else:
                r = g.shift_var(a, amt + [FALSE] * (w - len(amt)), kind) if kind == "ror" else self._shift_var(a, amt, kind)
            zero = g.eq_zero(amt)
            self.write(st, self.dst(ins), r)
            if kind not in ("rol", "ror"):                # a count of 0 keeps the flags; CF and OF approximated
                new = self.flags_res(r)
                st.nzcv = tuple(g.MUX(zero, old, nw) for old, nw in zip(st.nzcv, new))
            return
        if m in ("shlx", "shrx", "sarx", "rorx"):
            a = self.val(ins, st, 1)
            kind = {"shlx": "lsl", "shrx": "lsr", "sarx": "asr", "rorx": "ror"}[m]
            mask = 63 if w == 64 else 31
            if m == "rorx":
                r = self._shift(a, ops[2].imm & mask, "ror")
            else:
                r = self._shift_var(a, self.val(ins, st, 2)[:mask.bit_length()], kind)
            self.write(st, self.dst(ins), r); return

        if m.startswith("set") and _cc(m, "set"):
            c = self.cond(st, _cc(m, "set"))
            self.write(st, self.dst(ins), [c] + [FALSE] * 7); return
        if m.startswith("cmov") and _cc(m, "cmov"):
            c = self.cond(st, _cc(m, "cmov"))
            a, b = self.val(ins, st, 1, w), self.val(ins, st, 0)
            self.write(st, self.dst(ins), [g.MUX(c, x, y) for x, y in zip(a, b)]); return

        if m == "popcnt":
            a = self.val(ins, st, 1)
            cnt = g.const(0, w)
            for b in a:
                cnt = g.add(cnt, [b] + [FALSE] * (w - 1))[0]
            self.write(st, self.dst(ins), cnt)
            st.nzcv = (FALSE, g.eq_zero(a), FALSE, FALSE, FALSE); return
        if m in ("lzcnt", "tzcnt"):
            a = self.val(ins, st, 1)
            seq = a[::-1] if m == "lzcnt" else a
            cnt, allz = g.const(0, w), TRUE
            for b in seq:
                allz = g.AND(allz, b ^ 1)
                cnt = g.add(cnt, [allz] + [FALSE] * (w - 1))[0]
            self.write(st, self.dst(ins), cnt)
            st.nzcv = self.keep(st, zf=g.eq_zero(cnt), cf=g.eq_zero(a)); return
        if m in ("bsr", "bsf"):                          # a zero source leaves the destination as it was
            a = self.val(ins, st, 1)
            idx = g.const(0, w)
            found = FALSE
            order = range(w - 1, -1, -1) if m == "bsr" else range(w)
            for i in order:
                hit = g.AND(a[i], found ^ 1)
                idx = [g.MUX(hit, x, y) for x, y in zip(g.const(i, w), idx)]
                found = g.OR(found, a[i])
            old = self.val(ins, st, 0)
            self.write(st, self.dst(ins), [g.MUX(found, x, y) for x, y in zip(idx, old)])
            st.nzcv = self.keep(st, zf=found ^ 1); return
        if m == "bt":
            a = self.val(ins, st, 0)
            if ops[1].type == I:
                bit = a[ops[1].imm % w]
            else:
                bit = self._shift_var(a, self.val(ins, st, 1)[:(w - 1).bit_length()], "lsr")[0]
            st.nzcv = self.keep(st, cf=bit); return

        raise Unsupported(f"instruction not supported yet {ins.where()}")

    def _shift(self, a, n, kind):
        w = len(a)
        if kind == "rol":
            return self.g.shift_const(a, (w - n) % w, "ror")
        if n >= w:
            return [a[-1] if kind == "asr" else FALSE] * w
        return self.g.shift_const(a, n, kind)

    def _shift_var(self, a, amount, kind):
        w = len(a)
        r = list(a)
        for j, bit in enumerate(amount):
            s = 1 << j
            shifted = self._shift(r, s, kind)
            r = [self.g.MUX(bit, x, y) for x, y in zip(shifted, r)]
        return r

    def branch(self, ins, st):
        if ins.is_ret:
            return "ret"
        if ins.uncond:
            return TRUE
        return self.cond(st, ins.cc)
