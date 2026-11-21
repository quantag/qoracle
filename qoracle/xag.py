# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Quantag IT Solutions GmbH
"""XOR-AND graph (XAG): the logic network a classical function is lowered to before it becomes gates.

A literal is 2 * node + negated. Node 0 is the constant false, so literal 0 is false and literal 1 is
true. Every other node is an input bit, an AND of two literals, or an XOR of two literals. Nodes are
hash-consed (equal expressions share one node) and simplified on construction (constants, x & x,
x ^ x, ...), so constant folding and common-subexpression elimination happen as the graph is built.

On top of single bits, words are lists of literals, least significant bit first; this module also
has the word operations a processor needs: addition with flags, multiplication, division, shifts.
In a reversible circuit each AND costs one Toffoli gate (and one more to uncompute it); XOR is cheap.
"""

FALSE, TRUE = 0, 1


class XAG:
    def __init__(self):
        self.kind = ["const"]          # per node: const | in | and | xor
        self.a = [0]
        self.b = [0]
        self.name = [None]             # input nodes: (variable, bit)
        self._table = {}

    # ------------------------------------------------------------------ bits
    def input(self, name):
        self.kind.append("in"); self.a.append(0); self.b.append(0); self.name.append(name)
        return 2 * (len(self.kind) - 1)

    def _node(self, kind, x, y):
        key = (kind, x, y)
        n = self._table.get(key)
        if n is None:
            self.kind.append(kind); self.a.append(x); self.b.append(y); self.name.append(None)
            n = self._table[key] = len(self.kind) - 1
        return 2 * n

    def AND(self, x, y):
        if x > y:
            x, y = y, x
        if x == FALSE:
            return FALSE
        if x == TRUE:
            return y
        if x == y:
            return x
        if x ^ 1 == y:
            return FALSE
        return self._node("and", x, y)

    def XOR(self, x, y):
        neg = (x ^ y) & 1              # the negations move to the output: inputs of an XOR node are plain
        x, y = x & ~1, y & ~1
        if x > y:
            x, y = y, x
        if x == FALSE:
            return y ^ neg
        if x == y:
            return neg
        return self._node("xor", x, y) ^ neg

    @staticmethod
    def NOT(x):
        return x ^ 1

    def OR(self, x, y):
        return self.AND(x ^ 1, y ^ 1) ^ 1

    def MUX(self, c, a, b):
        """c ? a : b"""
        if a == b:
            return a
        return self.XOR(b, self.AND(c, self.XOR(a, b)))

    def OR_all(self, bits):
        bits = list(bits)
        while len(bits) > 1:          # a balanced tree: lower depth
            bits = [self.OR(bits[i], bits[i + 1]) if i + 1 < len(bits) else bits[i] for i in range(0, len(bits), 2)]
        return bits[0] if bits else FALSE

    def XOR_all(self, bits):
        r = FALSE
        for b in bits:
            r = self.XOR(r, b)
        return r

    # ------------------------------------------------------------------ words (LSB first)
    @staticmethod
    def const(v, w):
        return [TRUE if (v >> i) & 1 else FALSE for i in range(w)]

    def inputs(self, name, bits, w, signed=False):
        x = [self.input((name, i)) for i in range(bits)]
        fill = x[-1] if signed and x else FALSE
        return (x + [fill] * w)[:w]

    @staticmethod
    def value(word):
        """The integer value of a constant word, or None."""
        if any(b not in (FALSE, TRUE) for b in word):
            return None
        return sum(b << i for i, b in enumerate(word))

    def add(self, a, b, cin=FALSE):
        """a + b + cin. Returns (sum, carry out, carry into the top bit)."""
        s, c, c_top = [], cin, FALSE
        for i, (x, y) in enumerate(zip(a, b)):
            if i == len(a) - 1:
                c_top = c
            p = self.XOR(x, y)
            s.append(self.XOR(p, c))
            c = self.XOR(self.AND(x, y), self.AND(c, p))    # the two terms never both hold
        return s, c, c_top

    def sub(self, a, b, borrow_in=TRUE):
        """a - b as a + ~b + 1 (ARM: carry = NOT borrow)."""
        return self.add(a, [x ^ 1 for x in b], borrow_in)

    def neg(self, a):
        return self.sub(self.const(0, len(a)), a)[0]

    def mul(self, a, b, w=None):
        """a * b truncated to w bits (schoolbook; partial products with constant bits vanish)."""
        w = w or len(a)
        acc = self.const(0, w)
        for i in range(min(w, len(b))):
            if b[i] == FALSE:
                continue
            row = [FALSE] * i + [self.AND(a[j], b[i]) for j in range(w - i)]
            acc = self.add(acc, row[:w])[0]
        return acc

    def eq_zero(self, a):
        return self.OR_all(a) ^ 1

    def ult(self, a, b):
        """a < b, unsigned"""
        return self.sub(a, b)[1] ^ 1

    def udivmod(self, a, b):
        """Restoring division, w bits. Division by zero gives quotient 0 and remainder a (AArch64 udiv
        returns 0; the remainder is then a - 0 * b = a)."""
        w = len(a)
        k = self.value(b)
        if k is not None and k and k & (k - 1) == 0:          # a power of two: wiring
            s = k.bit_length() - 1
            return a[s:] + [FALSE] * s, a[:s] + [FALSE] * (w - s)
        r = self.const(0, w)
        q = [FALSE] * w
        for i in range(w - 1, -1, -1):
            top = r[-1]                                          # the bit shifted out: r << 1 can need w + 1 bits
            r = [a[i]] + r[:-1]                                  # r = r << 1 | a[i]
            d, carry, _ = self.sub(r, b)
            ok = self.OR(carry, top)                             # r >= b
            q[i] = ok
            r = [self.MUX(ok, d[j], r[j]) for j in range(w)]
        z = self.eq_zero(b)
        q = [self.AND(x, z ^ 1) for x in q]
        r = [self.MUX(z, a[j], r[j]) for j in range(w)]
        return q, r

    def sdiv(self, a, b):
        """Signed division truncating towards zero; by zero gives 0 (AArch64)."""
        w = len(a)
        sa, sb = a[-1], b[-1]
        ua = [self.MUX(sa, x, y) for x, y in zip(self.neg(a), a)]
        ub = [self.MUX(sb, x, y) for x, y in zip(self.neg(b), b)]
        q, _ = self.udivmod(ua, ub)
        s = self.XOR(sa, sb)
        nq = self.neg(q)
        return [self.MUX(s, x, y) for x, y in zip(nq, q)][:w]

    def shift_const(self, a, k, kind):
        """lsl | lsr | asr | ror by a constant."""
        w = len(a)
        if kind == "ror":
            k %= w
            return a[k:] + a[:k]
        if kind == "lsl":
            return ([FALSE] * k + a)[:w] if k < w else [FALSE] * w
        if kind == "lsr":
            return a[k:] + [FALSE] * k if k < w else [FALSE] * w
        if kind == "asr":
            k = min(k, w - 1)
            return a[k:] + [a[-1]] * k
        raise ValueError(kind)

    def shift_var(self, a, amount, kind):
        """Shift by a variable amount, taken modulo the width (AArch64 lslv/lsrv/asrv/rorv)."""
        w = len(a)
        r = list(a)
        for j in range((w - 1).bit_length()):
            s = 1 << j
            shifted = self.shift_const(r, s, kind)
            r = [self.MUX(amount[j], x, y) for x, y in zip(shifted, r)]
        return r
