# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Quantag IT Solutions GmbH
"""From a logic network to a reversible circuit: gates, uncomputation, simulation, export.

Every AND node reachable from the outputs gets a fresh qubit, set by one Toffoli; every XOR node a
fresh qubit, set by two CNOTs (Bennett's construction, without qubit reuse yet). The outputs are
copied out (compute oracle) or their phase is flipped (phase oracle), then all gates are undone in
reverse order, so every work qubit returns to 0 and the inputs are unchanged.

Gates: ("x", q), ("cx", c, t), ("ccx", c1, c2, t), ("z", q). A negated control is an X before and
after the gate.
"""
from .xag import FALSE, TRUE

# names an OpenQASM 3 register cannot have: gates of stdgates.inc and keywords
RESERVED = set("""p x y z h s sdg t tdg sx rx ry rz cx cy cz cp crx cry crz ch swap ccx cswap cu CX phase cphase id u1 u2
u3 U gphase measure reset barrier qubit bit int uint float angle bool const input output gate def if else for while in
return break continue box let delay stretch duration pi tau euler im ctrl negctrl inv pow include defcal cal extern
OPENQASM qreg creg out anc check""".split())


def qasm_name(name):
    return f"{name}_in" if name in RESERVED - {"out", "anc", "check"} else name


class Circuit:
    def __init__(self, g, inputs, outputs, mode="compute", extra=()):
        """inputs: {variable: [literal per bit]} (input literals of the XAG); outputs: [literal]
        (compute mode: one output qubit each; phase mode: exactly one literal, whose phase is flipped);
        extra: further literals given output qubits (verification of the loop bound)."""
        self.mode = mode
        self.inputs = {k: list(v) for k, v in inputs.items()}
        self.outputs = list(outputs)
        need = set()
        stack = [l >> 1 for l in list(outputs) + list(extra)]
        while stack:
            n = stack.pop()
            if n in need or n == 0:
                continue
            need.add(n)
            if g.kind[n] in ("and", "xor"):
                stack += [g.a[n] >> 1, g.b[n] >> 1]
        # XOR folding: an XOR node used only once, by another XOR node, gets no qubit of its own; the
        # XOR node above it CNOTs the folded node's operands straight into its own qubit. A quantum ROM's
        # value bit (the XOR of many decoder lines) then costs one qubit instead of one per XOR.
        uses = {}
        for n in need:
            if g.kind[n] in ("and", "xor"):
                for c in (g.a[n] >> 1, g.b[n] >> 1):
                    uses.setdefault(c, []).append(n)
        pinned = {l >> 1 for l in list(outputs) + list(extra)}
        folded = {n for n in need if g.kind[n] == "xor" and n not in pinned and len(uses.get(n, ())) == 1
                  and g.kind[uses[n][0]] == "xor"}
        self.n_folded = len(folded)
        self.q = {}                                   # XAG node -> qubit
        self.regs = []                                # (name, first qubit, size)
        nq = 0
        for name, lits in self.inputs.items():
            self.regs.append((name, nq, len(lits)))
            for l in lits:
                self.q[l >> 1] = nq
                nq += 1
        self.n_in = nq
        self.n_out = len(outputs) if mode == "compute" else 0
        self.out0 = nq
        if self.n_out:
            self.regs.append(("out", nq, self.n_out))
            nq += self.n_out
        self.extra0, self.n_extra = nq, len(extra)
        if extra:
            self.regs.append(("check", nq, len(extra)))
            nq += len(extra)
        self.anc0 = nq
        compute = []
        self.n_and = self.n_xor = 0
        for n in sorted(need):                        # node ids are a topological order
            k = g.kind[n]
            if n in folded:
                continue
            if k == "in":
                if n not in self.q:
                    raise ValueError(f"input bit {g.name[n]} is not part of the circuit's inputs")
                continue
            t = self.q[n] = nq
            nq += 1
            if k == "and":
                self.n_and += 1
                compute += self._ccx(g.a[n], g.b[n], t)
            else:
                self.n_xor += 1
                leaves, flip, stack = {}, 0, [g.a[n], g.b[n]]
                while stack:                          # the parity of the folded XOR tree below n
                    l = stack.pop()
                    flip ^= l & 1
                    c = l >> 1
                    if c in folded:
                        stack += [g.a[c], g.b[c]]
                    elif c:
                        leaves[c] = leaves.get(c, 0) ^ 1
                compute += [("cx", self.q[c], t) for c, odd in leaves.items() if odd] + [("x", t)] * flip
        self.n_anc = nq - self.anc0
        self.nqubits = nq
        middle = []
        if mode == "compute":
            for i, l in enumerate(outputs):
                middle += self._copy(l, self.out0 + i)
        elif mode == "phase":
            if len(outputs) != 1:
                raise ValueError("a phase oracle needs a one-bit output")
            l = outputs[0]
            if l in (FALSE, TRUE):
                middle = [] if l == FALSE else [("gphase",)]
            else:
                q = self.q[l >> 1]
                middle = ([("x", q)] if l & 1 else []) + [("z", q)] + ([("x", q)] if l & 1 else [])
        else:
            raise ValueError(mode)
        for i, l in enumerate(extra):
            middle += self._copy(l, self.extra0 + i)
        self.compute = compute
        self.gates = compute + middle + compute[::-1]

    def _ccx(self, a, b, t):
        qa, qb = self.q[a >> 1], self.q[b >> 1]
        pre = [("x", qa)] * (a & 1) + [("x", qb)] * (b & 1)
        return pre + [("ccx", qa, qb, t)] + pre[::-1]

    def _copy(self, l, t):
        if l == FALSE:
            return []
        if l == TRUE:
            return [("x", t)]
        return [("cx", self.q[l >> 1], t)] + ([("x", t)] if l & 1 else [])

    # ------------------------------------------------------------------ metrics
    def counts(self):
        c = {}
        for gt in self.gates:
            c[gt[0]] = c.get(gt[0], 0) + 1
        c.pop("gphase", None)
        return c

    def depth(self):
        level = [0] * self.nqubits
        for gt in self.gates:
            if gt[0] == "gphase":
                continue
            qs = gt[1:]
            d = max(level[q] for q in qs) + 1
            for q in qs:
                level[q] = d
        return max(level, default=0)

    def report(self):
        c = self.counts()
        tof = c.get("ccx", 0)
        return dict(qubits=self.nqubits, inputs=self.n_in, outputs=self.n_out, ancillas=self.n_anc,
                    and_nodes=self.n_and, xor_nodes=self.n_xor, xor_folded=self.n_folded, gates=sum(c.values()), toffoli=tof,
                    cnot=c.get("cx", 0), x=c.get("x", 0), z=c.get("z", 0), t_count=7 * tof, depth=self.depth())

    # ------------------------------------------------------------------ bit-parallel simulation
    def simulate(self, lanes, values):
        """Run the circuit on `lanes` classical inputs at once. values: {variable: [int per lane]}.
        Each qubit is a Python int whose bit j is its value in lane j. Returns the qubit list."""
        q = [0] * self.nqubits
        for name, first, size in self.regs:
            if name in values:
                for b in range(size):
                    m = 0
                    for j, v in enumerate(values[name]):
                        m |= ((v >> b) & 1) << j
                    q[first + b] = m
        ones = (1 << lanes) - 1
        for gt in self.gates:
            op = gt[0]
            if op == "x":
                q[gt[1]] ^= ones
            elif op == "cx":
                q[gt[2]] ^= q[gt[1]]
            elif op == "ccx":
                q[gt[3]] ^= q[gt[1]] & q[gt[2]]
        return q

    @staticmethod
    def lane_values(q, first, size, lanes):
        return [sum(((q[first + b] >> j) & 1) << b for b in range(size)) for j in range(lanes)]

    # ------------------------------------------------------------------ export
    def qasm3(self, title=""):
        names = {}
        L = ["OPENQASM 3.0;", 'include "stdgates.inc";']
        if title:
            L.insert(0, f"// {title}")
        for name, first, size in self.regs:
            reg = qasm_name(name)
            L.append(f"qubit[{size}] {reg};")
            for b in range(size):
                names[first + b] = f"{reg}[{b}]"
        if self.n_anc:
            L.append(f"qubit[{self.n_anc}] anc;")
            for b in range(self.n_anc):
                names[self.anc0 + b] = f"anc[{b}]"
        for gt in self.gates:
            if gt[0] == "gphase":
                L.append("gphase(pi);")
            else:
                L.append(f"{gt[0]} " + ", ".join(names[q] for q in gt[1:]) + ";")
        return "\n".join(L) + "\n"

    def to_qiskit(self, name="oracle"):
        from qiskit import QuantumCircuit, QuantumRegister
        import math
        regs = [QuantumRegister(size, n) for n, _, size in self.regs]
        if self.n_anc:
            regs.append(QuantumRegister(self.n_anc, "anc"))
        qc = QuantumCircuit(*regs, name=name)
        for gt in self.gates:
            op = gt[0]
            if op == "x":
                qc.x(gt[1])
            elif op == "z":
                qc.z(gt[1])
            elif op == "cx":
                qc.cx(gt[1], gt[2])
            elif op == "ccx":
                qc.ccx(gt[1], gt[2], gt[3])
            elif op == "gphase":
                qc.global_phase += math.pi
        return qc
