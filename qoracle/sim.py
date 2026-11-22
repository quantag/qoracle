# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Quantag IT Solutions GmbH
"""A small exact simulator for oracles in quantum algorithms.

The state is a dictionary {basis state: amplitude}. The oracle's gates (X, CNOT, Toffoli, Z) map
basis states to basis states, so the dictionary only ever holds as many entries as the input
register has values in superposition, however many work qubits the oracle uses.
"""
import math


def apply(state, gates):
    out = state
    for gt in gates:
        op = gt[0]
        if op == "x":
            m = 1 << gt[1]
            out = {s ^ m: a for s, a in out.items()}
        elif op == "cx":
            c, m = gt[1], 1 << gt[2]
            out = {(s ^ m) if (s >> c) & 1 else s: a for s, a in out.items()}
        elif op == "ccx":
            c1, c2, m = gt[1], gt[2], 1 << gt[3]
            out = {(s ^ m) if (s >> c1) & 1 and (s >> c2) & 1 else s: a for s, a in out.items()}
        elif op == "z":
            q = gt[1]
            out = {s: -a if (s >> q) & 1 else a for s, a in out.items()}
        elif op == "gphase":
            out = {s: -a for s, a in out.items()}
    return out


def register(circuit, name):
    for n, first, size in circuit.regs:
        if n == name:
            return first, size
    raise KeyError(name)


def uniform(circuit, name):
    """All values of input register `name` in equal superposition, every other qubit 0."""
    first, size = register(circuit, name)
    a = 1 / math.sqrt(1 << size)
    return {v << first: a for v in range(1 << size)}


def diffuse(state, circuit, name):
    """Grover's diffusion 2|s><s| - 1 on register `name` (the other qubits are a fixed pattern)."""
    first, size = register(circuit, name)
    mask = ((1 << size) - 1) << first
    groups = {}
    for s, a in state.items():
        groups.setdefault(s & ~mask, {})[(s & mask) >> first] = a
    out = {}
    for rest, amps in groups.items():
        mean = sum(amps.values()) / (1 << size)
        for v in range(1 << size):
            a = 2 * mean - amps.get(v, 0)
            if abs(a) > 1e-15:
                out[rest | (v << first)] = a
    return out


def grover(circuit, name, iterations=None):
    """Run Grover's algorithm with a phase oracle on input register `name`; returns {value: probability}."""
    if circuit.mode != "phase":
        raise ValueError("Grover needs a phase oracle (mode='phase')")
    first, size = register(circuit, name)
    st = uniform(circuit, name)
    if iterations is None:
        iterations = max(1, int(math.pi / 4 * math.sqrt(1 << size)))
    for _ in range(iterations):
        st = apply(st, circuit.gates)
        st = diffuse(st, circuit, name)
    probs = {}
    for s, a in st.items():
        v = (s >> first) & ((1 << size) - 1)
        probs[v] = probs.get(v, 0) + abs(a) ** 2
    return probs


def period_finding(circuit, name):
    """The quantum part of Shor's algorithm with a compute oracle |x>|0> -> |x>|f(x)>: register `name`
    in uniform superposition, the oracle, then the quantum Fourier transform on `name`.
    Returns {measured value of `name`: probability}."""
    if circuit.mode != "compute":
        raise ValueError("period finding needs a compute oracle (mode='compute')")
    first, size = register(circuit, name)
    N = 1 << size
    st = apply(uniform(circuit, name), circuit.gates)
    mask = (N - 1) << first
    groups = {}
    for s, a in st.items():
        groups.setdefault(s & ~mask, {})[(s & mask) >> first] = a
    probs = {}
    for amps in groups.values():                       # QFT on the register, for each value of f(x)
        for k in range(N):
            a = sum(v * complex(math.cos(2 * math.pi * x * k / N), math.sin(2 * math.pi * x * k / N))
                    for x, v in amps.items()) / math.sqrt(N)
            probs[k] = probs.get(k, 0) + abs(a) ** 2
    return probs
