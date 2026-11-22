# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Quantag IT Solutions GmbH
"""Shor's algorithm with an oracle compiled from examples/modexp.c (a^x mod n, ordinary C).

The quantum part (superposition, the oracle, the quantum Fourier transform) is simulated exactly;
the classical part (continued fractions, gcd) is ordinary Python.
Run: python3 examples/shor.py [n a]      default: 15 7, then 21 2"""
import math, os, random, sys
from fractions import Fraction
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from qoracle import build, sim


def shor(n, a, bits=8, shots=10, seed=1):
    o = build(os.path.join(ROOT, "examples", "modexp.c"), "modexp", [str(a), f"x:{bits}", str(n)],
              out_bits=n.bit_length())
    r = o.report
    print(f"n = {n}, a = {a}: oracle {r['qubits']} qubits, {r['toffoli']} Toffoli, {r['gates']} gates, "
          f"verified on all {r['verification']['inputs_tested']} inputs")
    probs = sim.period_finding(o.circuit, "x")
    rng = random.Random(seed)
    values, weights = zip(*sorted(probs.items()))
    for k in rng.choices(values, weights, k=shots):
        period = Fraction(k, 1 << bits).limit_denominator(n).denominator
        if pow(a, period, n) != 1 or period % 2:
            print(f"  measured {k:3d} -> candidate period {period}: not usable, measure again")
            continue
        p, q = math.gcd(pow(a, period // 2) - 1, n), math.gcd(pow(a, period // 2) + 1, n)
        if 1 < p < n:
            print(f"  measured {k:3d} -> period {period} -> {n} = {p} x {n // p}")
            return p
        print(f"  measured {k:3d} -> period {period}: trivial factors, measure again")


if __name__ == "__main__":
    runs = [(int(sys.argv[1]), int(sys.argv[2]))] if len(sys.argv) > 2 else [(15, 7), (21, 2)]
    for n, a in runs:
        shor(n, a)
