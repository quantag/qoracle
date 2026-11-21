# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Quantag IT Solutions GmbH
"""Experiment 1: oracles compiled by qoracle against the same functions run on the quantum von Neumann
interpreter of cpu-qpu (github.com/quantag/cpu-qpu). The interpreter numbers are slightly low: the
marking step between U and U^-1 is not counted.

Both start from the same C code compiled by clang for AArch64, and both are exact for the inputs
used. The interpreter fetches, decodes and executes the machine code with gates in every cycle; qoracle
translates the machine code once into a circuit for that function only. Both counts are complete
oracles: compute, mark (phase flip or copy of the result), uncompute.

Interpreter numbers were measured with cpu-qpu at commit d26eb5e:
    python3 grover.py examples/grover_square.c --entry is_root --bits 4
        machine: W = 9, 7 instruction classes, 8 program slots, 7 cycles:
        1242 qubits, 49252 gates (28994 Toffoli) per run of U; the oracle runs U, phase flip, U^-1
    python3 shor.py --N 15 --a 7
        machine: W = 9, 12 instruction classes, 32 program slots, 101 cycles:
        6588 qubits, 2445210 gates (1145340 Toffoli) per run of U;
        circuit: H on x, U, copy, U^-1, inverse QFT: 6592 qubits, 4890480 gates
Run: python3 experiments/vs_interpreter.py
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from qoracle import build

INTERPRETER = {
    # function: (qubits, gates, Toffoli) of the complete oracle: two runs of U (forward and backward); the few
    # gates of the marking step (phase flip or copy) are not counted. Qubits: the circuit as measured.
    "is_root": (1242, 2 * 49252, 2 * 28994),
    "modexp": (6592, 2 * 2445210, 2 * 1145340),
}


def main():
    rows = []
    o = build(os.path.join(ROOT, "examples", "is_root.c"), "is_root", ["x:4"], mode="phase")
    rows.append(("is_root", "Grover oracle, x: 4 bits, phase", o.report))
    o = build(os.path.join(ROOT, "examples", "modexp.c"), "modexp", ["7", "x:8", "15"], out_bits=4)
    rows.append(("modexp", "Shor, 7^x mod 15, x: 8 bits, 4-bit result", o.report))
    print(f"{'function':10s} {'':44s} {'qubits':>10s} {'gates':>12s} {'Toffoli':>12s}")
    for name, what, r in rows:
        iq, ig, it = INTERPRETER[name]
        print(f"{name:10s} {what:44s}")
        print(f"{'':10s} {'  interpreter (cpu-qpu)':44s} {iq:>10,} {ig:>12,} {it:>12,}")
        print(f"{'':10s} {'  qoracle':44s} {r['qubits']:>10,} {r['gates']:>12,} {r['toffoli']:>12,}")
        print(f"{'':10s} {'  ratio':44s} {iq / r['qubits']:>9.0f}x {ig / r['gates']:>11.0f}x {it / r['toffoli']:>11.0f}x")
        v = r["verification"]
        print(f"{'':10s}   qoracle verified on {'all ' if v['exhaustive'] else ''}{v['inputs_tested']} inputs against the AArch64 CPU")
    return rows


if __name__ == "__main__":
    main()
