# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Quantag IT Solutions GmbH
"""Invert the AES S-box with Grover search: the phase oracle is compiled from examples/aes_sbox.c,
whose S-box is a 256-byte table (a quantum ROM in the circuit). 12 iterations for 256 inputs."""
import math, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from qoracle import build, sim

src = os.path.join(ROOT, "examples", "aes_sbox.c")
full = build(src, "sbox", ["x:8"])
print("S-box oracle   :", {k: full.report[k] for k in ("qubits", "toffoli", "cnot", "rom_bits")})
o = build(src, "preimage", ["x:8", str(0xED)], mode="phase")
print("preimage oracle:", {k: o.report[k] for k in ("qubits", "toffoli", "cnot", "rom_bits")})
k = round(math.pi / 4 * math.sqrt(256))
probs = sim.grover(o.circuit, "x", iterations=k)
best = max(probs, key=probs.get)
print(f"after {k} Grover iterations: x = 0x{best:02x} with probability {probs[best]:.4f}  (SBOX[0x53] = 0xED)")
