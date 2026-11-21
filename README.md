# qoracle: quantum oracles from ordinary code

qoracle turns a classical function, written in C or given as compiled AArch64 machine code, into a
reversible circuit for a quantum computer: a *quantum oracle*. Every circuit is checked against the
real machine code running on a CPU emulator before it is written out.

```
$ qoracle examples/is_root.c is_root x:4 --mode phase -o is_root.qasm
is_root(x: 4 bits)  phase oracle, 1-bit result
  machine code   AArch64, 6 instructions, 1 block
  logic network  14 AND, 12 XOR
  circuit        30 qubits (4 in, 0 out, 26 work), 85 gates: 28 Toffoli, 48 CNOT, 9 single-qubit; depth 43
  T count        196 (7 per Toffoli)
  verified       all 16 inputs equal to the AArch64 CPU emulator (Unicorn); work qubits return to 0
  written        is_root.qasm (OpenQASM 3)
```

## Why

Most quantum algorithms are a short quantum skeleton around a classical function that must be
evaluated on all inputs at once: the oracle of Grover's search, the modular exponentiation of Shor's
algorithm, the predicate of amplitude amplification. The skeleton is textbook material; the oracle is
usually the largest part of the circuit, and it is still mostly written by hand, gate by gate or in a
special language. qoracle starts from the code that already exists, and its output can be trusted
because it is compared with the processor.

## Quick start

Needs Python 3.10+, clang with the AArch64 target (`clang --target=aarch64-linux-gnu`), and the Python
packages in `pyproject.toml` (capstone, unicorn, lief; qiskit only for `to_qiskit()`).

```bash
pip install -e .
qoracle examples/is_root.c is_root x:4 --mode phase -o is_root.qasm      # Grover oracle
qoracle examples/modexp.c modexp 7 x:8 15 --out-bits 4 -o modexp.qasm    # Shor: |x>|0> -> |x>|7^x mod 15>
python3 -m qoracle ...                                                    # without installing
```

Arguments follow the C parameters in order: `name:bits` is a quantum input (`name:bitss` signed), a
number is a constant. The result is the return value.

```python
from qoracle import build, sim
o = build("examples/is_root.c", "is_root", ["x:4"], mode="phase")
o.report                          # qubits, gates, Toffoli and T count, depth, verification
o.circuit.qasm3()                 # OpenQASM 3 text (loads in Qiskit with qiskit.qasm3.loads)
o.circuit.to_qiskit()             # a Qiskit QuantumCircuit
sim.grover(o.circuit, "x")        # Grover's search with this oracle: {x: probability}
```

## How it works

```
C source ─clang─> AArch64 machine code ─> symbolic execution ─> XOR-AND network ─> reversible circuit ─> OpenQASM 3
                         │                 (all paths at once,     (constants folded,    (Toffoli per AND,     Qiskit
                         │                  loops unrolled)         shared subterms)      uncomputation)
                         └──────────── CPU emulator ◄──── verification: every test input, circuit == machine code
```

1. **Compile** the C file with clang for AArch64 (or take an object file).
2. **Decode** the function's machine code (capstone) and split it into basic blocks.
3. **Execute symbolically**: every instruction's exact architectural meaning (add with flags,
   multiply, divide, shifts, bit fields, conditional select, ...) applied to symbolic bits. A circuit
   has no jumps, so all paths run at once: each block carries an activity predicate, branches split
   it, paths are merged where they meet, and loops are unrolled. The loop bound is measured by running
   the code on the test inputs.
4. **Logic network**: the result is an XOR-AND graph. Constants are folded and equal subterms are
   shared while it is built, so the compiler's own optimizations and the constant arguments shrink
   the circuit (in `modexp`, multiplying by the constant 7 and dividing by 15).
5. **Circuit**: one Toffoli per AND node, two CNOTs per XOR node, into fresh qubits; then the result
   is copied out (compute oracle) or its phase flipped (phase oracle), and everything is undone, so
   all work qubits return to 0.
6. **Verify**: the circuit is simulated on every input when there are at most 2^16, otherwise on edge
   values and 4,096 random inputs, all at once (one bit per input in a machine word), and compared
   with the machine code run in the Unicorn CPU emulator. Work qubits must return to 0 and inputs must
   be unchanged.

## First results

The same two functions as in the quantum von Neumann interpreter of
[cpu-qpu](https://github.com/quantag/cpu-qpu), which runs the machine code by fetching and decoding
it with gates in every cycle (`experiments/vs_interpreter.py`):

| oracle | | qubits | gates | Toffoli |
|---|---|---:|---:|---:|
| `is_root` (Grover, x: 4 bits) | interpreter | 1,242 | 98,504 | 57,988 |
| | qoracle | **30** | **85** | **28** |
| `modexp` (Shor, 7^x mod 15, x: 8 bits) | interpreter | 6,592 | 4,890,420 | 2,290,680 |
| | qoracle | **414** | **1,744** | **416** |

That is 1,000 to 3,000 times fewer gates and 16 to 41 times fewer qubits. Both are exact on all
inputs. The interpreter remains what runs *any* binary unchanged; a compiled oracle is what a fixed
function should be.

## What is supported

- **Input**: integer C functions, or AArch64 ELF object files. Up to 8 arguments, each a quantum input
  of 1 to 64 bits or a constant.
- **Instructions**: add / subtract with flags, logic, multiply (also 64-bit products and high halves),
  unsigned and signed divide, shifts and rotates by constants or registers, bit-field and extend
  instructions, bit and byte reversal, count leading zeros, conditional select / set / compare,
  branches of every kind, loops.
- **Output**: compute oracles |x⟩|0⟩ → |x⟩|f(x)⟩ and phase oracles |x⟩ → (−1)^f(x)|x⟩; OpenQASM 3, Qiskit;
  a resource report.

Not yet: memory (arrays, pointers, the stack at `-O0`, and lookup tables the compiler creates, which
an AES S-box needs), calls between functions, floating point, x86-64 and RISC-V code.

## Roadmap

1. Lookup tables as quantum ROM (table reads with a superposed index), then arrays.
2. Smaller circuits: in-place XOR, qubit reuse (pebbling), measurement-based uncomputation (no
   Toffoli gates to undo ANDs), better adders and multipliers.
3. A case study: a round of SHA-256 or AES from its reference C code, against published hand-made circuits.
4. x86-64 and RISC-V input, using the decoders of cpu-qpu; QIR and CUDA-Q export.

## Tests

```bash
python3 -m tests.test_oracles          # word operations; 23 functions at -O0..-O3/-Os; random functions; errors; Grover
python3 experiments/vs_interpreter.py  # the comparison above
```

## License

Copyright 2026 Quantag IT Solutions GmbH. Apache License 2.0 (`LICENSE`); redistributions must keep
the attribution in `NOTICE`.
