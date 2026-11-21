# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Quantag IT Solutions GmbH
"""Command line: python3 -m qoracle (or `qoracle` when installed)."""
import argparse, json, sys
from . import build, OracleError, __version__

EPILOG = """examples:
  qoracle examples/is_root.c is_root x:4 --mode phase -o is_root.qasm
        Grover oracle: flips the phase of the inputs x (4 bits) for which is_root(x) is 1
  qoracle examples/modexp.c modexp 7 x:8 15 --out-bits 4 -o modexp.qasm
        Shor: |x>|0> -> |x>|7^x mod 15>, 8-bit exponent, 4-bit result; 7 and 15 are constants

arguments are given in the order of the C parameters: name:bits for a quantum input (name:bitss for a
signed one), or an integer for a constant. The result is the function's return value (register x0)."""


def main(argv=None):
    ap = argparse.ArgumentParser(prog="qoracle", description="Compile a C function or AArch64 machine code into "
                                 "a verified reversible circuit (a quantum oracle).", epilog=EPILOG,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="C file, or an AArch64 ELF object file")
    ap.add_argument("function", help="name of the function")
    ap.add_argument("args", nargs="+", help="per parameter: name:bits (quantum input) or a number (constant)")
    ap.add_argument("--mode", choices=("compute", "phase"), default="compute",
                    help="compute: |x>|0> -> |x>|f(x)>; phase: |x> -> (-1)^f(x) |x> for 1-bit results (default compute)")
    ap.add_argument("--out-bits", type=int, default=32, help="bits of the result to output (default 32; constant top bits are dropped)")
    ap.add_argument("--loop-bound", default="auto", help="most jumps back over all loops on any path, or auto: measured on the CPU (default)")
    ap.add_argument("-O", dest="opt", default="1", help="clang optimization level (default 1)")
    ap.add_argument("--cc", default="clang", help="C compiler (clang with the AArch64 target; default clang)")
    ap.add_argument("-o", "--output", help="write the circuit as OpenQASM 3 to this file")
    ap.add_argument("--verify", type=int, default=4096, help="test inputs when not all can be tested (default 4096)")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    ap.add_argument("--asm", action="store_true", help="also print the machine code")
    ap.add_argument("--version", action="version", version=f"qoracle {__version__}")
    a = ap.parse_args(argv)
    try:
        o = build(a.source, a.function, a.args, out_bits=a.out_bits, mode=a.mode, loop_bound=a.loop_bound,
                  opt=a.opt, cc=a.cc, verify_limit=a.verify)
    except OracleError as e:
        print(f"qoracle: {e}", file=sys.stderr)
        return 1
    r = o.report
    if a.output:
        sig = ", ".join(x.name + ":" + str(x.bits) + ("s" if x.signed else "") if x.name else str(x.value) for x in o.args)
        open(a.output, "w").write(o.circuit.qasm3(f"qoracle {__version__}: {a.function}({sig}), {a.mode} oracle"))
    if a.json:
        print(json.dumps(r, indent=2))
        return 0
    sig = ", ".join(f"{x.name}: {x.bits} bits{' signed' if x.signed else ''}" if x.name else str(x.value) for x in o.args)
    v = r["verification"]
    print(f"{a.function}({sig})  {a.mode} oracle, {r['output_bits']}-bit result")
    print(f"  machine code   AArch64, {r['instructions']} instructions, {r['blocks']} block{'s' if r['blocks'] > 1 else ''}"
          + (f", loops unrolled to {r['loop_bound']} repetitions ({r['loop_bound_source']})" if r["loops"] else ""))
    if a.asm:
        for line in o.listing:
            print("                 " + line)
    print(f"  logic network  {r['and_nodes']} AND, {r['xor_nodes']} XOR")
    print(f"  circuit        {r['qubits']} qubits ({r['inputs']} in, {r['outputs']} out, {r['ancillas']} work), "
          f"{r['gates']} gates: {r['toffoli']} Toffoli, {r['cnot']} CNOT, {r['x'] + r['z']} single-qubit; depth {r['depth']}")
    print(f"  T count        {r['t_count']} (7 per Toffoli)")
    tested = f"all {v['inputs_tested']}" if v["exhaustive"] else f"{v['inputs_tested']} (edge values and random)"
    print(f"  verified       {tested} inputs equal to the {v['against']}; work qubits return to 0")
    if a.output:
        print(f"  written        {a.output} (OpenQASM 3)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
