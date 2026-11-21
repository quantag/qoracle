# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Quantag IT Solutions GmbH
"""qoracle: compile classical functions -- C source or AArch64 machine code -- into verified
reversible circuits (quantum oracles).

    from qoracle import build
    o = build("is_root.c", "is_root", args=["x:4"], mode="phase")
    print(o.report)                 # qubits, Toffoli count, depth, verification
    open("is_root.qasm", "w").write(o.circuit.qasm3())
    qc = o.circuit.to_qiskit()      # a Qiskit circuit
"""
import os, time
from dataclasses import dataclass, field

from .xag import XAG, FALSE
from .circuit import Circuit
from . import frontend, symexec, verify
from .a64 import Unsupported

__version__ = "0.2.0"
MASK64 = (1 << 64) - 1


class OracleError(Exception):
    pass


@dataclass
class Arg:
    """An argument: a quantum input (name, bits, signed) or a classical constant (value)."""
    name: str = None
    bits: int = 0
    signed: bool = False
    value: int = None

    @staticmethod
    def parse(text):
        """'x:4' (4-bit unsigned input x), 'x:4s' (signed), or an integer constant ('7', '-3', '0x10')."""
        t = str(text).strip()
        if ":" in t:
            name, b = t.split(":", 1)
            signed = b.endswith("s")
            bits = int(b.rstrip("su"))
            if not name.isidentifier() or not 1 <= bits <= 64:
                raise OracleError(f"bad input '{t}': expected name:bits, for example x:8 or x:8s")
            return Arg(name=name, bits=bits, signed=signed)
        try:
            return Arg(value=int(t, 0))
        except ValueError:
            raise OracleError(f"bad argument '{t}': expected name:bits or an integer constant")


@dataclass
class Oracle:
    function: str
    circuit: Circuit
    report: dict
    args: list
    out_bits: int
    out_signed: bool
    listing: list = field(default_factory=list)


def build(source, function, args, out_bits=32, mode="compute", loop_bound="auto", opt="1", cc="clang",
          verify_limit=4096, keep_constant_outputs=False, check=True):
    """Compile `function` from a C file or AArch64 object file into an oracle circuit, and verify it.
    loop_bound: the most jumps back (loop repetitions, all loops together) on any path; "auto" measures it
    by running the machine code on the test inputs (exact when the inputs are tested exhaustively)."""
    t0 = time.time()
    args = [a if isinstance(a, Arg) else Arg.parse(a) for a in args]
    if len(args) > 8:
        raise OracleError("at most 8 arguments (x0..x7)")
    names = [a.name for a in args if a.name]
    if len(set(names)) != len(names):
        raise OracleError("input names must be different")
    if not names:
        raise OracleError("give at least one quantum input, for example x:4")
    try:
        if source.endswith(".c"):
            obj, cmd = frontend.compile_c(source, opt, cc)
        else:
            obj, cmd = source, None
        image = frontend.load(obj, function)
        memory = [(image.section_addr, image.section)] + [(a, d) for a, d, _ in image.data]
        fn = symexec.Function(image.code, image.entry, function, memory)
    except (frontend.FrontendError, Unsupported) as e:
        raise OracleError(str(e))
    finally:
        if source.endswith(".c") and "obj" in locals() and os.path.exists(obj):
            os.unlink(obj)
    bound_note = None
    if not fn.has_loops:
        loop_bound = 0
    elif loop_bound == "auto":
        spec = [(a.name, a.bits, a.signed) for a in args if a.name]
        vectors, exhaustive = verify.test_inputs(spec, verify_limit)
        cpu = verify.CPU(image)
        edges = [(fn.blocks[u][-1].addr, v) for u, v, _ in fn.back]
        try:
            loop_bound = max(cpu.back_edges(cpu_args(args, vec), edges) for vec in vectors)
        except RuntimeError as e:
            raise OracleError(f"measuring the loop bound: {e}")
        bound_note = (f"measured on {'all' if exhaustive else 'a sample of'} {len(vectors)} inputs"
                      + ("" if exhaustive else "; inputs outside the sample may need more: set loop_bound explicitly"))
    loop_bound = int(loop_bound)

    g = XAG()
    words, inputs = [], {}
    for a in args:
        if a.name:
            w = g.inputs(a.name, a.bits, 64, a.signed)
            inputs[a.name] = w[:a.bits]
        else:
            w = g.const(a.value & MASK64, 64)
        words.append(w)
    try:
        x0, unfinished, copies, ex = symexec.run(fn, g, words, loop_bound)
    except Unsupported as e:
        raise OracleError(str(e))
    outs = x0[:out_bits]
    dropped = 0
    if not keep_constant_outputs:                       # constant top bits carry no information
        while len(outs) > 1 and outs[-1] == FALSE:
            outs.pop()
            dropped += 1
    if mode == "phase" and len(outs) != 1:
        raise OracleError(f"a phase oracle needs a 1-bit result; {function} returns {len(outs)} varying bits "
                          "(write the function to return 0 or 1)")
    t1 = time.time()
    circ = Circuit(g, inputs, outs, mode)
    rep = circ.report()
    rep.update(function=function, instructions=len(fn.insns), blocks=len(fn.blocks), loops=len(fn.back),
               loop_bound=loop_bound, loop_bound_source=bound_note or ("given" if fn.has_loops else None), block_copies=copies, tables=[n for _, _, n in image.data], rom_bits=ex.rom_bits, xag_nodes=len(g.kind), output_bits=len(outs), constant_top_bits_dropped=dropped,
               compiler=cmd, build_seconds=round(t1 - t0, 2))
    listing = [repr(i) for i in fn.insns]
    o = Oracle(function, circ, rep, args, len(outs), False, listing)
    o.requested_bits = out_bits
    if check:
        rep["verification"] = check_oracle(o, g, inputs, outs, unfinished, ex.oob, image, verify_limit)
    return o


def cpu_args(args, vec):
    regs = []
    for a in args:
        if a.name:
            v = vec[a.name]
            if a.signed and v >> (a.bits - 1):
                v -= 1 << a.bits
            regs.append(v)
        else:
            regs.append(a.value)
    return regs


def check_oracle(o, g, inputs, outs, unfinished, oob, image, limit):
    """Simulate the compute-mode circuit on many inputs and compare with the CPU emulator."""
    spec = [(a.name, a.bits, a.signed) for a in o.args if a.name]
    vectors, exhaustive = verify.test_inputs(spec, limit)
    vc = Circuit(g, inputs, outs, "compute", extra=[unfinished, oob])
    lanes = len(vectors)
    q = vc.simulate(lanes, {name: [v[name] for v in vectors] for name, _, _ in spec})
    got = Circuit.lane_values(q, vc.out0, vc.n_out, lanes)
    unf = Circuit.lane_values(q, vc.extra0, 1, lanes)
    out_of_range = Circuit.lane_values(q, vc.extra0 + 1, 1, lanes)
    if any(out_of_range):
        j = out_of_range.index(1)
        raise OracleError(f"table index out of range for input {vectors[j]}: the function reads outside its data")
    if any(unf):
        j = unf.index(1)
        raise OracleError(f"the loop bound is too small: input {vectors[j]} needs more iterations")
    dirty = [b for b in range(vc.anc0, vc.nqubits) if q[b]]
    if dirty:
        raise OracleError(f"internal error: {len(dirty)} work qubits are not returned to 0")
    for name, first, size in vc.regs:
        if name in inputs and Circuit.lane_values(q, first, size, lanes) != [v[name] for v in vectors]:
            raise OracleError(f"internal error: input {name} was changed")
    cpu = verify.CPU(image)
    mask = (1 << o.requested_bits) - 1          # the full width: dropped top bits must be 0 on the CPU too
    for j, vec in enumerate(vectors):
        try:
            want = cpu.call(cpu_args(o.args, vec)) & mask
        except RuntimeError as e:
            raise OracleError(f"input {vec}: {e}")
        if got[j] != want:
            raise OracleError(f"MISMATCH for input {vec}: circuit {got[j]}, CPU {want}")
    return dict(inputs_tested=lanes, exhaustive=exhaustive, against="AArch64 CPU emulator (Unicorn)",
                work_qubits_clean=True, inputs_unchanged=True)
