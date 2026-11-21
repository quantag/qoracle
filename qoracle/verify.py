# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Quantag IT Solutions GmbH
"""Verification: the oracle circuit against the real machine code, run on an AArch64 CPU emulator."""
import random
from unicorn import Uc, UC_ARCH_ARM64, UC_MODE_ARM, UcError
from unicorn import arm64_const as U

CODE, RET = 0x100000, 0x200000
REGS = [getattr(U, f"UC_ARM64_REG_X{i}") for i in range(8)]


class CPU:
    """Calls the function's machine code with given arguments (x0..x7) and returns x0."""

    def __init__(self, code, base):
        self.mu = Uc(UC_ARCH_ARM64, UC_MODE_ARM)
        size = (len(code) + base + 0xFFF) & ~0xFFF
        self.mu.mem_map(CODE, max(size, 0x1000))
        self.mu.mem_map(RET, 0x1000)
        self.mu.mem_write(CODE + base, code)
        self.entry = CODE + base

    def back_edges(self, args, edges, max_insns=10_000_000):
        """How many times the code jumps along one of `edges` ((from-instruction, to) addresses) in one call."""
        from unicorn import UC_HOOK_CODE
        seen = {"prev": None, "n": 0}
        e = {(CODE + a, CODE + b) for a, b in edges}

        def hook(uc, address, size, _):
            if (seen["prev"], address) in e:
                seen["n"] += 1
            seen["prev"] = address
        h = self.mu.hook_add(UC_HOOK_CODE, hook)
        try:
            self.call(args, max_insns)
        finally:
            self.mu.hook_del(h)
        return seen["n"]

    def call(self, args, max_insns=10_000_000):
        mu = self.mu
        for r, v in zip(REGS, list(args) + [0] * 8):
            mu.reg_write(r, v & (2 ** 64 - 1))
        mu.reg_write(U.UC_ARM64_REG_X30, RET)
        try:
            mu.emu_start(self.entry, RET, count=max_insns)
        except UcError as e:
            raise RuntimeError(f"the CPU emulator stopped: {e}")
        if mu.reg_read(U.UC_ARM64_REG_PC) != RET:
            raise RuntimeError(f"the function did not return within {max_insns} instructions")
        return mu.reg_read(U.UC_ARM64_REG_X0)


def test_inputs(spec, limit=4096, seed=1):
    """All inputs when there are few, else edge values plus random ones. spec: [(name, bits, signed)]."""
    total = sum(b for _, b, _ in spec)
    if total <= 16:
        vectors = []
        for v in range(1 << total):
            vec, sh = {}, 0
            for name, bits, _ in spec:
                vec[name] = (v >> sh) & ((1 << bits) - 1)
                sh += bits
            vectors.append(vec)
        return vectors, True
    rng = random.Random(seed)
    vectors = []
    for name0, _, _ in spec:                          # edge values of each input, the others random
        for e in ("zero", "one", "max", "top"):
            vec = {}
            for name, bits, _ in spec:
                if name == name0:
                    vec[name] = {"zero": 0, "one": 1, "max": (1 << bits) - 1, "top": 1 << (bits - 1)}[e]
                else:
                    vec[name] = rng.getrandbits(bits)
            vectors.append(vec)
    while len(vectors) < limit:
        vectors.append({name: rng.getrandbits(bits) for name, bits, _ in spec})
    return vectors, False
