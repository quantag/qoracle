# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Quantag IT Solutions GmbH
"""Turn a function's control flow into one straight-line logic network.

A circuit has no jumps: in superposition, every path runs. So the function is executed on all paths
at once. Each basic block gets an activity predicate (a literal: "execution is here"); a branch splits
the predicate between its two successors; where paths meet, register values are merged with the
predicates (x = p1 & x1 ^ p2 & x2, the predicates being mutually exclusive). Loops are unrolled: the
back edges of the control-flow graph are counted, and copy k of a block is the block after k back
edges. A path that still wants to loop after `bound` back edges sets the `unfinished` literal, which
verification requires to be false for every tested input.
"""
from .xag import FALSE, TRUE
from . import a64


class Function:
    """A decoded function: instructions, basic blocks, and its back edges."""

    def __init__(self, code, base, name, memory=()):
        self.memory = list(memory)
        self.name, self.base, self.end = name, base, base + len(code)
        self.insns = a64.decode(code, base)
        for i in self.insns:
            a64.check(i)
        self.at = {i.addr: k for k, i in enumerate(self.insns)}
        leaders = {base}
        for k, i in enumerate(self.insns):
            if i.is_branch:
                for t in i.targets:
                    if not base <= t < self.end:
                        raise a64.Unsupported(f"a jump out of the function (a tail call?) {i.where()}")
                    leaders.add(t)
                if k + 1 < len(self.insns):
                    leaders.add(self.insns[k + 1].addr)
        starts = sorted(leaders)
        self.blocks = {}                       # start address -> list of instructions
        for j, s in enumerate(starts):
            stop = starts[j + 1] if j + 1 < len(starts) else self.end
            self.blocks[s] = [i for i in self.insns if s <= i.addr < stop]
        # successors: (target, kind) with kind 'taken' / 'fall'
        self.succ = {}
        for s, body in self.blocks.items():
            last = body[-1]
            nxt = last.addr + last.size
            if last.mnemonic == "ret":
                self.succ[s] = []
            elif last.mnemonic == "b":
                self.succ[s] = [(last.targets[0], "taken")]
            elif last.is_branch:
                self.succ[s] = [(last.targets[0], "taken"), (nxt, "fall")]
            else:
                if nxt >= self.end:
                    raise a64.Unsupported(f"the code runs past the end of {name}")
                self.succ[s] = [(nxt, "fall")]
        # back edges: depth-first search from the entry
        self.back, state = set(), {}
        order = []

        def dfs(u):
            state[u] = 1
            for v, kind in self.succ[u]:
                if state.get(v) == 1:
                    self.back.add((u, v, kind))
                elif v not in state:
                    dfs(v)
            state[u] = 2
            order.append(u)
        dfs(base)
        self.reachable = set(state)
        self.topo = order[::-1]                # blocks in an order where every forward edge goes forward

    @property
    def has_loops(self):
        return bool(self.back)


def run(fn, g, args, bound=0, max_nodes=2_000_000):
    """Execute fn on all paths. args: list of 64-literal words for x0, x1, ... Returns (x0 word at return,
    unfinished literal, number of block copies executed, the executor: ex.oob, ex.rom_bits)."""
    ex = a64.Exec(g, fn.memory)
    init = a64.State([list(args[i]) if i < len(args) else [FALSE] * 64 for i in range(31)], (FALSE,) * 4)
    pending = {(fn.base, 0): [(TRUE, init)]}   # (block, copy) -> incoming (predicate, state)
    out = [FALSE] * 64
    done, unfinished, copies = FALSE, FALSE, 0
    for k in range(bound + 1):
        for b in fn.topo:
            inc = pending.pop((b, k), None)
            if not inc:
                continue
            copies += 1
            active, st = merge(g, inc)
            if active == FALSE:
                continue
            ex.active = active
            body = fn.blocks[b]
            for ins in body[:-1] if body[-1].is_branch else body:
                ex.step(ins, st)
            last = body[-1]
            if last.is_branch:
                t = ex.branch(last, st)
                if t == "ret":
                    out = [g.XOR(o, g.AND(active, x)) for o, x in zip(out, st.x[0])]
                    done = g.XOR(done, active)
                    continue
                preds = {"taken": g.AND(active, t), "fall": g.AND(active, t ^ 1)}
            else:
                preds = {"fall": active}
            for v, kind in fn.succ[b]:
                p = preds[kind]
                if p == FALSE:
                    continue
                if (b, v, kind) in fn.back:
                    if k == bound:
                        unfinished = g.XOR(unfinished, p)       # wants another iteration: bound too small
                        continue
                    key = (v, k + 1)
                else:
                    key = (v, k)
                pending.setdefault(key, []).append((p, st))
            if len(g.kind) > max_nodes:
                raise a64.Unsupported(f"the logic network grew beyond {max_nodes} nodes (loop bound {bound})")
    return out, unfinished, copies, ex


def merge(g, inc):
    """Join the incoming paths of a block: predicates are mutually exclusive."""
    if len(inc) == 1:
        p, st = inc[0]
        return p, st.copy()
    active = g.XOR_all(p for p, _ in inc)
    states = [s for _, s in inc]
    x = []
    for r in range(31):
        words = [s.x[r] for s in states]
        if all(wd is words[0] or wd == words[0] for wd in words[1:]):
            x.append(words[0])
            continue
        x.append([lits[0] if all(l == lits[0] for l in lits) else
                  g.XOR_all(g.AND(p, l) for (p, _), l in zip(inc, lits)) for lits in zip(*words)])
    nz = tuple(f[0] if all(v == f[0] for v in f) else g.XOR_all(g.AND(p, v) for (p, _), v in zip(inc, f))
               for f in zip(*(s.nzcv for s in states)))
    return active, a64.State(x, nz)
