# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Quantag IT Solutions GmbH
"""qoracle against the AArch64 CPU, and against the x86-64 CPU (parts 2-4 again with --arch x86_64).

  1. Word operations of the logic network, exhaustively on small widths.
  2. Hand-written functions (arithmetic, division by constants and variables, shifts, comparisons,
     branches, loops, bit tricks, lookup tables and switch tables), at -O0..-O3 where the code stays register-only: every oracle is
     simulated on every input and compared with the machine code in the CPU emulator; work qubits
     must return to 0 and inputs must be unchanged.
  3. Random functions: random expressions with every C integer operator, conditionals and loops.
  4. Errors are reported: an index outside a table, writing memory, calls, a loop bound that is too
     small, phase mode with a multi-bit result.
  5. Phase oracles: Grover search with the is_root oracle finds the four square roots of 9 mod 16.
Run: python3 -m tests.test_oracles [random functions]"""
import os, random, sys, tempfile, textwrap
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from qoracle import build, OracleError
from qoracle.xag import XAG
from qoracle import sim

FUNCTIONS = r"""
int add3(int a, int b) { return (a + 3 * b) ^ (a >> 2); }
unsigned div7(unsigned x) { return x / 7 + (x & 0xf0) + (x << 5); }
int pick(int a, int b, int c) { return a > b ? a - c : b + (c << 2); }
long mixed(long a, int b) { return a * b + (unsigned char)b - (short)a; }
int popcount(int x) { int c = 0; while (x) { c += x & 1; x >>= 1; } return c; }
unsigned rem_or(unsigned a, unsigned b) { return (a % (b + 1)) | (a >> (b & 7)) | ((a & 0xf0) >> 4); }
int either(int a, int b) { return (a == 3 || b < 0) ? -a : ~b; }
int sdiv(int a, int b) { return b ? a / b : 0; }
int smod(int a, int b) { return b ? a % b : 1; }
unsigned gcd(unsigned a, unsigned b) { while (b) { unsigned t = a % b; a = b; b = t; } return a; }
int collatz(int n) { int s = 0; while (n > 1 && s < 20) { n = (n & 1) ? 3 * n + 1 : n / 2; s++; } return s; }
int isqrt(int n) { int r = 0; while ((r + 1) * (r + 1) <= n) r++; return r; }
int is_prime(int n) { if (n < 2) return 0; for (int d = 2; d * d <= n; d++) if (n % d == 0) return 0; return 1; }
unsigned rotl(unsigned x, unsigned k) { k &= 31; return (x << k) | (x >> ((32 - k) & 31)); }
int clamp(int x, int lo, int hi) { return x < lo ? lo : x > hi ? hi : x; }
int absdiff(int a, int b) { return a > b ? a - b : b - a; }
unsigned bitrev8(unsigned x) { unsigned r = 0; for (int i = 0; i < 8; i++) { r = (r << 1) | (x & 1); x >>= 1; } return r; }
int majority(int a, int b, int c) { return (a & b) | (a & c) | (b & c); }
unsigned long mulhi(unsigned long a, unsigned long b) { return (unsigned long)(((unsigned __int128)a * b) >> 64); }
int clz32(unsigned x) { return x ? __builtin_clz(x) : 32; }
int parity(unsigned x) { return __builtin_parity(x); }
int sign(int x) { return (x > 0) - (x < 0); }
int fib(int n) { int a = 0, b = 1; for (int i = 0; i < n; i++) { int t = a + b; a = b; b = t; } return a; }
static const unsigned char SB[16] = {0xC, 5, 6, 0xB, 9, 0, 0xA, 0xD, 3, 0xE, 0xF, 8, 4, 7, 1, 2};
unsigned sbox2(unsigned x) { return SB[x & 15] | SB[(x >> 4) & 15] << 4; }
static const short ST[8] = {-5, 3, -100, 7, 0, 1000, -1, 2};
long short_tab(unsigned x, long y) { return ST[x & 7] * y; }
static const unsigned SQ[8] = {0, 1, 4, 9, 16, 25, 36, 49};
unsigned sq_loop(unsigned x) { unsigned s = 0; while (x) { s += SQ[x & 7]; x >>= 3; } return s; }
int swtab(int x) { switch (x) { case 0: return 11; case 1: return 22; case 2: return 37; case 3: return 41;
                                case 4: return 59; case 5: return 63; default: return -1; } }
static const unsigned long BIG[4] = {0x0123456789abcdefUL, 0xfedcba9876543210UL, 42, 7};
unsigned long big_tab(unsigned x) { return BIG[x & 3] >> (x >> 2); }
"""

# (function, inputs, extra constant arguments placed by position: None = an input)
CASES = [
    ("add3", ["a:5s", "b:5s"]), ("div7", ["x:10"]), ("pick", ["a:4s", "b:4s", "c:4s"]), ("mixed", ["a:6s", "b:6s"]),
    ("popcount", ["x:10"]), ("rem_or", ["a:6", "b:6"]), ("either", ["a:5s", "b:5s"]), ("sdiv", ["a:6s", "b:5s"]),
    ("smod", ["a:6s", "b:5s"]), ("gcd", ["a:6", "b:6"]), ("collatz", ["n:6"]), ("isqrt", ["n:8"]),
    ("is_prime", ["n:8"]), ("rotl", ["x:8", "k:5"]), ("clamp", ["x:6s", "-5", "20"]), ("absdiff", ["a:6s", "b:6s"]),
    ("bitrev8", ["x:8"]), ("majority", ["a:4", "b:4", "c:4"]), ("mulhi", ["a:12", "b:4"]), ("clz32", ["x:12"]),
    ("parity", ["x:12"]), ("sign", ["x:8s"]), ("fib", ["n:4"]),
    ("sbox2", ["x:8"]), ("short_tab", ["x:3", "y:4s"]), ("sq_loop", ["x:9"]), ("swtab", ["x:4s"]), ("big_tab", ["x:6"]),
]


def word_ops():
    import itertools
    w = 5
    M = (1 << w) - 1
    g = XAG()
    A, B = g.inputs("a", w, w), g.inputs("b", w, w)
    outs = dict(add=g.add(A, B)[0], sub=g.sub(A, B)[0], mul=g.mul(A, B), udiv=g.udivmod(A, B)[0],
                urem=g.udivmod(A, B)[1], sdiv=g.sdiv(A, B))
    sx = lambda v: v - (1 << w) if v >> (w - 1) else v

    def ev(l, env, memo):
        n = l >> 1
        if n not in memo:
            k = g.kind[n]
            memo[n] = 0 if k == "const" else env[g.name[n]] if k == "in" else \
                (ev(g.a[n], env, memo) & ev(g.b[n], env, memo)) if k == "and" else (ev(g.a[n], env, memo) ^ ev(g.b[n], env, memo))
        return memo[n] ^ (l & 1)
    n = 0
    for a, b in itertools.product(range(1 << w), repeat=2):
        env = {("a", i): (a >> i) & 1 for i in range(w)}
        env.update({("b", i): (b >> i) & 1 for i in range(w)})
        memo = {}
        want = dict(add=(a + b) & M, sub=(a - b) & M, mul=(a * b) & M, udiv=a // b if b else 0, urem=a % b if b else a,
                    sdiv=(int(sx(a) / sx(b)) if b else 0) & M)
        for k, v in want.items():
            got = sum(ev(x, env, memo) << i for i, x in enumerate(outs[k]))
            assert got == v, (k, a, b, got, v)
            n += 1
    return n


def compile_cases(d, arch="aarch64"):
    src = os.path.join(d, "funcs.c")
    open(src, "w").write(FUNCTIONS)
    stats = dict(oracles=0, inputs=0, skipped=[])
    for opt in ("0", "1", "2", "3", "s"):
        for name, args in CASES:
            try:
                o = build(src, name, args, opt=opt, arch=arch)
            except OracleError as e:
                msg = str(e)
                if opt == "0" and "stack" in msg:
                    stats["skipped"].append(f"{name} -O0 (stack)")
                    continue
                if arch == "x86_64" and "jump table" in msg:
                    stats["skipped"].append(f"{name} -O{opt} (a jump table)")
                    continue
                raise AssertionError(f"{name} -O{opt}: {msg}")
            v = o.report["verification"]
            assert v["work_qubits_clean"] and v["inputs_unchanged"]
            stats["oracles"] += 1
            stats["inputs"] += v["inputs_tested"]
    return stats


OPS = ["+", "-", "*", "&", "|", "^", "<<", ">>", "/", "%", "<", "==", "!="]


def rand_expr(r, vars_, depth):
    if depth == 0 or r.random() < 0.25:
        return r.choice(vars_ + [str(r.randrange(-9, 30)), str(r.randrange(1, 300))])
    op = r.choice(OPS)
    a, b = rand_expr(r, vars_, depth - 1), rand_expr(r, vars_, depth - 1)
    if op in ("<<", ">>"):
        b = f"(({b}) & 7)"
    if op in ("/", "%"):
        b = f"(({b}) | 1)"                       # no division by zero, no INT_MIN / -1 trap in C
        if r.random() < 0.5:
            return f"((unsigned)({a}) {op} (unsigned)({b}))"
    if r.random() < 0.3:
        return f"(({a}) > ({b}) ? ({a}) {op} 1 : ({b}))" if op not in ("<<", ">>", "/", "%") else f"(({a}) {op} ({b}))"
    return f"(({a}) {op} ({b}))"


def random_functions(d, n, seed=11, arch="aarch64"):
    r = random.Random(seed)
    lines, cases = [], []
    for k in range(n):
        nv = r.randrange(1, 4)
        vs = ["a", "b", "c"][:nv]
        body = f"int r = {rand_expr(r, vs, 3)};"
        if r.random() < 0.4:
            body += f" for (int i = 0; i < (a & 3); i++) r = r * 3 + ({rand_expr(r, vs + ['i', 'r'], 2)});"
        body += " return r;"
        lines.append(f"int f{k}({', '.join('int ' + v for v in vs)}) {{ {body} }}")
        cases.append((f"f{k}", [f"{v}:{r.choice([3, 4, 5])}{r.choice(['', 's'])}" for v in vs]))
    src = os.path.join(d, "rand.c")
    open(src, "w").write("\n".join(lines) + "\n")
    count = tables = 0
    for name, args in cases:
        o = build(src, name, args, opt=r.choice(["1", "2"]), arch=arch)
        tables += bool(o.report["rom_bits"])
        assert o.report["verification"]["exhaustive"]
        count += 1
    return count, tables


def errors(d, arch="aarch64"):
    src = os.path.join(d, "bad.c")
    open(src, "w").write(textwrap.dedent("""\
        static const unsigned char t4[4] = {3, 1, 4, 1};
        int g(int);
        int lookup(unsigned x) { return t4[x]; }
        int counter;
        int store(int x) { counter = x; return x; }
        int calls(int x) { return g(x) + 1; }
        int twice(int x) { return 2 * x; }
        int loop(int x) { int s = 0; while (x) { s = s * 3 + (x & 1); x >>= 1; } return s; }
        """))
    for fn, args, kw, msg in (("lookup", ["x:2"], {}, None), ("lookup", ["x:12"], {}, "table index out of range"),
                              ("store", ["x:4"], {}, "memory write"), ("calls", ["x:4"], {}, "refers to 'g'"),
                              ("twice", ["x:4"], {"mode": "phase"}, "1-bit result"),
                              ("loop", ["x:6"], {"loop_bound": 2}, "loop bound is too small"),
                              ("nothere", ["x:4"], {}, "no function 'nothere'")):
        try:
            build(src, fn, args, arch=arch, **kw)
            if msg is None:
                continue
        except OracleError as e:
            assert msg in str(e), (fn, str(e))
        else:
            raise AssertionError(f"{fn}: no error")
    return 7


def grover():
    o = build(os.path.join(ROOT, "examples", "is_root.c"), "is_root", ["x:4"], mode="phase")
    probs = sim.grover(o.circuit, "x", iterations=1)
    good = {3, 5, 11, 13}
    p = sum(v for k, v in probs.items() if k in good)
    assert abs(p - 1.0) < 1e-9, probs
    return o.report, p


def main():
    n_rand = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    print(f"1) word operations: {word_ops()} results exact (5-bit, all pairs)")
    with tempfile.TemporaryDirectory() as d:
        s = compile_cases(d)
        print(f"2) {s['oracles']} oracles from {len(CASES)} functions at -O0/-O1/-O2/-O3/-Os: every one equal to the "
              f"AArch64 CPU on all {s['inputs']} inputs; work qubits clean; ({len(s['skipped'])} -O0 cases use the stack: skipped)")
        ok, tables = random_functions(d, n_rand)
        print(f"3) {ok} random functions: equal to the CPU on all inputs ({tables} of them with lookup tables the compiler made)")
        print(f"4) {errors(d)} error cases reported correctly")
        s = compile_cases(d, "x86_64")
        print(f"2x) x86-64: {s['oracles']} oracles from {len(CASES)} functions at -O0..-Os: every one equal to the x86-64 "
              f"CPU on all {s['inputs']} inputs; skipped: {', '.join(s['skipped']) or 'none'}")
        ok, tables = random_functions(d, n_rand, arch="x86_64")
        print(f"3x) x86-64: {ok} random functions equal to the CPU on all inputs ({tables} with lookup tables)")
        print(f"4x) x86-64: {errors(d, 'x86_64')} error cases reported correctly")
    rep, p = grover()
    print(f"5) Grover with the is_root phase oracle ({rep['qubits']} qubits, {rep['toffoli']} Toffoli): "
          f"P(x in {{3, 5, 11, 13}}) = {p:.6f} after 1 iteration")
    print("PASS")


if __name__ == "__main__":
    main()
