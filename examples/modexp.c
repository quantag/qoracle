/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Quantag IT Solutions GmbH */
/* Shor's algorithm: modular exponentiation a^x mod n, the function whose period gives the factors. */
unsigned modexp(unsigned a, unsigned x, unsigned n)
{
    unsigned r = 1;
    while (x) {
        if (x & 1)
            r = r * a % n;
        a = a * a % n;
        x >>= 1;
    }
    return r;
}
