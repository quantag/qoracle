/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Quantag IT Solutions GmbH */
/* Grover oracle: is x a square root of 9 modulo 16? (x = 3, 5, 11, 13) */
int is_root(int x)
{
    return ((x * x) & 15) == 9;
}
