"""
6.042J Applied Theory  --  RSA Public-Key Cryptography from First Principles
===========================================================================

This script implements RSA using only the number-theoretic primitives taught
in 6.042: the Euclidean Algorithm, the Extended Euclidean Algorithm, Fermat's
Little Theorem (via a Miller-Rabin primality test), and modular exponentiation
by repeated squaring. No `cryptography`, no `sympy`, no external number-theory
libraries. The only import is Python's own `random` module for randomized
primality testing.

Why RSA specifically: its security and correctness are *both* pure consequences
of theorems from this course. Correctness follows from Euler's theorem
(a generalization of Fermat's Little Theorem); the encryption/decryption
exponents exist because of Bezout's identity, produced by the Extended
Euclidean Algorithm.

------------------------------------------------------------------------------
THEORY RECAP (see Pset 5-6 in Psets/pset_roadmap.md)
------------------------------------------------------------------------------
1. Euclidean Algorithm: gcd(a, b) = gcd(b, a mod b), base case gcd(a, 0) = a.

2. Extended Euclidean Algorithm: computes integers (x, y) such that
       a*x + b*y = gcd(a, b)              (Bezout's identity)
   If gcd(a, b) = 1, then x is the modular inverse of a mod b.

3. Fermat's Little Theorem: if p is prime and gcd(a, p) = 1, then
       a^(p-1) ≡ 1 (mod p)
   Miller-Rabin uses a stronger, refined version of this fact to test
   primality with negligible false-positive probability.

4. RSA construction:
   - Choose two large primes p, q.  Let n = p*q,  φ(n) = (p-1)(q-1).
   - Choose e coprime to φ(n).  Compute d = e^{-1} mod φ(n) via Extended Euclid.
   - Public key: (n, e).  Private key: (n, d).
   - Encrypt:  c = m^e mod n.
   - Decrypt:  m = c^d mod n.
   - Correctness (why this recovers m): c^d = m^(ed) mod n, and because
     ed ≡ 1 (mod φ(n)), Euler's theorem gives m^(ed) ≡ m (mod n) whenever
     gcd(m, n) = 1  --  which holds with overwhelming probability for random m
     since n's only prime factors are p and q.
------------------------------------------------------------------------------
"""

import random


# ---------------------------------------------------------------------------
# 1. Euclidean Algorithm and Extended Euclidean Algorithm
# ---------------------------------------------------------------------------

def gcd(a: int, b: int) -> int:
    """Euclid's algorithm: gcd(a, b) = gcd(b, a mod b)."""
    while b:
        a, b = b, a % b
    return a


