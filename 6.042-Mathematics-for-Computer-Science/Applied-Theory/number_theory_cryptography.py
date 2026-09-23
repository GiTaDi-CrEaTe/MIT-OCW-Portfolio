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


def extended_gcd(a: int, b: int):
    """
    Returns (g, x, y) such that a*x + b*y = g = gcd(a, b).
    Implemented iteratively (equivalent to the recursive textbook version,
    but avoids recursion-depth issues for large moduli).
    """
    old_r, r = a, b
    old_x, x = 1, 0
    old_y, y = 0, 1
    while r != 0:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_x, x = x, old_x - quotient * x
        old_y, y = y, old_y - quotient * y
    return old_r, old_x, old_y  # g, x, y


def mod_inverse(a: int, m: int) -> int:
    """
    Modular inverse of a mod m, i.e. the unique x in [0, m) with a*x ≡ 1 (mod m).
    Exists iff gcd(a, m) = 1 (this is exactly Bezout's identity specialized
    to g = 1).
    """
    g, x, _ = extended_gcd(a, m)
    if g != 1:
        raise ValueError(f"No modular inverse: gcd({a}, {m}) = {g} != 1")
    return x % m


# ---------------------------------------------------------------------------
# 2. Modular exponentiation by repeated squaring  --  O(log exponent) multiplications
#    instead of the naive O(exponent). This is the computational engine that
#    makes RSA feasible at all.
# ---------------------------------------------------------------------------

def mod_pow(base: int, exponent: int, modulus: int) -> int:
    """
    Computes (base ** exponent) % modulus without ever materializing the
    (astronomically large) unreduced power. Standard binary/repeated-squaring
    exponentiation: write the exponent in binary and square-and-multiply.
    """
    if modulus == 1:
        return 0
    result = 1
    base = base % modulus
    while exponent > 0:
        if exponent & 1:  # current bit is 1 -> fold this power of base in
            result = (result * base) % modulus
        exponent >>= 1
        base = (base * base) % modulus
    return result


# ---------------------------------------------------------------------------
# 3. Miller-Rabin primality test  --  a randomized algorithm built directly on
#    Fermat's Little Theorem, strengthened to rule out Fermat pseudoprimes.
# ---------------------------------------------------------------------------

def is_probable_prime(n: int, rounds: int = 40) -> bool:
    """
    Miller-Rabin primality test.

    Theory: write n - 1 = 2^r * d with d odd. If n is prime, then for any
    witness a in [2, n-2], the sequence
        a^d, a^(2d), a^(4d), ..., a^((2^(r-1))d)   (mod n)
    must either start at 1, or hit -1 (mod n) at some point before reaching
    a^(n-1). This follows because Z/nZ is a field when n is prime, so x^2 = 1
    has only the roots x = 1 and x = -1 -- there can be no other square root
    of unity. A composite n will fail this for at least 3/4 of possible
    witnesses a, so repeating with independent random witnesses drives the
    false-positive probability down to at most 4^(-rounds).
    """
    if n < 2:
        return False
    for small_prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n == small_prime:
            return True
        if n % small_prime == 0:
            return False

    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    for _ in range(rounds):
        a = random.randrange(2, n - 1)
        x = mod_pow(a, d, n)
        if x == 1 or x == n - 1:
            continue  # this witness is consistent with primality
        composite = True
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                composite = False
                break
        if composite:
            return False
    return True


def generate_prime(bits: int) -> int:
    """Generates a random odd bits-bit number and tests it with Miller-Rabin
    until a probable prime is found."""
    while True:
        candidate = random.getrandbits(bits) | (1 << (bits - 1)) | 1  # force top and bottom bit
        if is_probable_prime(candidate):
            return candidate


# ---------------------------------------------------------------------------
# 4. RSA key generation, encryption, decryption
# ---------------------------------------------------------------------------

def generate_rsa_keypair(bits: int = 256):
    """
    Generates an RSA keypair with n of roughly `bits` bits.
    Returns (public_key, private_key) = ((n, e), (n, d)).
    """
    p = generate_prime(bits // 2)
    q = generate_prime(bits // 2)
    while p == q:
        q = generate_prime(bits // 2)

    n = p * q
    phi = (p - 1) * (q - 1)

    e = 65537  # standard choice: small, prime, and F4-shaped for fast encryption
    if gcd(e, phi) != 1:
        # Extremely unlikely with random primes, but handle it correctly anyway.
        e = 3
        while gcd(e, phi) != 1:
            e += 2

    d = mod_inverse(e, phi)
    return (n, e), (n, d)


