"""
6.042J Applied Theory — RSA Public-Key Cryptography from First Principles
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
     gcd(m, n) = 1 — which holds with overwhelming probability for random m
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
# 2. Modular exponentiation by repeated squaring — O(log exponent) multiplications
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
# 3. Miller-Rabin primality test — a randomized algorithm built directly on
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


def rsa_encrypt(message_int: int, public_key) -> int:
    n, e = public_key
    if message_int >= n:
        raise ValueError("Message integer must be smaller than the modulus n.")
    return mod_pow(message_int, e, n)


def rsa_decrypt(cipher_int: int, private_key) -> int:
    n, d = private_key
    return mod_pow(cipher_int, d, n)


def encode_text(text: str) -> int:
    """Encodes a UTF-8 string as one big integer (simple positional encoding)."""
    return int.from_bytes(text.encode("utf-8"), byteorder="big")


def decode_text(number: int) -> str:
    length = (number.bit_length() + 7) // 8
    return number.to_bytes(length, byteorder="big").decode("utf-8")


# ---------------------------------------------------------------------------
# 5. Self-verification: round-trip correctness + spot-check Miller-Rabin
#    against trial division on small numbers.
# ---------------------------------------------------------------------------

def _self_test():
    print("=" * 70)
    print("SELF-TEST 1: Miller-Rabin vs. trial-division ground truth (n < 5000)")
    print("=" * 70)

    def is_prime_trial_division(n):
        if n < 2:
            return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True

    mismatches = 0
    for n in range(2, 5000):
        if is_probable_prime(n, rounds=10) != is_prime_trial_division(n):
            mismatches += 1
    print(f"Checked n = 2..4999. Mismatches against trial division: {mismatches}")
    assert mismatches == 0, "Miller-Rabin disagreed with ground truth!"
    print("PASSED: Miller-Rabin matches trial division on the full test range.\n")

    print("=" * 70)
    print("SELF-TEST 2: Extended Euclid produces a valid Bezout identity")
    print("=" * 70)
    for (a, b) in [(240, 46), (17, 5), (1000003, 99991), (48, 18)]:
        g, x, y = extended_gcd(a, b)
        assert a * x + b * y == g
        assert g == gcd(a, b)
        print(f"gcd({a}, {b}) = {g},  verified {a}*({x}) + {b}*({y}) = {g}")
    print("PASSED: Bezout identity holds for every tested pair.\n")

    print("=" * 70)
    print("SELF-TEST 3: End-to-end RSA round trip on a real message")
    print("=" * 70)
    random.seed(6042)  # reproducibility for the portfolio reader
    public_key, private_key = generate_rsa_keypair(bits=512)
    n, e = public_key
    print(f"Generated {n.bit_length()}-bit modulus n.")
    print(f"Public key e  = {e}")

    message = "MIT 6.042J: proof by induction, then proof by construction."
    m_int = encode_text(message)
    assert m_int < n, "Message too large for this modulus; shorten the message."

    cipher = rsa_encrypt(m_int, public_key)
    recovered_int = rsa_decrypt(cipher, private_key)
    recovered_text = decode_text(recovered_int)

    print(f"Plaintext:  {message!r}")
    print(f"Ciphertext (int, truncated repr): {str(cipher)[:40]}...")
    print(f"Decrypted: {recovered_text!r}")
    assert recovered_text == message, "RSA round trip failed!"
    print("PASSED: Decrypted text exactly matches the original plaintext.\n")

    print("=" * 70)
    print("SELF-TEST 4: Encrypting with the public key alone cannot be undone")
    print("without the private exponent d (structural sanity check only —")
    print("this script does not attempt a factoring attack).")
    print("=" * 70)
    wrong_d = private_key[1] + 2  # a nearby, wrong private exponent
    garbage = rsa_decrypt(cipher, (n, wrong_d))
    print(f"Decryption with a wrong d yields nonsense integer (as expected): "
          f"{garbage != m_int}")
    assert garbage != m_int
    print("PASSED: correctness of RSA depends critically on the exact d.\n")

    print("All self-tests passed. RSA implementation is verified against")
    print("ground truth and demonstrates a full theory-to-execution pipeline.")


if __name__ == "__main__":
    _self_test()
