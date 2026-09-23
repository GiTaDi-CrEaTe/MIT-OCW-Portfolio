"""
Tests for 6.042 Number Theory and Cryptography
"""

import math
import random
import pytest
from conftest import load_course_module

nt = load_course_module("nt_crypto", "6.042-Mathematics-for-Computer-Science/Applied-Theory/number_theory_cryptography.py")


def test_gcd_matches_standard_library():
    test_pairs = [(48, 18), (270, 192), (17, 13), (100, 0), (0, 25), (1000003, 99991)]
    for a, b in test_pairs:
        assert nt.gcd(a, b) == math.gcd(a, b)


def test_extended_gcd_bezout_identity():
    pairs = [(35, 15), (101, 10), (252, 198), (240, 46), (17, 5)]
    for a, b in pairs:
        g, x, y = nt.extended_gcd(a, b)
        assert g == nt.gcd(a, b)
        assert a * x + b * y == g


def test_mod_pow_repeated_squaring():
    test_cases = [(2, 10, 1000), (3, 13, 50), (7, 256, 13), (12345, 6789, 100007)]
    for base, exp, mod in test_cases:
        pow_fn = getattr(nt, "mod_pow", getattr(nt, "mod_exp", None))
        assert pow_fn(base, exp, mod) == pow(base, exp, mod)


def test_mod_inverse_correctness():
    pairs = [(3, 11), (7, 26), (65537, 3120), (17, 3120)]
    for a, m in pairs:
        inv = nt.mod_inverse(a, m)
        assert (a * inv) % m == 1


def test_miller_rabin_primality():
    primes = [2, 3, 5, 7, 11, 13, 97, 101, 7919]
    composites = [4, 6, 8, 9, 15, 21, 100, 561]  # 561 is a Carmichael number
    for p in primes:
        assert nt.is_probable_prime(p, rounds=20) is True
    for c in composites:
        assert nt.is_probable_prime(c, rounds=20) is False


def test_rsa_keygen_and_roundtrip():
    keygen_fn = getattr(nt, "generate_rsa_keypair", getattr(nt, "rsa_keygen", None))
    pub, priv = keygen_fn(bits=256)
    n, e = pub
    _, d = priv

    assert n > 0 and e > 0 and d > 0

    message = 123456789
    cipher = nt.rsa_encrypt(message, pub)
    decrypted = nt.rsa_decrypt(cipher, priv)
    assert decrypted == message
    assert cipher != message



