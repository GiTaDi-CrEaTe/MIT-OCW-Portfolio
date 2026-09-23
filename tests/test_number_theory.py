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



def test_mod_inverse_raises_when_not_coprime():
    """Modular inverse should raise ValueError when gcd(a, m) != 1."""
    with pytest.raises(ValueError, match="No modular inverse"):
        nt.mod_inverse(6, 9)


def test_gcd_with_zero():
    """gcd(a, 0) should return a, gcd(0, b) should return b."""
    assert nt.gcd(42, 0) == 42
    assert nt.gcd(0, 37) == 37


def test_mod_pow_edge_cases():
    """mod_pow with edge cases: exponent 0, modulus 1."""
    pow_fn = getattr(nt, "mod_pow", getattr(nt, "mod_exp", None))
    assert pow_fn(5, 0, 13) == 1
    assert pow_fn(5, 100, 1) == 0


def test_rsa_encrypt_rejects_oversized_message():
    """Encrypting a message >= n should raise ValueError."""
    keygen_fn = getattr(nt, "generate_rsa_keypair", getattr(nt, "rsa_keygen", None))
    pub, _ = keygen_fn(bits=256)
    n, e = pub
    with pytest.raises(ValueError, match="smaller than the modulus"):
        nt.rsa_encrypt(n + 1, pub)


def test_encode_decode_text_roundtrip():
    """Text encoding/decoding should be a perfect roundtrip."""
    if not hasattr(nt, "encode_text") or not hasattr(nt, "decode_text"):
        pytest.skip("encode_text/decode_text not available")
    original = "Hello, 6.042J! [MIT-OCW]"
    assert nt.decode_text(nt.encode_text(original)) == original


def test_miller_rabin_on_carmichael_numbers():
    """Miller-Rabin should correctly identify Carmichael numbers as composite."""
    carmichael = [561, 1105, 1729, 2465, 2821, 6601]
    for c in carmichael:
        assert nt.is_probable_prime(c, rounds=20) is False, f"{c} is Carmichael but reported prime"
