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


