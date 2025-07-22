import re

import numpy as np
import pytest


def test_generate_codes_shape_and_type(tools):
    n, length = 5, 12
    codes = tools.generate_alphanum_codes(n=n, length=length)

    assert isinstance(codes, np.ndarray)
    assert codes.shape == (n,)
    assert all(len(code) == length for code in codes)

    pattern = re.compile(f"^[0-9A-Z]{{{length}}}$")
    assert all(pattern.match(code) for code in codes)


def test_generate_codes_uniqueness(tools):
    n = 10000
    codes = tools.generate_alphanum_codes(n=n, length=10)
    assert len(np.unique(codes)) == n


def test_generate_codes_n_zero(tools):
    codes = tools.generate_alphanum_codes(n=0, length=8)
    assert isinstance(codes, np.ndarray)
    assert codes.shape == (0,)


def test_generate_codes_length_zero(tools):
    codes = tools.generate_alphanum_codes(n=3, length=0)
    assert isinstance(codes, np.ndarray)
    assert all(code == "" for code in codes)


def test_generate_codes_negative_n(tools):
    with pytest.raises(ValueError):
        tools.generate_alphanum_codes(n=-1)


def test_generate_codes_negative_length(tools):
    with pytest.raises(ValueError):
        tools.generate_alphanum_codes(n=10, length=-5)
