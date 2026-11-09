import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import clean_text


def test_lowercase_and_punctuation_removed():
    assert clean_text("HELLO, World!!!") == "hello world"


def test_url_replaced_with_marker():
    assert "urltoken" in clean_text("Visit http://spam.com/offer now")


def test_html_removed():
    assert "<" not in clean_text("<html><b>Hi</b></html>")


def test_numbers_replaced():
    assert "numtoken" in clean_text("Win 1000 dollars")


def test_non_string_returns_empty():
    assert clean_text(None) == ""
