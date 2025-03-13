import pytest
from src.preprocessing import clean_text

def test_clean_text():
    assert clean_text("Hello World! http://example.com") == "hello world"
    assert clean_text("I love this product!!!") == "love product"
    assert clean_text("") == ""