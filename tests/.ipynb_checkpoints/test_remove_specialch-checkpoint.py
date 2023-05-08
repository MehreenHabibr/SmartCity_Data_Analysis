def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-Z0-9\s]|\[|\]' if not remove_digits else r'[^a-zA-Z\s]|\[|\]'
    text = re.sub(pattern, '', text)
    return text
import re

def test_remove_special_characters():
    
    
    # Test case 4: Empty string
    text = ""
    expected_output = ""
    assert remove_special_characters(text) == expected_output

