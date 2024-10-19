import unittest
# from loguru import logger
from nano_graphrag._utils import convert_response_to_json  

class TestJSONExtraction(unittest.TestCase):

    def setUp(self):
        """Set up runs before each test case."""
        ...

    def test_standard_json(self):
        """Test standard JSON extraction."""
        response = '''
        {
            "reasoning": "This is a test.",
            "answer": 42,
            "data": {"key1": "value1", "key2": "value2"}
        }
        '''
        expected = {
            "reasoning": "This is a test.",
            "answer": 42,
            "data": {"key1": "value1", "key2": "value2"}
        }
        self.assertEqual(convert_response_to_json(response), expected)

    def test_non_standard_json_without_quotes(self):
        """Test non-standard JSON without quotes on numbers and booleans."""
        response = '''
        {
            "reasoning": "Boolean and numbers test.",
            "answer": 42,
            "isCorrect": true,
            "data": {key1: value1}
        }
        '''
        expected = {
            "reasoning": "Boolean and numbers test.",
            "answer": 42,
            "isCorrect": True,
            "data": {"key1": "value1"}
        }
        self.assertEqual(convert_response_to_json(response), expected)

    def test_nested_json(self):
        """Test extraction of nested JSON objects."""
        response = '''
        {
            "reasoning": "Nested structure.",
            "answer": 42,
            "data": {"nested": {"key": "value"}}
        }
        '''
        expected = {
            "reasoning": "Nested structure.",
            "answer": 42,
            "data": {
                "nested": {"key": "value"}
            }
        }
        self.assertEqual(convert_response_to_json(response), expected)

    def test_malformed_json(self):
        """Test handling of malformed JSON."""
        response = '''
        Some text before JSON
        {
            "reasoning": "This is malformed.",
            "answer": 42,
            "data": {"key": "value"}
        }
        Some text after JSON
        '''
        expected = {
            "reasoning": "This is malformed.",
            "answer": 42,
            "data": {"key": "value"}
        }
        self.assertEqual(convert_response_to_json(response), expected)

    def test_incomplete_json(self):
        """Test handling of incomplete JSON."""
        response = '''
        {
            "reasoning": "Incomplete structure",
            "answer": 42
        '''
        expected = {
            "reasoning": "Incomplete structure",
            "answer": 42
        }
        self.assertEqual(convert_response_to_json(response), expected)

    def test_value_with_special_characters(self):
        """Test JSON with special characters in values."""
        response = '''
        {
            "reasoning": "Special characters !@#$%^&*()",
            "answer": 42,
            "data": {"key": "value with special characters !@#$%^&*()"}
        }
        '''
        expected = {
            "reasoning": "Special characters !@#$%^&*()",
            "answer": 42,
            "data": {"key": "value with special characters !@#$%^&*()"}
        }
        self.assertEqual(convert_response_to_json(response), expected)

    def test_boolean_and_null_values(self):
        """Test JSON with boolean and null values."""
        response = '''
        {
            "reasoning": "Boolean and null test.",
            "isCorrect": true,
            "isWrong": false,
            "unknown": null,
            "answer": 42
        }
        '''
        expected = {
            "reasoning": "Boolean and null test.",
            "isCorrect": True,
            "isWrong": False,
            "unknown": None,
            "answer": 42
        }
        self.assertEqual(convert_response_to_json(response), expected)

if __name__ == "__main__":
    unittest.main()