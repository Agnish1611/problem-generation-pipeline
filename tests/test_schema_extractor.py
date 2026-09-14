from ingestion.models import Example
from ingestion.schema_extractor import infer_schema


def test_infer_schema_from_leetcode_style_examples():
    examples = [
        Example(input="nums = [2,7,11,15], target = 9", output="[0,1]"),
        Example(input="nums = [3,2,4], target = 6", output="[1,2]"),
    ]
    result = infer_schema(examples)
    assert result.input_schema["type"] == "object"
    assert "nums" in result.input_schema["properties"]
    assert "target" in result.input_schema["properties"]
    assert result.confidence > 0.5


def test_infer_schema_from_json_examples():
    examples = [
        Example(input='{"a": 1, "b": 2}', output="3"),
        Example(input='{"a": 3, "b": 4}', output="7"),
    ]
    result = infer_schema(examples)
    assert result.input_schema["type"] == "object"
    assert result.output_schema["type"] == "integer"
    assert result.confidence > 0.8


def test_infer_schema_no_examples_returns_zero_confidence():
    result = infer_schema([])
    assert result.confidence == 0.0
    assert result.input_schema == {}
