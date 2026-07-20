from reasoning_chain.context_compression import compress_step_results, compress_text


def test_compress_text_preserves_final_numeric_value():
    original = "verbose payload " * 100 + "final result 42.75"

    compressed = compress_text(original, max_tokens=20, preserve_final_number=True)

    assert len(compressed) < len(original)
    assert "compressed for context budget" in compressed
    assert "preserved_final_numeric_value: 42.75" in compressed


def test_step_result_compression_does_not_mutate_trace_source():
    output = "weather detail " * 100 + "temperature 32"
    steps = [{"step_id": 1, "output": output, "error": None, "success": True}]

    compressed, compressed_fields = compress_step_results(steps, max_tokens=20)

    assert compressed_fields == 1
    assert steps[0]["output"] == output
    assert compressed[0]["output"] != output
    assert "preserved_final_numeric_value: 32" in compressed[0]["output"]
