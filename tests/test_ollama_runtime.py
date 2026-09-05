from traffic_data_agent.ollama_runtime import parse_model_names


def test_parse_model_names_accepts_current_ollama_shape():
    payload = {
        "models": [
            {"name": "qwen2.5:3b"},
            {"model": "llama3.2:latest"},
            {"name": "qwen2.5:3b"},
            {},
        ]
    }
    assert parse_model_names(payload) == ["llama3.2:latest", "qwen2.5:3b"]


def test_parse_model_names_handles_empty_response():
    assert parse_model_names({}) == []
