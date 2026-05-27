from main import format_sse
import json


class TestFormatSSE:
    def test_basic_format(self):
        result = format_sse({"event": "token", "data": '{"token":"你好"}'})
        assert result.startswith("data: ")
        assert result.endswith("\n\n")

    def test_parseable_json(self):
        """输出的 data 行可以被 JS 端正确解析"""
        result = format_sse({"event": "start", "data": '{"session_id":"abc"}'})
        data_line = result.strip().split("\n")[0]
        assert data_line.startswith("data: ")
        parsed = json.loads(data_line[len("data: "):])
        assert parsed["event"] == "start"
        assert json.loads(parsed["data"]) == {"session_id": "abc"}

    def test_double_newline_separator(self):
        result = format_sse({"event": "done", "data": "{}"})
        assert result.endswith("\n\n")
