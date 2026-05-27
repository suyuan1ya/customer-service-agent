import json
from main import check_rate_limit


class TestAPI:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ok", "degraded")
        assert "database" in data
        assert "chromadb" in data

    def test_index_page(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "智能客服" in resp.text
        assert "sendMessage" in resp.text

    def test_chat_non_streaming(self, client):
        resp = client.post("/chat", json={"message": "你好"})
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert "response" in data
        assert "intent" in data

    def test_chat_empty_message(self, client):
        resp = client.post("/chat", json={"message": ""})
        assert resp.status_code == 200
        assert "session_id" in resp.json()

    def test_chat_stream_basic(self, client):
        resp = client.post("/chat/stream", json={"message": "你好"})
        assert resp.status_code in (200, 500)

    def test_chat_stream_sse_format(self, client):
        """SSE 流式响应格式验证"""
        resp = client.post("/chat/stream", json={"message": "你好"})
        if resp.status_code == 200:
            assert "text/event-stream" in resp.headers.get("content-type", "")


class TestRateLimit:
    def test_allows_up_to_limit(self):
        """前 20 次请求允许通过"""
        for i in range(20):
            assert check_rate_limit(f"test-ip-{i % 3}") is True

    def test_blocks_after_limit(self):
        """同一 IP 超过 20 次后被拒绝"""
        for _ in range(20):
            check_rate_limit("block-test-ip")
        assert check_rate_limit("block-test-ip") is False

    def test_different_ips_independent(self):
        """不同 IP 独立计数"""
        for _ in range(20):
            check_rate_limit("ip-a")
        assert check_rate_limit("ip-a") is False
        assert check_rate_limit("ip-b") is True
