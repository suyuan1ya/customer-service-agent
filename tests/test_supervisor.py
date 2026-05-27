from agents.supervisor import _parse_intent


class TestParseIntent:
    def test_valid_intents(self):
        """正常 JSON 解析"""
        assert _parse_intent('{"intent":"technical","reason":"bug"}') == {
            "intent": "technical", "reason": "bug"
        }
        assert _parse_intent('{"intent":"aftersales","reason":"退货"}') == {
            "intent": "aftersales", "reason": "退货"
        }
        assert _parse_intent('{"intent":"customer","reason":"查询订单"}') == {
            "intent": "customer", "reason": "查询订单"
        }
        assert _parse_intent('{"intent":"escalate","reason":"投诉"}') == {
            "intent": "escalate", "reason": "投诉"
        }

    def test_markdown_code_block(self):
        """被 markdown 代码块包裹的 JSON"""
        raw = '```json\n{"intent":"technical","reason":"问题"}\n```'
        assert _parse_intent(raw) == {"intent": "technical", "reason": "问题"}

    def test_code_block_no_lang(self):
        """无语言标记的代码块"""
        raw = '```\n{"intent":"customer","reason":"问订单"}\n```'
        assert _parse_intent(raw) == {"intent": "customer", "reason": "问订单"}

    def test_extra_text_around_json(self):
        """JSON 前后有额外文本"""
        raw = '分析后结果是: {"intent":"aftersales","reason":"退款"} 需要处理'
        assert _parse_intent(raw) == {"intent": "aftersales", "reason": "退款"}

    def test_invalid_intent_fallback(self):
        """无效 intent 回退到 customer"""
        assert _parse_intent('{"intent":"unknown","reason":"x"}') == {
            "intent": "customer", "reason": "x"
        }

    def test_malformed_json_fallback(self):
        """完全不可解析的 JSON 回退到 customer"""
        result = _parse_intent("今天天气真好")
        assert result["intent"] == "customer"
        assert "fallback" in result["reason"]

    def test_no_json_fallback(self):
        """无 JSON 对象的文本回退"""
        result = _parse_intent("请帮我转人工")
        assert result["intent"] == "customer"
        assert "fallback" in result["reason"]

    def test_empty_string(self):
        """空字符串回退"""
        result = _parse_intent("")
        assert result["intent"] == "customer"
