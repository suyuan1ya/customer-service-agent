from orchestrator.graph import route_by_intent


class TestRouting:
    def test_route_technical(self):
        assert route_by_intent({"intent": "technical"}) == "technical_worker"

    def test_route_aftersales(self):
        assert route_by_intent({"intent": "aftersales"}) == "aftersales_worker"

    def test_route_customer(self):
        assert route_by_intent({"intent": "customer"}) == "customer_worker"

    def test_route_escalate(self):
        assert route_by_intent({"intent": "escalate"}) == "finish"

    def test_route_missing_intent(self):
        """state 中没有 intent 字段时回退到 customer"""
        assert route_by_intent({}) == "customer_worker"

    def test_route_unknown_intent(self):
        """未知 intent 回退到 customer"""
        assert route_by_intent({"intent": "unknown"}) == "customer_worker"
