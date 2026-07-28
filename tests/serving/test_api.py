"""Every route, including the failure paths."""

from __future__ import annotations


def test_health_is_ok_with_a_checkpoint_present(api_client):
    body = api_client.get("/health").json()
    assert body["status"] == "ok"
    assert body["checkpoints_found"] == 1
    assert body["models"]["churn"]["available"] is True


def test_health_reports_untrained_models_as_unavailable(api_client):
    """A fresh clone must not look broken; it must look untrained."""
    body = api_client.get("/health").json()
    assert body["models"]["fraud"]["available"] is False
    assert body["models"]["credit_risk"]["available"] is False


def test_models_exposes_provenance(api_client):
    body = api_client.get("/models").json()
    assert "churn" in body
    assert body["churn"]["model_type"] == "lightgbm"
    assert "git_sha" in body["churn"]


def test_predict_churn_returns_a_probability_and_an_action(api_client):
    response = api_client.post("/predict/churn", json={"Customer_Age": 45})
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["attrition_probability"] <= 1.0
    assert body["retention_action"] in {"NO_ACTION", "PROACTIVE_CHECKIN", "URGENT_OUTREACH"}
    assert "feature_frame" not in body, "internal frame must not leak into the response"


def test_explain_churn_returns_shap_contributions(api_client):
    response = api_client.post("/explain/churn", json={"Customer_Age": 45})
    assert response.status_code == 200
    body = response.json()
    assert body["explanation_type"] == "shap"
    assert body["top_features"]


def test_an_untrained_model_is_503_not_500(api_client):
    """503 is honest: the service is fine, the model does not exist yet."""
    for path in ("/predict/fraud", "/predict/credit-risk"):
        response = api_client.post(path, json={})
        assert response.status_code == 503, path
        assert "train.py" in response.json()["detail"]


def test_an_invalid_payload_is_422(api_client):
    response = api_client.post("/predict/churn", json={"Credit_Limit": -5})
    assert response.status_code == 422


def test_a_non_numeric_field_is_422(api_client):
    response = api_client.post("/predict/churn", json={"Customer_Age": "middle-aged"})
    assert response.status_code == 422


def test_defaults_alone_are_a_valid_request(api_client):
    """The schema defaults are seeded demo values, so an empty body must work."""
    assert api_client.post("/predict/churn", json={}).status_code == 200


def test_the_route_surface_is_exactly_eight(api_client):
    """21 routes for 4 usable checkpoints was the old shape. Eight is the new one."""
    paths = {
        route["path"]
        for route in api_client.get("/openapi.json").json()["paths"].keys()
        for route in [{"path": route}]
    }
    assert paths == {
        "/predict/fraud",
        "/predict/credit-risk",
        "/predict/churn",
        "/explain/fraud",
        "/explain/credit-risk",
        "/explain/churn",
        "/models",
        "/health",
    }
