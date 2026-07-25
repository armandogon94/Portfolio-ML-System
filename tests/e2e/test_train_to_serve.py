"""End to end on committed fixtures: adapter -> train -> checkpoint -> HTTP predict.

Runs in a couple of seconds, needs no credentials and no network. It proves the
wiring, never a metric: the fixtures are synthetic and a model trained on them
should be no better than chance. That is asserted explicitly below, because a good
score here would mean the fixture generator had leaked the label.
"""

from __future__ import annotations

import json
import time

import joblib
from fastapi.testclient import TestClient

from src.data.split import make_splits
from src.serving.registry import CheckpointRegistry
from src.training.tabular import TabularTrainer

BUDGET_SECONDS = 30


def test_fixtures_to_http_prediction(tmp_path, monkeypatch):
    started = time.time()

    trainer = TabularTrainer("churn", sample=True)
    frame = trainer.load_data()
    assert len(frame) > 100

    split = make_splits(frame, {"type": "random", "test_size": 0.25}, "is_attrited", 42)[0]
    data = trainer.build_matrix(frame, split)
    model = trainer.train(data)
    metrics = trainer.evaluate(model, data)
    trainer.finish()

    # A model trained on independent-noise labels must be near chance. Anything
    # much better means the fixture generator leaked the label into a feature.
    assert 0.25 <= metrics["test_roc_auc"] <= 0.75, (
        f"fixture ROC-AUC {metrics['test_roc_auc']:.3f} is too good — the CI fixture "
        f"has a label leak. Regenerate with scripts/make_fixtures.py."
    )

    root = tmp_path / "checkpoints"
    directory = root / "churn"
    directory.mkdir(parents=True)
    joblib.dump(model, directory / "model.joblib")
    joblib.dump(trainer.feature_bundle(), directory / "features.joblib")
    (directory / "metadata.json").write_text(
        json.dumps(
            {
                "problem": "churn",
                "model_type": "lightgbm",
                "seed": 42,
                "git_sha": "e2e",
                "dataset": {"slug": "SYNTHETIC-CI-FIXTURE-NOT-REAL-DATA"},
                "metrics": metrics,
            },
            default=str,
        )
    )

    import src.serving.api as api

    monkeypatch.setattr(api, "registry", CheckpointRegistry(root))
    client = TestClient(api.app)

    assert client.get("/health").json()["models"]["churn"]["available"] is True

    prediction = client.post("/predict/churn", json={}).json()
    assert 0.0 <= prediction["attrition_probability"] <= 1.0

    explanation = client.post("/explain/churn", json={}).json()
    assert explanation["top_features"]

    elapsed = time.time() - started
    assert elapsed < BUDGET_SECONDS, f"e2e took {elapsed:.1f}s, budget is {BUDGET_SECONDS}s"
