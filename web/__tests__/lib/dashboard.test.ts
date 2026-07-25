/**
 * getDashboardRows joins the industry registry, the FastAPI `/models` response
 * and MLflow run history into the rows the dashboard renders.
 *
 * The dependencies are mocked so the test is hermetic. What it actually asserts
 * is degradation behaviour: with no checkpoints, or with MLflow down, the
 * dashboard must still render rather than 500. On a fresh clone — which is the
 * current state of this repository — every row is `not_built`, and that is the
 * honest display, not a bug.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// Mocked BEFORE importing dashboard.ts so the import graph binds the mock.
vi.mock("@/lib/mlflow", () => ({ getRunHistory: vi.fn() }));

import { getDashboardRows } from "@/lib/dashboard";
import { getRunHistory } from "@/lib/mlflow";

const ORIGINAL_FETCH = globalThis.fetch;

/** One trained checkpoint, shaped like a real metadata.json. */
const MODELS_RESPONSE = {
  credit_risk: {
    problem: "credit_risk",
    model_type: "lightgbm",
    git_sha: "a1b2c3d4",
    metrics: { test_pr_auc: 0.31, test_roc_auc: 0.71 },
    trained_at: "2026-07-24T13:40:00+00:00",
  },
};

function mockModelsEndpoint(body: unknown) {
  globalThis.fetch = vi.fn().mockResolvedValueOnce(
    new Response(JSON.stringify(body), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    }),
  ) as unknown as typeof fetch;
}

beforeEach(() => {
  vi.mocked(getRunHistory).mockReset();
});

afterEach(() => {
  globalThis.fetch = ORIGINAL_FETCH;
});

describe("getDashboardRows", () => {
  it("returns exactly one row per catalogued model", async () => {
    mockModelsEndpoint({});
    vi.mocked(getRunHistory).mockResolvedValue([]);

    const rows = await getDashboardRows();
    expect(rows.map((r) => r.modelSlug).sort()).toEqual([
      "churn",
      "credit-risk",
      "fraud",
    ]);
  });

  it("joins checkpoint metadata into the matching row", async () => {
    mockModelsEndpoint(MODELS_RESPONSE);
    vi.mocked(getRunHistory).mockResolvedValue([
      { endTime: 1, metric: 0.28 },
      { endTime: 2, metric: 0.31 },
    ]);

    const row = (await getDashboardRows()).find((r) => r.modelSlug === "credit-risk");
    expect(row?.industrySlug).toBe("fintech");
    expect(row?.status).toBe("ready");
    expect(row?.href).toBe("/fintech/credit-risk");
    expect(row?.keyMetric?.value).toBe(0.31);
    expect(row?.lastTrained).toBe("2026-07-24T13:40:00+00:00");
    expect(row?.history).toEqual([0.28, 0.31]);
  });

  it("surfaces PR-AUC, not ROC-AUC, as the key metric", async () => {
    // At 3.5% positives ROC-AUC is dominated by the true-negative mass; a weak
    // model still scores 0.8+. Showing it as the headline would repeat the
    // mistake this repository was rebuilt to correct.
    mockModelsEndpoint(MODELS_RESPONSE);
    vi.mocked(getRunHistory).mockResolvedValue([]);

    const row = (await getDashboardRows()).find((r) => r.modelSlug === "credit-risk");
    expect(row?.keyMetric?.label).toMatch(/PR-AUC/);
    expect(row?.keyMetric?.value).toBe(0.31);
  });

  it("reads the churn metric from the cross-validated mean", async () => {
    // Churn is scored by 5-fold CV; there is no single test_ value to read.
    mockModelsEndpoint({
      churn: {
        problem: "churn",
        metrics: { cv_pr_auc_mean: 0.66, cv_pr_auc_std: 0.03 },
        trained_at: "2026-07-24T14:00:00+00:00",
      },
    });
    vi.mocked(getRunHistory).mockResolvedValue([]);

    const row = (await getDashboardRows()).find((r) => r.modelSlug === "churn");
    expect(row?.keyMetric?.value).toBe(0.66);
    expect(row?.keyMetric?.label).toContain("5-fold");
  });

  it("marks a model with no checkpoint as not_built", async () => {
    mockModelsEndpoint(MODELS_RESPONSE);
    vi.mocked(getRunHistory).mockResolvedValue([]);

    const row = (await getDashboardRows()).find((r) => r.modelSlug === "fraud");
    expect(row?.status).toBe("not_built");
    expect(row?.keyMetric).toBeNull();
    expect(row?.history).toEqual([]);
  });

  it("keeps rendering when MLflow is down", async () => {
    mockModelsEndpoint(MODELS_RESPONSE);
    vi.mocked(getRunHistory).mockRejectedValue(new Error("MLflow unreachable"));

    const row = (await getDashboardRows()).find((r) => r.modelSlug === "credit-risk");
    // The sparkline is decorative; an outage must not take the page down.
    expect(row?.history).toEqual([]);
    expect(row?.status).toBe("ready");
    expect(row?.keyMetric?.value).toBe(0.31);
  });

  it("keeps rendering when FastAPI is unreachable", async () => {
    globalThis.fetch = vi.fn().mockRejectedValue(new Error("Connection refused"));
    vi.mocked(getRunHistory).mockResolvedValue([]);

    const rows = await getDashboardRows();
    expect(rows).toHaveLength(3);
    expect(rows.every((r) => r.status === "not_built")).toBe(true);
  });

  it("shows every row as not_built on a fresh clone", async () => {
    // This is the repository's actual current state: the code is complete but
    // no model has been trained, because the datasets need Kaggle credentials.
    mockModelsEndpoint({});
    vi.mocked(getRunHistory).mockResolvedValue([]);

    const rows = await getDashboardRows();
    expect(rows.every((r) => r.status === "not_built")).toBe(true);
    expect(rows.every((r) => r.keyMetric === null)).toBe(true);
  });
});
