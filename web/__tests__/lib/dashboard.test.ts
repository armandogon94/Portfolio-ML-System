/** A.9.9 — getDashboardRows join logic tests.
 *
 * Verifies the INDUSTRIES × FastAPI /models × MLflow history join
 * produces the right shape. We mock global.fetch (FastAPI) and the
 * mlflow.getRunHistory module export so the unit test is hermetic.
 *
 * The route-handler approach in SPEC §A.9.9 was abandoned because
 * next.config.mjs already wholesale rewrites /api/* to FastAPI;
 * route handlers under that path would never receive requests.
 * Instead getDashboardRows() fetches directly from
 * `${INTERNAL_API_URL}/models` server-side and calls getRunHistory
 * from lib/mlflow.ts directly. Both are documented in
 * lib/dashboard.ts.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock the mlflow module BEFORE importing dashboard.ts so the import
// graph picks up the mock binding.
vi.mock("@/lib/mlflow", () => ({
  getRunHistory: vi.fn(),
}));

import { getDashboardRows } from "@/lib/dashboard";
import { getRunHistory } from "@/lib/mlflow";

const ORIGINAL_FETCH = globalThis.fetch;

beforeEach(() => {
  vi.mocked(getRunHistory).mockReset();
});

afterEach(() => {
  globalThis.fetch = ORIGINAL_FETCH;
});

import { afterEach } from "vitest";

describe("getDashboardRows", () => {
  it("joins INDUSTRIES with FastAPI /models metadata + MLflow history", async () => {
    // FastAPI returns metadata keyed by problem name. We give it a
    // subset (3 of 10) to keep the assertions tight; the rest of
    // the catalog will surface as not_built rows.
    globalThis.fetch = vi.fn().mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          credit_risk: {
            problem: "credit_risk",
            metrics: { test_auc_roc: 0.85 },
            timestamp: "2026-04-12T13:40:00",
          },
          fraud_detection: {
            problem: "fraud_detection",
            metrics: { test_autoencoder_auc_roc: 0.91 },
            timestamp: "2026-04-23T05:00:00",
          },
          h1b_approval: {
            problem: "h1b_approval",
            metrics: { test_auc_roc: 0.79 },
            timestamp: "2026-04-23T04:30:00",
          },
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    ) as unknown as typeof fetch;

    vi.mocked(getRunHistory).mockResolvedValue([
      { endTime: 1, metric: 0.81 },
      { endTime: 2, metric: 0.85 },
    ]);

    const rows = await getDashboardRows();
    // 6 industries × ~3-4 models = 20 entries (registry source-of-truth).
    expect(rows.length).toBeGreaterThanOrEqual(20);

    // Find the credit-risk row and verify its shape.
    const cr = rows.find((r) => r.modelSlug === "credit-risk");
    expect(cr).toBeDefined();
    expect(cr?.industrySlug).toBe("fintech");
    expect(cr?.industryTitle).toBe("Fintech");
    expect(cr?.status).toBe("ready");
    expect(cr?.href).toBe("/fintech/credit-risk");
    expect(cr?.keyMetric?.value).toBe(0.85);
    expect(cr?.keyMetric?.label).toMatch(/auc/i);
    expect(cr?.lastTrained).toBe("2026-04-12T13:40:00");
    expect(cr?.history).toEqual([0.81, 0.85]);

    // A model with no checkpoint in /models still shows up but as
    // not_built (when industries.ts also has it as ready: false) or
    // ready (when industries.ts says ready but checkpoint absent — we
    // treat as not_built since /models is the runtime truth).
    const treatmentPlan = rows.find((r) => r.modelSlug === "treatment-plan");
    expect(treatmentPlan?.status).toBe("not_built");
    expect(treatmentPlan?.keyMetric).toBeNull();
    expect(treatmentPlan?.history).toEqual([]);
  });

  it("returns rows with empty history when MLflow getRunHistory throws", async () => {
    globalThis.fetch = vi.fn().mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          credit_risk: {
            problem: "credit_risk",
            metrics: { test_auc_roc: 0.85 },
            timestamp: "2026-04-12T13:40:00",
          },
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    ) as unknown as typeof fetch;

    vi.mocked(getRunHistory).mockRejectedValue(new Error("MLflow down"));

    const rows = await getDashboardRows();
    const cr = rows.find((r) => r.modelSlug === "credit-risk");
    // Sparkline degrades gracefully — no crash, just an empty array.
    expect(cr?.history).toEqual([]);
    // Other fields still populate from FastAPI.
    expect(cr?.status).toBe("ready");
    expect(cr?.keyMetric?.value).toBe(0.85);
  });

  it("returns all-not_built rows when FastAPI /models is unreachable", async () => {
    globalThis.fetch = vi.fn().mockRejectedValue(new Error("Connection refused"));
    vi.mocked(getRunHistory).mockResolvedValue([]);

    const rows = await getDashboardRows();
    expect(rows.length).toBeGreaterThanOrEqual(20);
    // Without any /models data, no row can be ready — every row falls
    // back to not_built so the dashboard still renders.
    expect(rows.every((r) => r.status === "not_built")).toBe(true);
  });
});
