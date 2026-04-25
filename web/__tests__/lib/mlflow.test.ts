/** A.9.2 — MLflow REST client tests. External fetch is mocked. */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

import { MlflowError, getRunHistory } from "@/lib/mlflow";

// Helper: build a mocked Response with JSON body + given status.
function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const BASE_URL = "http://mlflow.test:5000";

describe("MlflowError", () => {
  it("preserves status and body", () => {
    const err = new MlflowError(500, { error_code: "INTERNAL_ERROR" });
    expect(err).toBeInstanceOf(Error);
    expect(err.status).toBe(500);
    expect(err.body).toEqual({ error_code: "INTERNAL_ERROR" });
    expect(err.message).toContain("500");
  });
});

describe("getRunHistory — happy path", () => {
  const fetchSpy = vi.spyOn(globalThis, "fetch");

  beforeEach(() => {
    fetchSpy.mockReset();
  });
  afterEach(() => {
    fetchSpy.mockReset();
  });

  it("returns history points sorted oldest-first when given valid responses", async () => {
    // 1st call: get-by-name → returns experiment id
    fetchSpy.mockResolvedValueOnce(
      jsonResponse({ experiment: { experiment_id: "42", name: "credit_risk" } }),
    );
    // 2nd call: runs/search → MLflow gives DESC; we expect the client to flip to ASC
    fetchSpy.mockResolvedValueOnce(
      jsonResponse({
        runs: [
          {
            info: { run_id: "newest", end_time: 3000, status: "FINISHED" },
            data: {
              metrics: [{ key: "auc", value: 0.93, timestamp: 3000, step: 0 }],
            },
          },
          {
            info: { run_id: "middle", end_time: 2000, status: "FINISHED" },
            data: {
              metrics: [{ key: "auc", value: 0.91, timestamp: 2000, step: 0 }],
            },
          },
          {
            info: { run_id: "oldest", end_time: 1000, status: "FINISHED" },
            data: {
              metrics: [{ key: "auc", value: 0.88, timestamp: 1000, step: 0 }],
            },
          },
          // A run that's missing the requested metric — should be skipped silently.
          {
            info: { run_id: "no-metric", end_time: 1500, status: "FAILED" },
            data: { metrics: [{ key: "loss", value: 0.5, timestamp: 1500, step: 0 }] },
          },
        ],
      }),
    );

    const points = await getRunHistory("credit_risk", "auc", { baseUrl: BASE_URL });

    expect(points).toEqual([
      { endTime: 1000, metric: 0.88 },
      { endTime: 2000, metric: 0.91 },
      { endTime: 3000, metric: 0.93 },
    ]);

    // Verify the right endpoints were hit, in the right order.
    expect(fetchSpy).toHaveBeenCalledTimes(2);
    const [expUrl] = fetchSpy.mock.calls[0];
    expect(String(expUrl)).toBe(
      `${BASE_URL}/api/2.0/mlflow/experiments/get-by-name?experiment_name=credit_risk`,
    );

    const [searchUrl, searchInit] = fetchSpy.mock.calls[1];
    expect(String(searchUrl)).toBe(`${BASE_URL}/api/2.0/mlflow/runs/search`);
    expect(searchInit?.method).toBe("POST");
    const sentBody = JSON.parse(searchInit?.body as string);
    expect(sentBody.experiment_ids).toEqual(["42"]);
    expect(sentBody.order_by).toEqual(["attributes.end_time DESC"]);
    expect(sentBody.max_results).toBe(10); // default limit
  });

  it("respects the limit option in max_results", async () => {
    fetchSpy.mockResolvedValueOnce(
      jsonResponse({ experiment: { experiment_id: "1", name: "foo" } }),
    );
    fetchSpy.mockResolvedValueOnce(jsonResponse({ runs: [] }));

    await getRunHistory("foo", "auc", { baseUrl: BASE_URL, limit: 25 });

    const sentBody = JSON.parse(fetchSpy.mock.calls[1][1]?.body as string);
    expect(sentBody.max_results).toBe(25);
  });
});

describe("getRunHistory — graceful empty cases", () => {
  const fetchSpy = vi.spyOn(globalThis, "fetch");

  beforeEach(() => {
    fetchSpy.mockReset();
  });

  it("returns empty array when experiment does not exist", async () => {
    // MLflow's documented "not found" shape — 404 + RESOURCE_DOES_NOT_EXIST.
    fetchSpy.mockResolvedValueOnce(
      jsonResponse(
        {
          error_code: "RESOURCE_DOES_NOT_EXIST",
          message: "Could not find experiment with name 'untrained_model'",
        },
        404,
      ),
    );

    const points = await getRunHistory("untrained_model", "auc", { baseUrl: BASE_URL });

    expect(points).toEqual([]);
    // It should NOT have called runs/search after seeing the missing experiment.
    expect(fetchSpy).toHaveBeenCalledTimes(1);
  });

  it("returns empty array when experiment exists but has no runs", async () => {
    fetchSpy.mockResolvedValueOnce(
      jsonResponse({ experiment: { experiment_id: "7", name: "fresh" } }),
    );
    // MLflow omits the `runs` key entirely when there are zero runs.
    fetchSpy.mockResolvedValueOnce(jsonResponse({}));

    const points = await getRunHistory("fresh", "auc", { baseUrl: BASE_URL });
    expect(points).toEqual([]);
  });
});

describe("getRunHistory — error handling", () => {
  const fetchSpy = vi.spyOn(globalThis, "fetch");

  beforeEach(() => {
    fetchSpy.mockReset();
  });

  it("throws MlflowError on non-2xx response", async () => {
    // 500 from MLflow is not the "missing experiment" path — should bubble up.
    fetchSpy.mockResolvedValueOnce(
      jsonResponse({ error_code: "INTERNAL_ERROR", message: "boom" }, 500),
    );

    try {
      await getRunHistory("anything", "auc", { baseUrl: BASE_URL });
      expect.fail("should have thrown MlflowError");
    } catch (err) {
      expect(err).toBeInstanceOf(MlflowError);
      expect((err as MlflowError).status).toBe(500);
      expect((err as MlflowError).body).toEqual({
        error_code: "INTERNAL_ERROR",
        message: "boom",
      });
    }
  });

  it("throws MlflowError on invalid response shape (Zod failure)", async () => {
    // 200 OK but the experiment field is missing — Zod must reject.
    fetchSpy.mockResolvedValueOnce(jsonResponse({ totally_wrong: true }));

    await expect(
      getRunHistory("garbage_shape", "auc", { baseUrl: BASE_URL }),
    ).rejects.toBeInstanceOf(MlflowError);
  });
});
