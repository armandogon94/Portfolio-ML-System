/**
 * The typed API client. Zod validation at the boundary is the point: a backend
 * shape change must surface here, not as `undefined` inside a chart.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  ApiError,
  apiErrorMessage,
  isUntrainedModelError,
  predictChurn,
  predictCreditRisk,
  predictFraud,
} from "@/lib/api";
import { CHURN_DEFAULTS, CREDIT_RISK_DEFAULTS, FRAUD_DEFAULTS } from "@/lib/schemas";

const FRAUD_RESPONSE = {
  fraud_probability: 0.42,
  risk_band: "MEDIUM",
  recommended_action: "MONITOR",
  model_version: "abc12345",
  trained_on: "ieee-fraud-detection",
};

function mockFetch(body: unknown, status = 200) {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("predict endpoints", () => {
  it("posts to the fraud route and parses the response", async () => {
    const fetchMock = mockFetch(FRAUD_RESPONSE);
    const result = await predictFraud(FRAUD_DEFAULTS);

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/predict/fraud",
      expect.objectContaining({ method: "POST" }),
    );
    expect(result.risk_band).toBe("MEDIUM");
    expect(result.model_version).toBe("abc12345");
  });

  it("posts to the credit-risk route", async () => {
    const fetchMock = mockFetch({
      default_probability: 0.08,
      decision: "APPROVE",
      threshold_basis: "Illustrative cut points.",
      model_version: "abc12345",
      trained_on: "wordsforthewise/lending-club",
    });
    await predictCreditRisk(CREDIT_RISK_DEFAULTS);
    expect(fetchMock.mock.calls[0][0]).toBe("/api/predict/credit-risk");
  });

  it("posts to the churn route and keeps the small-n caveat", async () => {
    mockFetch({
      attrition_probability: 0.7,
      retention_action: "URGENT_OUTREACH",
      caveat: "n = 10,127 and the dataset is easy.",
      model_version: "abc12345",
      trained_on: "sakshigoyal7/credit-card-customers",
    });
    const result = await predictChurn(CHURN_DEFAULTS);
    expect(result.caveat).toContain("10,127");
  });
});

describe("response validation", () => {
  it("rejects a probability outside [0, 1]", async () => {
    mockFetch({ ...FRAUD_RESPONSE, fraud_probability: 1.4 });
    await expect(predictFraud(FRAUD_DEFAULTS)).rejects.toThrow();
  });

  it("rejects an unknown risk band", async () => {
    mockFetch({ ...FRAUD_RESPONSE, risk_band: "APOCALYPTIC" });
    await expect(predictFraud(FRAUD_DEFAULTS)).rejects.toThrow();
  });

  it("rejects a response missing provenance", async () => {
    // A score with no model_version is unreviewable, and that is the whole point.
    const { model_version: _omitted, ...withoutProvenance } = FRAUD_RESPONSE;
    mockFetch(withoutProvenance);
    await expect(predictFraud(FRAUD_DEFAULTS)).rejects.toThrow();
  });
});

describe("error handling", () => {
  it("throws ApiError carrying the status and body", async () => {
    mockFetch({ detail: "boom" }, 500);
    await expect(predictFraud(FRAUD_DEFAULTS)).rejects.toBeInstanceOf(ApiError);
  });

  it("recognises 503 as the untrained-model case", () => {
    expect(isUntrainedModelError(new ApiError(503, { detail: "no checkpoint" }))).toBe(true);
    expect(isUntrainedModelError(new ApiError(422, {}))).toBe(false);
    expect(isUntrainedModelError(new Error("network"))).toBe(false);
  });

  it("surfaces the API's own detail text verbatim", () => {
    // The 503 detail contains the exact commands that fix the problem.
    const detail = "No checkpoint for 'fraud'. Run: uv run python scripts/train.py --model fraud";
    expect(apiErrorMessage(new ApiError(503, { detail }))).toBe(detail);
  });

  it("falls back gracefully for a non-ApiError", () => {
    expect(apiErrorMessage(new Error("offline"))).toBe("offline");
    expect(apiErrorMessage("weird")).toBe("weird");
  });
});
