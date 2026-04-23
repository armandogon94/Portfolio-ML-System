/** A.2.4 — typed API client tests. External fetch is mocked. */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

import { ApiError, predictCreditRisk, explainCreditRisk } from "@/lib/api";

describe("ApiError", () => {
  it("preserves status and body", () => {
    const err = new ApiError(422, { detail: "bad input" });
    expect(err).toBeInstanceOf(Error);
    expect(err.status).toBe(422);
    expect(err.body).toEqual({ detail: "bad input" });
    expect(err.message).toContain("422");
  });
});

describe("API client — relative proxy path", () => {
  const fetchSpy = vi.spyOn(globalThis, "fetch");

  beforeEach(() => {
    fetchSpy.mockReset();
  });
  afterEach(() => {
    fetchSpy.mockReset();
  });

  it("predictCreditRisk POSTs to /api/predict/credit-risk (relative, not absolute)", async () => {
    fetchSpy.mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          risk_score: 0.42,
          recommendation: "REVIEW",
          confidence: 0.58,
          default_probability: 0.42,
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    );

    await predictCreditRisk({
      age: 35,
      annual_income: 65000,
      credit_score: 700,
      num_open_accounts: 3,
      payment_history_pct: 85,
      debt_to_income_ratio: 0.3,
      employment_years: 8,
      loan_amount: 25000,
    });

    expect(fetchSpy).toHaveBeenCalledOnce();
    const [url, init] = fetchSpy.mock.calls[0];
    expect(url).toBe("/api/predict/credit-risk");
    // Relative path, not absolute — the browser never crosses origins
    expect(String(url).startsWith("http")).toBe(false);
    expect(init?.method).toBe("POST");
    expect(init?.headers).toEqual({ "Content-Type": "application/json" });
    expect(JSON.parse(init?.body as string).credit_score).toBe(700);
  });

  it("explainCreditRisk POSTs to /api/explain/credit-risk", async () => {
    fetchSpy.mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          feature_importances: { credit_score: -0.3, debt_to_income_ratio: 0.2 },
          top_features: [
            { feature: "credit_score", importance: -0.3 },
            { feature: "debt_to_income_ratio", importance: 0.2 },
          ],
          explanation_type: "shap",
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    );

    const result = await explainCreditRisk({
      age: 35,
      annual_income: 65000,
      credit_score: 700,
      num_open_accounts: 3,
      payment_history_pct: 85,
      debt_to_income_ratio: 0.3,
      employment_years: 8,
      loan_amount: 25000,
    });

    const url = fetchSpy.mock.calls[0][0];
    expect(url).toBe("/api/explain/credit-risk");
    expect(result.explanation_type).toBe("shap");
    expect(result.feature_importances.credit_score).toBe(-0.3);
  });
});

describe("API client — error handling", () => {
  const fetchSpy = vi.spyOn(globalThis, "fetch");

  beforeEach(() => {
    fetchSpy.mockReset();
  });

  it("throws ApiError with status + body on 4xx", async () => {
    fetchSpy.mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: "validation failed" }), {
        status: 422,
        headers: { "Content-Type": "application/json" },
      }),
    );

    try {
      await predictCreditRisk({} as never);
      expect.fail("should have thrown ApiError");
    } catch (err) {
      expect(err).toBeInstanceOf(ApiError);
      expect((err as ApiError).status).toBe(422);
      expect((err as ApiError).body).toEqual({ detail: "validation failed" });
    }
  });

  it("throws ApiError on 5xx too", async () => {
    fetchSpy.mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: "internal error" }), {
        status: 500,
      }),
    );

    await expect(predictCreditRisk({} as never)).rejects.toBeInstanceOf(ApiError);
  });

  it("throws on malformed response body (Zod parse failure)", async () => {
    // Response is 200 OK but missing required fields — Zod should reject
    fetchSpy.mockResolvedValueOnce(
      new Response(JSON.stringify({ foo: "bar" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    await expect(predictCreditRisk({} as never)).rejects.toThrow();
  });
});
