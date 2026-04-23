/** A.2.7 — credit-risk page smoke test.
 *
 * Page-level composition tests are intentionally minimal: ModelForm,
 * PredictionResult, ExplainabilityChart, and lib/api each have their
 * own thorough tests. Here we only verify that the page mounts, the
 * form renders, and a successful submit round-trips through the
 * mocked API to paint the result card + chart.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

// Mock the API client so the page doesn't try to reach the backend.
vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    predictCreditRisk: vi.fn(),
    explainCreditRisk: vi.fn(),
  };
});

import CreditRiskPage from "@/app/fintech/credit-risk/page";
import { predictCreditRisk, explainCreditRisk } from "@/lib/api";

function withQueryClient(ui: React.ReactElement) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return <QueryClientProvider client={client}>{ui}</QueryClientProvider>;
}

describe("CreditRiskPage", () => {
  beforeEach(() => {
    vi.mocked(predictCreditRisk).mockReset();
    vi.mocked(explainCreditRisk).mockReset();
  });

  it("renders the input form with a submit button", () => {
    render(withQueryClient(<CreditRiskPage />));
    expect(screen.getByLabelText(/annual income/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/credit score/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /submit|predict/i })).toBeInTheDocument();
  });

  it("submits defaults and renders prediction + chart on success", async () => {
    const user = userEvent.setup();

    vi.mocked(predictCreditRisk).mockResolvedValueOnce({
      risk_score: 0.22,
      recommendation: "REVIEW",
      confidence: 0.78,
      default_probability: 0.22,
    });
    vi.mocked(explainCreditRisk).mockResolvedValueOnce({
      feature_importances: {
        credit_score: -0.4,
        debt_to_income_ratio: 0.25,
        annual_income: -0.1,
      },
      top_features: [],
      explanation_type: "shap",
    });

    render(withQueryClient(<CreditRiskPage />));
    await user.click(screen.getByRole("button", { name: /submit|predict/i }));

    // Both endpoints should have been called in parallel
    await waitFor(() => {
      expect(predictCreditRisk).toHaveBeenCalledOnce();
      expect(explainCreditRisk).toHaveBeenCalledOnce();
    });

    // Result card paints with the mocked values
    await waitFor(() => expect(screen.getByTestId("recommendation")).toHaveTextContent("REVIEW"));
    expect(screen.getByTestId("risk-score")).toHaveTextContent("22.00%");
    expect(screen.getByTestId("confidence")).toHaveTextContent("78.00%");
  });
});
