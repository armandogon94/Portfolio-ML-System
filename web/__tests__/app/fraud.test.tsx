/** A.9.4 — fraud-detection page smoke test.
 *
 * Mirrors the credit-risk smoke test pattern: ModelForm + result card +
 * ExplainabilityChart all have their own thorough unit tests. Here we
 * just verify the page mounts, the form renders (including the
 * categorical merchant_category select), and submit round-trips through
 * mocked api.predictFraud + api.explainFraud to paint the FraudResult
 * card with the 4-tier risk_level (LOW / MEDIUM / HIGH / CRITICAL).
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    predictFraud: vi.fn(),
    explainFraud: vi.fn(),
  };
});

import FraudPage from "@/app/fintech/fraud/page";
import { predictFraud, explainFraud } from "@/lib/api";

function withQueryClient(ui: React.ReactElement) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return <QueryClientProvider client={client}>{ui}</QueryClientProvider>;
}

describe("FraudPage", () => {
  beforeEach(() => {
    vi.mocked(predictFraud).mockReset();
    vi.mocked(explainFraud).mockReset();
  });

  it("renders the input form including the merchant-category select", () => {
    render(withQueryClient(<FraudPage />));
    // Categorical select — proves ModelForm's new "select" type works.
    expect(screen.getByLabelText(/merchant category/i)).toBeInTheDocument();
    // A representative numeric field
    expect(screen.getByLabelText(/transaction amount/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /submit|analyze|predict/i })).toBeInTheDocument();
  });

  it("submits defaults and renders fraud result + chart on success", async () => {
    const user = userEvent.setup();

    vi.mocked(predictFraud).mockResolvedValueOnce({
      fraud_probability: 0.62,
      risk_level: "HIGH",
      reconstruction_error: 0.34,
      anomaly_threshold: 0.18,
      is_anomaly_autoencoder: true,
      is_anomaly_isolation_forest: true,
      isolation_forest_score: 0.41,
    });
    vi.mocked(explainFraud).mockResolvedValueOnce({
      feature_importances: {
        amount_vs_avg_ratio: 0.55,
        distance_from_home: 0.30,
        hour_of_day: 0.10,
      },
      top_features: [],
      explanation_type: "gradient",
    });

    render(withQueryClient(<FraudPage />));
    await user.click(screen.getByRole("button", { name: /submit|analyze|predict/i }));

    await waitFor(() => {
      expect(predictFraud).toHaveBeenCalledOnce();
      expect(explainFraud).toHaveBeenCalledOnce();
    });

    // The 4-tier risk badge — HIGH = red-tinted styling, just verify the text.
    await waitFor(() => expect(screen.getByTestId("risk-level")).toHaveTextContent("HIGH"));
    expect(screen.getByTestId("fraud-probability")).toHaveTextContent("62.00%");
  });
});
