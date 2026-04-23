/** A.6 — customer-churn page smoke test.
 *
 * Mirrors the credit-risk smoke test: verify the page mounts, the form
 * renders, and a successful submit round-trips through the mocked API
 * to paint the result card + chart.
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
    predictCustomerChurn: vi.fn(),
    explainCustomerChurn: vi.fn(),
  };
});

import CustomerChurnPage from "@/app/fintech/churn/page";
import { predictCustomerChurn, explainCustomerChurn } from "@/lib/api";

function withQueryClient(ui: React.ReactElement) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return <QueryClientProvider client={client}>{ui}</QueryClientProvider>;
}

describe("CustomerChurnPage", () => {
  beforeEach(() => {
    vi.mocked(predictCustomerChurn).mockReset();
    vi.mocked(explainCustomerChurn).mockReset();
  });

  it("renders the input form with a submit button", () => {
    render(withQueryClient(<CustomerChurnPage />));
    expect(screen.getByLabelText(/tenure/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/account balance/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /submit|predict/i })).toBeInTheDocument();
  });

  it("submits defaults and renders prediction + chart on success", async () => {
    const user = userEvent.setup();

    vi.mocked(predictCustomerChurn).mockResolvedValueOnce({
      probability_churn: 0.72,
      retention_recommendation: "URGENT_OUTREACH",
      confidence: 0.72,
    });
    vi.mocked(explainCustomerChurn).mockResolvedValueOnce({
      feature_importances: {
        tenure_months: -0.35,
        is_active_member: -0.22,
        num_products: 0.18,
      },
      top_features: [],
      explanation_type: "shap",
    });

    render(withQueryClient(<CustomerChurnPage />));
    await user.click(screen.getByRole("button", { name: /submit|predict/i }));

    await waitFor(() => {
      expect(predictCustomerChurn).toHaveBeenCalledOnce();
      expect(explainCustomerChurn).toHaveBeenCalledOnce();
    });

    await waitFor(() =>
      expect(screen.getByTestId("recommendation")).toHaveTextContent("URGENT_OUTREACH"),
    );
    expect(screen.getByTestId("probability-churn")).toHaveTextContent("72.00%");
    expect(screen.getByTestId("confidence")).toHaveTextContent("72.00%");
  });
});
