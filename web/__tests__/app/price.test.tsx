/** A.9.5 — price-prediction page smoke test.
 *
 * Mirrors rental-price (A.3) — both are regression models with a
 * predicted-amount + low/high range. Page-level test only verifies
 * the form mounts, defaults submit, and the result card renders the
 * predicted price as USD plus the confidence range.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    predictPrice: vi.fn(),
    explainPrice: vi.fn(),
  };
});

import PricePage from "@/app/real-estate/price/page";
import { predictPrice, explainPrice } from "@/lib/api";

function withQueryClient(ui: React.ReactElement) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return <QueryClientProvider client={client}>{ui}</QueryClientProvider>;
}

describe("PricePage", () => {
  beforeEach(() => {
    vi.mocked(predictPrice).mockReset();
    vi.mocked(explainPrice).mockReset();
  });

  it("renders the input form with key housing fields", () => {
    render(withQueryClient(<PricePage />));
    expect(screen.getByLabelText(/square feet/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/bedrooms/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/year built/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /submit|predict/i })).toBeInTheDocument();
  });

  it("submits defaults and renders predicted price + range on success", async () => {
    const user = userEvent.setup();

    vi.mocked(predictPrice).mockResolvedValueOnce({
      predicted_price: 420_000,
      price_range_low: 378_000,
      price_range_high: 462_000,
    });
    vi.mocked(explainPrice).mockResolvedValueOnce({
      feature_importances: {
        square_feet: 0.45,
        neighborhood_tier: 0.3,
        year_built: 0.15,
      },
      top_features: [],
      explanation_type: "shap",
    });

    render(withQueryClient(<PricePage />));
    await user.click(screen.getByRole("button", { name: /submit|predict/i }));

    await waitFor(() => {
      expect(predictPrice).toHaveBeenCalledOnce();
      expect(explainPrice).toHaveBeenCalledOnce();
    });

    // Predicted price formatted as a dollar string with thousands grouping.
    await waitFor(() =>
      expect(screen.getByTestId("predicted-price")).toHaveTextContent(/\$420,000/),
    );
    expect(screen.getByTestId("price-range")).toHaveTextContent(/\$378,000.*\$462,000/);
  });
});
