/** A.3 — rental-price page smoke test.
 *
 * Mirrors the credit-risk smoke test: mount the page, check the form
 * renders, mock the API client, submit, assert the predicted nightly
 * rate and confidence interval paint.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    predictRentalPrice: vi.fn(),
    explainRentalPrice: vi.fn(),
  };
});

import RentalPricePage from "@/app/real-estate/rental-price/page";
import { predictRentalPrice, explainRentalPrice } from "@/lib/api";

function withQueryClient(ui: React.ReactElement) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return <QueryClientProvider client={client}>{ui}</QueryClientProvider>;
}

describe("RentalPricePage", () => {
  beforeEach(() => {
    vi.mocked(predictRentalPrice).mockReset();
    vi.mocked(explainRentalPrice).mockReset();
  });

  it("renders the input form with a submit button", () => {
    render(withQueryClient(<RentalPricePage />));
    expect(screen.getByLabelText(/bedrooms/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/square feet/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/peer nightly rate/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /submit|predict/i })).toBeInTheDocument();
  });

  it("submits defaults and renders nightly rate + CI + chart on success", async () => {
    const user = userEvent.setup();

    vi.mocked(predictRentalPrice).mockResolvedValueOnce({
      predicted_rate: 184.5,
      confidence_interval: [166.05, 202.95],
    });
    vi.mocked(explainRentalPrice).mockResolvedValueOnce({
      feature_importances: {
        square_feet: 42.1,
        location_tier: 18.3,
        distance_to_downtown_km: -5.7,
      },
      top_features: [],
      explanation_type: "shap",
    });

    render(withQueryClient(<RentalPricePage />));
    await user.click(screen.getByRole("button", { name: /submit|predict/i }));

    await waitFor(() => {
      expect(predictRentalPrice).toHaveBeenCalledOnce();
      expect(explainRentalPrice).toHaveBeenCalledOnce();
    });

    await waitFor(() =>
      expect(screen.getByTestId("predicted-rate")).toHaveTextContent("$184.50"),
    );
    expect(screen.getByTestId("confidence-interval")).toHaveTextContent("$166.05");
    expect(screen.getByTestId("confidence-interval")).toHaveTextContent("$202.95");
  });
});
