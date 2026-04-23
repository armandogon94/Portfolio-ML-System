/** Phase A.4 — dental no-show page smoke test.
 *
 * Mirrors the credit-risk smoke test: mounts the page, verifies the form
 * renders, and that a successful submit paints the NoShowResult card +
 * the ExplainabilityChart.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    predictDentalNoShow: vi.fn(),
    explainDentalNoShow: vi.fn(),
  };
});

import DentalNoShowPage from "@/app/dental/no-show/page";
import { predictDentalNoShow, explainDentalNoShow } from "@/lib/api";

function withQueryClient(ui: React.ReactElement) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return <QueryClientProvider client={client}>{ui}</QueryClientProvider>;
}

describe("DentalNoShowPage", () => {
  beforeEach(() => {
    vi.mocked(predictDentalNoShow).mockReset();
    vi.mocked(explainDentalNoShow).mockReset();
  });

  it("renders the input form with a submit button", () => {
    render(withQueryClient(<DentalNoShowPage />));
    expect(screen.getByLabelText(/patient age/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/prior no-shows/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /submit|predict/i })).toBeInTheDocument();
  });

  it("submits defaults and renders prediction + chart on success", async () => {
    const user = userEvent.setup();

    vi.mocked(predictDentalNoShow).mockResolvedValueOnce({
      probability_no_show: 0.55,
      risk_band: "HIGH_RISK",
      confidence: 0.55,
    });
    vi.mocked(explainDentalNoShow).mockResolvedValueOnce({
      feature_importances: {
        prior_no_shows: 0.32,
        distance_km: 0.18,
        prior_appointments: -0.22,
      },
      top_features: [],
      explanation_type: "shap",
    });

    render(withQueryClient(<DentalNoShowPage />));
    await user.click(screen.getByRole("button", { name: /submit|predict/i }));

    await waitFor(() => {
      expect(predictDentalNoShow).toHaveBeenCalledOnce();
      expect(explainDentalNoShow).toHaveBeenCalledOnce();
    });

    await waitFor(() =>
      expect(screen.getByTestId("risk-band")).toHaveTextContent("HIGH_RISK"),
    );
    expect(screen.getByTestId("probability-no-show")).toHaveTextContent("55.00%");
    expect(screen.getByTestId("confidence")).toHaveTextContent("55.00%");
  });
});
