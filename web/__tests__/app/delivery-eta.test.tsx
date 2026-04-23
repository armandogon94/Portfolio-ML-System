/** A.7 — delivery-ETA page smoke test.
 *
 * Mirrors the credit-risk smoke: mount the page, mock the API layer,
 * submit defaults, and verify the ETA card + explanation chart paint.
 * Component-level tests live under __tests__/components; this only
 * checks that the page wires everything together correctly.
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
    predictDeliveryEta: vi.fn(),
    explainDeliveryEta: vi.fn(),
  };
});

import DeliveryEtaPage from "@/app/logistics/eta/page";
import { predictDeliveryEta, explainDeliveryEta } from "@/lib/api";

function withQueryClient(ui: React.ReactElement) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return <QueryClientProvider client={client}>{ui}</QueryClientProvider>;
}

describe("DeliveryEtaPage", () => {
  beforeEach(() => {
    vi.mocked(predictDeliveryEta).mockReset();
    vi.mocked(explainDeliveryEta).mockReset();
  });

  it("renders the input form with a submit button", () => {
    render(withQueryClient(<DeliveryEtaPage />));
    expect(screen.getByLabelText(/distance/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/package weight/i)).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /submit|predict/i }),
    ).toBeInTheDocument();
  });

  it("submits defaults and renders ETA + confidence band on success", async () => {
    const user = userEvent.setup();

    vi.mocked(predictDeliveryEta).mockResolvedValueOnce({
      eta_hours: 12.5,
      confidence_interval: [10.0, 15.0],
    });
    vi.mocked(explainDeliveryEta).mockResolvedValueOnce({
      feature_importances: {
        distance_km: 2.5,
        traffic_congestion: 0.8,
        weather_severity: -0.3,
      },
      top_features: [],
      explanation_type: "shap",
    });

    render(withQueryClient(<DeliveryEtaPage />));
    await user.click(screen.getByRole("button", { name: /submit|predict/i }));

    await waitFor(() => {
      expect(predictDeliveryEta).toHaveBeenCalledOnce();
      expect(explainDeliveryEta).toHaveBeenCalledOnce();
    });

    // Result card paints — ETA formatted as "N.N hours" and CI as "(lo – hi)"
    await waitFor(() =>
      expect(screen.getByTestId("eta-hours")).toHaveTextContent("12.5 hours"),
    );
    expect(screen.getByTestId("confidence-interval")).toHaveTextContent(
      "(10.0 – 15.0)",
    );
  });
});
