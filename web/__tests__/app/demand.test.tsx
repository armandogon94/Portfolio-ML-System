/** A.9.6 — demand-forecast page smoke test.
 *
 * The Logistics demand model is a 7-day LSTM forecast — a single
 * dropdown input (product), no SHAP explanation (gradient explainer
 * not implemented for the LSTM), and a Recharts LineChart of the
 * 7 predicted values instead of a result-card badge.
 *
 * Smoke test verifies: the product select renders with all 5
 * categories, defaults submit, and the forecast chart shows up
 * after the prediction returns. We assert on the "forecast-chart"
 * test id rather than poking at Recharts internals — the chart's
 * own rendering is library-tested and not our responsibility.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    predictDemand: vi.fn(),
  };
});

import DemandPage from "@/app/logistics/demand/page";
import { predictDemand } from "@/lib/api";

function withQueryClient(ui: React.ReactElement) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return <QueryClientProvider client={client}>{ui}</QueryClientProvider>;
}

describe("DemandPage", () => {
  beforeEach(() => {
    vi.mocked(predictDemand).mockReset();
  });

  it("renders the product select with all 5 categories", () => {
    render(withQueryClient(<DemandPage />));
    const select = screen.getByLabelText(/product/i) as HTMLSelectElement;
    expect(select).toBeInTheDocument();
    // Display labels are humanized: electronics → "Electronics" etc.
    expect(screen.getByRole("option", { name: "Electronics" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "Clothing" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "Groceries" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "Furniture" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "Sports" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /submit|forecast|predict/i })).toBeInTheDocument();
  });

  it("submits defaults and renders the 7-day forecast on success", async () => {
    const user = userEvent.setup();

    vi.mocked(predictDemand).mockResolvedValueOnce({
      product: "electronics",
      forecast_days: 7,
      predictions: [142.5, 148.1, 151.0, 156.7, 159.2, 161.8, 165.0],
      avg_predicted_demand: 154.9,
    });

    render(withQueryClient(<DemandPage />));
    await user.click(screen.getByRole("button", { name: /submit|forecast|predict/i }));

    await waitFor(() => expect(predictDemand).toHaveBeenCalledOnce());

    // Forecast chart should appear once data lands.
    await waitFor(() => expect(screen.getByTestId("forecast-chart")).toBeInTheDocument());
    // Average is shown in the summary line above the chart.
    expect(screen.getByTestId("avg-demand")).toHaveTextContent(/154\.9/);
  });
});
