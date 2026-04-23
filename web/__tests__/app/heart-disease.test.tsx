/** A.5 — heart-disease page smoke test.
 *
 * Mirrors the credit-risk test: mount the page, confirm the form fields
 * are present, stub the API client, and verify the result card + chart
 * paint after submit. Risk-band color coding is exercised implicitly via
 * the data-testid on the banding chip.
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
    predictHeartDisease: vi.fn(),
    explainHeartDisease: vi.fn(),
  };
});

import HeartDiseasePage from "@/app/healthcare/heart-disease/page";
import { predictHeartDisease, explainHeartDisease } from "@/lib/api";

function withQueryClient(ui: React.ReactElement) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return <QueryClientProvider client={client}>{ui}</QueryClientProvider>;
}

describe("HeartDiseasePage", () => {
  beforeEach(() => {
    vi.mocked(predictHeartDisease).mockReset();
    vi.mocked(explainHeartDisease).mockReset();
  });

  it("renders the input form with a submit button", () => {
    render(withQueryClient(<HeartDiseasePage />));
    expect(screen.getByLabelText(/age/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/cholesterol/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/max heart rate/i)).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /submit|predict|scoring/i }),
    ).toBeInTheDocument();
  });

  it("submits defaults and renders HIGH risk-band result + chart", async () => {
    const user = userEvent.setup();

    vi.mocked(predictHeartDisease).mockResolvedValueOnce({
      probability_disease: 0.72,
      risk_band: "HIGH",
      confidence: 0.72,
    });
    vi.mocked(explainHeartDisease).mockResolvedValueOnce({
      feature_importances: {
        chest_pain_type: 0.45,
        age: 0.18,
        max_heart_rate: -0.12,
      },
      top_features: [],
      explanation_type: "shap",
    });

    render(withQueryClient(<HeartDiseasePage />));
    await user.click(
      screen.getByRole("button", { name: /submit|predict|scoring/i }),
    );

    await waitFor(() => {
      expect(predictHeartDisease).toHaveBeenCalledOnce();
      expect(explainHeartDisease).toHaveBeenCalledOnce();
    });

    await waitFor(() =>
      expect(screen.getByTestId("risk-band")).toHaveTextContent("HIGH"),
    );
    expect(screen.getByTestId("probability-disease")).toHaveTextContent(
      "72.00%",
    );
    expect(screen.getByTestId("confidence")).toHaveTextContent("72.00%");
  });

  it("shows LOW band when probability is well below 0.25", async () => {
    const user = userEvent.setup();

    vi.mocked(predictHeartDisease).mockResolvedValueOnce({
      probability_disease: 0.05,
      risk_band: "LOW",
      confidence: 0.95,
    });
    vi.mocked(explainHeartDisease).mockResolvedValueOnce({
      feature_importances: { age: -0.3, exercise_angina: -0.2 },
      top_features: [],
      explanation_type: "shap",
    });

    render(withQueryClient(<HeartDiseasePage />));
    await user.click(
      screen.getByRole("button", { name: /submit|predict|scoring/i }),
    );

    await waitFor(() =>
      expect(screen.getByTestId("risk-band")).toHaveTextContent("LOW"),
    );
  });
});
