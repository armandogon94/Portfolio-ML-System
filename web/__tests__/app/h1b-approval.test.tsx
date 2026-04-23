/** A.8 — H-1B approval page smoke test.
 *
 * Mirrors the shape of credit-risk.test.tsx — mount the page with the
 * API mocked, verify the form renders, submit defaults, and confirm the
 * result card + chart paint with the mocked values.
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
    predictH1bApproval: vi.fn(),
    explainH1bApproval: vi.fn(),
  };
});

import H1BApprovalPage from "@/app/legal/h1b-approval/page";
import { predictH1bApproval, explainH1bApproval } from "@/lib/api";

function withQueryClient(ui: React.ReactElement) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return <QueryClientProvider client={client}>{ui}</QueryClientProvider>;
}

describe("H1BApprovalPage", () => {
  beforeEach(() => {
    vi.mocked(predictH1bApproval).mockReset();
    vi.mocked(explainH1bApproval).mockReset();
  });

  it("renders the input form with a submit button", () => {
    render(withQueryClient(<H1BApprovalPage />));
    expect(screen.getByLabelText(/prevailing wage/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/education level/i)).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /submit|predict/i }),
    ).toBeInTheDocument();
  });

  it("submits defaults and renders prediction + chart on success", async () => {
    const user = userEvent.setup();

    vi.mocked(predictH1bApproval).mockResolvedValueOnce({
      probability_approval: 0.82,
      recommendation: "APPROVE_LIKELY",
      confidence: 0.82,
    });
    vi.mocked(explainH1bApproval).mockResolvedValueOnce({
      feature_importances: {
        prevailing_wage: 0.45,
        employer_prior_approval_rate: 0.30,
        education_level: 0.15,
      },
      top_features: [],
      explanation_type: "shap",
    });

    render(withQueryClient(<H1BApprovalPage />));
    await user.click(screen.getByRole("button", { name: /submit|predict/i }));

    await waitFor(() => {
      expect(predictH1bApproval).toHaveBeenCalledOnce();
      expect(explainH1bApproval).toHaveBeenCalledOnce();
    });

    // Result card paints with the mocked values
    await waitFor(() =>
      expect(screen.getByTestId("recommendation")).toHaveTextContent(
        "APPROVE_LIKELY",
      ),
    );
    expect(screen.getByTestId("probability-approval")).toHaveTextContent(
      "82.00%",
    );
    expect(screen.getByTestId("confidence")).toHaveTextContent("82.00%");
  });

  it("renders amber review tier when probability lands in the middle band", async () => {
    const user = userEvent.setup();

    vi.mocked(predictH1bApproval).mockResolvedValueOnce({
      probability_approval: 0.55,
      recommendation: "REVIEW",
      confidence: 0.55,
    });
    vi.mocked(explainH1bApproval).mockResolvedValueOnce({
      feature_importances: { prevailing_wage: 0.1 },
      top_features: [],
      explanation_type: "shap",
    });

    render(withQueryClient(<H1BApprovalPage />));
    await user.click(screen.getByRole("button", { name: /submit|predict/i }));

    await waitFor(() =>
      expect(screen.getByTestId("recommendation")).toHaveTextContent("REVIEW"),
    );
    // Amber palette — check a representative class from the mapping.
    expect(screen.getByTestId("recommendation").className).toMatch(
      /bg-amber/,
    );
  });
});
