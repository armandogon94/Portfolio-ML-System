/** A.2.6 — PredictionResult component tests. */
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";

import { PredictionResult } from "@/components/PredictionResult";
import type { CreditRiskPrediction } from "@/lib/api";

const basePrediction = (overrides: Partial<CreditRiskPrediction> = {}): CreditRiskPrediction => ({
  risk_score: 0.25,
  recommendation: "REVIEW",
  confidence: 0.75,
  default_probability: 0.25,
  ...overrides,
});

describe("PredictionResult", () => {
  it("renders the risk score rounded to 2 decimal places (as percentage)", () => {
    render(
      <PredictionResult
        result={basePrediction({ risk_score: 0.12345, default_probability: 0.12345 })}
      />,
    );
    // 0.12345 → 12.35% (two decimals) or 12% depending on rounding rule
    // Contract: two-decimal rendering must be present somewhere.
    expect(screen.getByTestId("risk-score")).toHaveTextContent("12.35%");
  });

  it("renders confidence as a percentage with two decimals", () => {
    render(<PredictionResult result={basePrediction({ confidence: 0.8765 })} />);
    expect(screen.getByTestId("confidence")).toHaveTextContent("87.65%");
  });

  it("shows the recommendation text verbatim", () => {
    render(<PredictionResult result={basePrediction({ recommendation: "APPROVE" })} />);
    expect(screen.getByTestId("recommendation")).toHaveTextContent("APPROVE");
  });

  it("color-codes APPROVE green", () => {
    render(<PredictionResult result={basePrediction({ recommendation: "APPROVE" })} />);
    const badge = screen.getByTestId("recommendation");
    // Tailwind green utility: any class containing 'green' (bg or text).
    // Exact palette may evolve; the contract is semantic color.
    expect(badge.className).toMatch(/green/);
    expect(badge.className).not.toMatch(/red|yellow|amber/);
  });

  it("color-codes REVIEW yellow/amber", () => {
    render(<PredictionResult result={basePrediction({ recommendation: "REVIEW" })} />);
    const badge = screen.getByTestId("recommendation");
    expect(badge.className).toMatch(/yellow|amber/);
    expect(badge.className).not.toMatch(/green|red/);
  });

  it("color-codes DECLINE red", () => {
    render(<PredictionResult result={basePrediction({ recommendation: "DECLINE" })} />);
    const badge = screen.getByTestId("recommendation");
    expect(badge.className).toMatch(/red/);
    expect(badge.className).not.toMatch(/green|yellow|amber/);
  });
});
