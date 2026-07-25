/**
 * The shared result card. Provenance is not decoration: a score without the
 * commit that produced it is unreviewable, and that is why this repository
 * needed rebuilding.
 */
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { PredictionResult, formatPercent } from "@/components/PredictionResult";

const BASE = {
  decisionLabel: "Decision",
  decision: "APPROVE",
  decisionTone: "good" as const,
  probabilityLabel: "Default probability",
  probability: 0.0834,
  modelVersion: "a1b2c3d4",
  trainedOn: "wordsforthewise/lending-club",
};

describe("PredictionResult", () => {
  it("renders the decision and the probability", () => {
    render(<PredictionResult {...BASE} />);
    expect(screen.getByTestId("decision")).toHaveTextContent("APPROVE");
    expect(screen.getByTestId("probability")).toHaveTextContent("8.34%");
  });

  it("always shows which commit trained the model and on what data", () => {
    render(<PredictionResult {...BASE} />);
    expect(screen.getByTestId("model-version")).toHaveTextContent("a1b2c3d4");
    expect(screen.getByTestId("trained-on")).toHaveTextContent("wordsforthewise/lending-club");
  });

  it("renders a caveat when the API sends one", () => {
    render(<PredictionResult {...BASE} caveat="n = 10,127 and the dataset is easy." />);
    expect(screen.getByTestId("caveat")).toHaveTextContent("10,127");
  });

  it("omits the caveat block entirely when there is none", () => {
    render(<PredictionResult {...BASE} />);
    expect(screen.queryByTestId("caveat")).toBeNull();
  });

  it("colours the decision by tone", () => {
    const { rerender } = render(<PredictionResult {...BASE} decisionTone="bad" />);
    expect(screen.getByTestId("decision").className).toContain("red");
    rerender(<PredictionResult {...BASE} decisionTone="warn" />);
    expect(screen.getByTestId("decision").className).toContain("amber");
  });

  it("keeps two decimals so the column width does not jitter", () => {
    expect(formatPercent(0.5)).toBe("50.00%");
    expect(formatPercent(0.123456)).toBe("12.35%");
    expect(formatPercent(0)).toBe("0.00%");
  });
});
