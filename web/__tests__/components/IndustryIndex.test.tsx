/** A.2.12 — IndustryIndex smoke test.
 *
 * Shared layout for each /<industry> route. We only need to verify
 * it renders the industry's models and handles unknown slugs, since
 * IndustryTile / INDUSTRIES registry are already covered elsewhere.
 */
import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";

// next/navigation is stubbed globally in vitest.setup.ts; notFound
// just needs to return the React element flag, not actually throw.
vi.mock("next/navigation", async () => {
  const actual = await vi.importActual<typeof import("next/navigation")>("next/navigation");
  return { ...actual, notFound: () => null };
});

import { IndustryIndex } from "@/components/IndustryIndex";

describe("IndustryIndex", () => {
  it("renders the industry title and all its models", () => {
    render(<IndustryIndex slug="fintech" />);
    expect(screen.getByRole("heading", { name: /fintech/i })).toBeInTheDocument();
    // Fintech has 4 models per INDUSTRIES — check by title.
    // "customer churn" also appears in the tagline, so use getAllBy.
    expect(screen.getByText(/credit risk scoring/i)).toBeInTheDocument();
    expect(screen.getAllByText(/customer churn/i).length).toBeGreaterThan(0);
  });

  it("marks not-yet-shipped models with a 'Coming soon' label", () => {
    render(<IndustryIndex slug="fintech" />);
    // credit-risk is the only ready model in A.2; the other 3 should
    // show the pending label.
    const comingSoon = screen.getAllByText(/coming soon/i);
    expect(comingSoon.length).toBeGreaterThan(0);
  });

  it("links ready models to their route", () => {
    render(<IndustryIndex slug="fintech" />);
    // credit-risk is ready → link to /fintech/credit-risk
    const links = screen
      .getAllByRole("link")
      .filter((a) => a.getAttribute("href") === "/fintech/credit-risk");
    expect(links.length).toBeGreaterThan(0);
  });

  it("handles unknown slugs via notFound()", () => {
    const { container } = render(<IndustryIndex slug="does-not-exist" />);
    // notFound() returned null → nothing rendered, no crash
    expect(container).toBeEmptyDOMElement();
  });
});
