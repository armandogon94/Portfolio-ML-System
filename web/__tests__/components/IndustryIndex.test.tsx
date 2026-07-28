/**
 * IndustryIndex: the shared layout for `/<industry>`.
 *
 * Every catalogued model is now `ready: true`, so this component should never
 * render a "Coming soon" state. That is asserted explicitly: a placeholder row
 * is what the old six-industry catalogue was full of.
 */
import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";

// notFound() only needs to return the React null flag here, not actually throw.
vi.mock("next/navigation", async () => {
  const actual = await vi.importActual<typeof import("next/navigation")>("next/navigation");
  return { ...actual, notFound: () => null };
});

import { IndustryIndex } from "@/components/IndustryIndex";

describe("IndustryIndex", () => {
  it("renders the industry heading", () => {
    render(<IndustryIndex slug="fintech" />);
    expect(screen.getByRole("heading", { name: /fintech/i })).toBeInTheDocument();
  });

  it("lists all three models by title", () => {
    render(<IndustryIndex slug="fintech" />);
    // getAllByText: the industry tagline names the same three topics, so an
    // exact-count query would be brittle for the wrong reason.
    expect(screen.getAllByText(/payment fraud/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/consumer credit risk/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/card attrition/i).length).toBeGreaterThan(0);
  });

  it("renders no 'Coming soon' placeholder anywhere", () => {
    render(<IndustryIndex slug="fintech" />);
    expect(screen.queryByText(/coming soon/i)).toBeNull();
  });

  it("links every model to its route", () => {
    render(<IndustryIndex slug="fintech" />);
    const hrefs = screen.getAllByRole("link").map((a) => a.getAttribute("href"));
    for (const href of ["/fintech/fraud", "/fintech/credit-risk", "/fintech/churn"]) {
      expect(hrefs).toContain(href);
    }
  });

  it("names the underlying dataset in each description", () => {
    render(<IndustryIndex slug="fintech" />);
    expect(screen.getByText(/IEEE-CIS/)).toBeInTheDocument();
    expect(screen.getByText(/LendingClub/)).toBeInTheDocument();
  });

  it("handles an unknown slug via notFound()", () => {
    const { container } = render(<IndustryIndex slug="does-not-exist" />);
    expect(container).toBeEmptyDOMElement();
  });
});
