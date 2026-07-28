/** A.2.8: IndustryTile component tests. */
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { DollarSign } from "lucide-react";

import { IndustryTile } from "@/components/IndustryTile";

describe("IndustryTile", () => {
  const baseProps = {
    href: "/fintech",
    title: "Fintech",
    tagline: "Credit risk, fraud, loans, churn",
    icon: DollarSign,
    modelCount: 4,
  };

  it("renders title, tagline, and model count", () => {
    render(<IndustryTile {...baseProps} />);
    expect(screen.getByText("Fintech")).toBeInTheDocument();
    expect(screen.getByText(/credit risk/i)).toBeInTheDocument();
    expect(screen.getByText(/4 models/i)).toBeInTheDocument();
  });

  it("links the CTA to the industry route", () => {
    render(<IndustryTile {...baseProps} />);
    const link = screen.getByRole("link", { name: /try models|fintech/i });
    expect(link).toHaveAttribute("href", "/fintech");
  });

  it("renders the provided lucide icon as an svg", () => {
    const { container } = render(<IndustryTile {...baseProps} />);
    expect(container.querySelector("svg")).toBeInTheDocument();
  });
});
