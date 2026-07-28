/** A.2.3 smoke tests: Vitest + jsdom + providers scaffold. */
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";

import { queryClient } from "@/lib/query-client";
import { queryClient as queryClientAgain } from "@/lib/query-client";

describe("Vitest + testing-library scaffold", () => {
  it("renders JSX into jsdom", () => {
    render(<div>hello portfolio</div>);
    expect(screen.getByText("hello portfolio")).toBeInTheDocument();
  });
});

describe("lib/query-client", () => {
  it("exports a singleton QueryClient (same identity across imports)", () => {
    expect(queryClient).toBe(queryClientAgain);
  });

  it("singleton has sane defaults", () => {
    const opts = queryClient.getDefaultOptions();
    // retry is either a number or a function; verify it's defined
    expect(opts.queries?.retry).toBeDefined();
  });
});
