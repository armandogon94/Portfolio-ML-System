/** A.2.9: Nav component tests. */
import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";

// next-themes is stubbed globally by the ThemeToggle test; repeat
// here so Nav (which includes <ThemeToggle/>) renders in isolation.
vi.mock("next-themes", () => ({
  useTheme: () => ({ theme: "light", resolvedTheme: "light", setTheme: vi.fn() }),
}));

import { Nav } from "@/components/Nav";

describe("Nav", () => {
  it("renders the project title linked to the landing page", () => {
    render(<Nav />);
    const titleLink = screen.getByRole("link", { name: /portfolio ml system/i });
    expect(titleLink).toHaveAttribute("href", "/");
  });

  it("includes the theme toggle", () => {
    render(<Nav />);
    expect(screen.getByRole("button", { name: /theme/i })).toBeInTheDocument();
  });
});
