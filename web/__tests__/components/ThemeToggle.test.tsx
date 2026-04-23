/** A.2.9 — ThemeToggle component tests.
 *
 * next-themes persistence to localStorage is exercised by the library's
 * own tests; here we only verify our button calls `setTheme` with the
 * opposite value and is keyboard-accessible.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

// Mock next-themes so we can assert setTheme calls + control the
// resolvedTheme each test sees.
let mockTheme: string | undefined = "light";
const setTheme = vi.fn((t: string) => {
  mockTheme = t;
});

vi.mock("next-themes", () => ({
  useTheme: () => ({
    theme: mockTheme,
    resolvedTheme: mockTheme,
    setTheme,
  }),
}));

import { ThemeToggle } from "@/components/ThemeToggle";

describe("ThemeToggle", () => {
  beforeEach(() => {
    mockTheme = "light";
    setTheme.mockClear();
  });

  it("renders a button with an accessible name", () => {
    render(<ThemeToggle />);
    expect(screen.getByRole("button", { name: /theme/i })).toBeInTheDocument();
  });

  it("flips from light to dark on click", async () => {
    const user = userEvent.setup();
    render(<ThemeToggle />);
    await user.click(screen.getByRole("button", { name: /theme/i }));
    expect(setTheme).toHaveBeenCalledWith("dark");
  });

  it("flips from dark to light on click", async () => {
    const user = userEvent.setup();
    mockTheme = "dark";
    render(<ThemeToggle />);
    await user.click(screen.getByRole("button", { name: /theme/i }));
    expect(setTheme).toHaveBeenCalledWith("light");
  });
});
