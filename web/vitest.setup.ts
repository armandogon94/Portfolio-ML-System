/**
 * Vitest setup — runs once before every test file.
 * Brings in @testing-library/jest-dom's custom matchers
 * (toBeInTheDocument, toHaveClass, etc.) and stubs Next.js
 * hooks that jsdom can't satisfy (e.g., next/navigation).
 */
import "@testing-library/jest-dom/vitest";
import { vi, afterEach } from "vitest";
import { cleanup } from "@testing-library/react";

// Stub next/navigation — shadcn components + any page that uses
// usePathname / useRouter crashes in jsdom without this. Each test
// can override via vi.mocked() if needed.
vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    refresh: vi.fn(),
    prefetch: vi.fn(),
  }),
  usePathname: () => "/",
  useSearchParams: () => new URLSearchParams(),
}));

// Clean up the DOM between tests to prevent cross-test leakage.
afterEach(() => cleanup());
