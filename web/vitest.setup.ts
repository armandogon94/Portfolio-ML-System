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

// jsdom polyfills for Browser APIs that Radix primitives expect.
// - ResizeObserver: Slider uses @radix-ui/react-use-size which instantiates
//   one on mount. jsdom doesn't implement it, so mounting any Slider throws.
// - PointerEvent.hasPointerCapture: Radix Slider + Switch call this during
//   pointer events; jsdom's Element lacks the implementation.
if (typeof globalThis.ResizeObserver === "undefined") {
  globalThis.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  } as unknown as typeof ResizeObserver;
}
if (typeof Element !== "undefined" && !Element.prototype.hasPointerCapture) {
  Element.prototype.hasPointerCapture = () => false;
  Element.prototype.setPointerCapture = () => {};
  Element.prototype.releasePointerCapture = () => {};
}

// Clean up the DOM between tests to prevent cross-test leakage.
afterEach(() => cleanup());
