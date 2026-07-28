/** A.9.7: ModelStatusBadge component tests.
 *
 * Maps a 3-value enum (ready / training / not_built) to a colored
 * label badge for the dashboard table. Display labels are
 * humanized: ready→"Ready", training→"Training", not_built→"Not built".
 */
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";

import { ModelStatusBadge } from "@/components/ModelStatusBadge";

describe("ModelStatusBadge", () => {
  it("renders 'Ready' label with green-tinted classes for status='ready'", () => {
    render(<ModelStatusBadge status="ready" />);
    const badge = screen.getByTestId("model-status-badge");
    expect(badge).toHaveTextContent(/ready/i);
    expect(badge.className).toMatch(/green/);
  });

  it("renders 'Training' label with amber-tinted classes for status='training'", () => {
    render(<ModelStatusBadge status="training" />);
    const badge = screen.getByTestId("model-status-badge");
    expect(badge).toHaveTextContent(/training/i);
    expect(badge.className).toMatch(/amber|yellow/);
  });

  it("renders 'Not built' label with muted-tinted classes for status='not_built'", () => {
    render(<ModelStatusBadge status="not_built" />);
    const badge = screen.getByTestId("model-status-badge");
    expect(badge).toHaveTextContent(/not built/i);
    expect(badge.className).toMatch(/muted|gray|slate|zinc/);
  });
});
