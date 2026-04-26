/** A.9.9 — /dashboard page smoke test.
 *
 * The page itself is an async Server Component that calls
 * getDashboardRows() and passes the result down to a synchronous
 * <DashboardContent>. We test the synchronous content directly so
 * vitest doesn't need to handle async RSC rendering. The
 * getDashboardRows logic is covered separately in lib/dashboard.test.ts.
 */
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";

import { DashboardContent } from "@/app/dashboard/DashboardContent";
import { INDUSTRIES } from "@/lib/industries";
import type { DashboardRow } from "@/lib/dashboard";

const FIXTURE: DashboardRow[] = [
  ...INDUSTRIES.flatMap((ind) =>
    ind.models.map((m, idx) => ({
      industrySlug: ind.slug,
      industryTitle: ind.title,
      modelSlug: m.slug,
      modelTitle: m.title,
      modelDescription: m.description,
      status: m.ready ? ("ready" as const) : ("not_built" as const),
      href: m.ready ? `/${ind.slug}/${m.slug}` : null,
      keyMetric: m.ready
        ? {
            name: "test_auc_roc",
            label: "AUC-ROC",
            value: 0.8 + idx * 0.01,
            higherIsBetter: true,
          }
        : null,
      lastTrained: m.ready ? "2026-04-23T00:00:00" : null,
      history: [],
    })),
  ),
];

describe("DashboardContent", () => {
  it("renders one IndustrySummaryTile per industry plus the table", () => {
    render(<DashboardContent rows={FIXTURE} />);
    // Each industry shows up once as a tile heading.
    for (const ind of INDUSTRIES) {
      expect(screen.getAllByText(ind.title).length).toBeGreaterThanOrEqual(1);
    }
    // Table is present as exactly one role="table".
    expect(screen.getByRole("table")).toBeInTheDocument();
  });

  it("renders the page heading and a model count line in the description", () => {
    render(<DashboardContent rows={FIXTURE} />);
    expect(screen.getByRole("heading", { name: /model dashboard/i })).toBeInTheDocument();
    // Total row count is mentioned in the page subtitle.
    expect(screen.getByText(new RegExp(`${FIXTURE.length} models`, "i"))).toBeInTheDocument();
  });
});
