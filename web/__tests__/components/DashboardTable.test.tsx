/**
 * DashboardTable — pure data-in / DOM-out. The join and the fetching live in
 * lib/dashboard.ts; this only covers rendering, sorting, filtering and the
 * empty state.
 *
 * The fixture is the repository's three real models. The `not_built` row is not
 * hypothetical either: with no Kaggle credentials on this machine, that is the
 * state all three are currently in.
 */
import { describe, expect, it } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { DashboardTable } from "@/components/DashboardTable";
import type { DashboardRow } from "@/lib/dashboard";

function makeRow(
  modelSlug: string,
  modelTitle: string,
  status: DashboardRow["status"],
  metric: number | null,
  history: number[] = [],
): DashboardRow {
  return {
    industrySlug: "fintech",
    industryTitle: "Fintech",
    modelSlug,
    modelTitle,
    modelDescription: "",
    status,
    href: status === "ready" ? `/fintech/${modelSlug}` : null,
    keyMetric:
      metric === null
        ? null
        : { name: "test_pr_auc", label: "PR-AUC", value: metric, higherIsBetter: true },
    lastTrained: status === "ready" ? "2026-07-24T13:40:00+00:00" : null,
    history,
  };
}

const FIXTURE: DashboardRow[] = [
  makeRow("credit-risk", "Consumer Credit Risk", "ready", 0.31, [0.28, 0.3, 0.31]),
  makeRow("fraud", "Payment Fraud", "ready", 0.47, [0.41, 0.45, 0.47]),
  makeRow("churn", "Card Attrition", "not_built", null),
];

describe("DashboardTable", () => {
  it("renders one row per model plus a header row", () => {
    render(<DashboardTable rows={FIXTURE} />);
    const table = screen.getByRole("table");
    expect(within(table).getAllByRole("row")).toHaveLength(4);
    expect(within(table).getByText("Consumer Credit Risk")).toBeInTheDocument();
    expect(within(table).getByText("Payment Fraud")).toBeInTheDocument();
    expect(within(table).getByText("Card Attrition")).toBeInTheDocument();
  });

  it("links ready models and leaves untrained ones as plain text", () => {
    render(<DashboardTable rows={FIXTURE} />);
    expect(screen.getByRole("link", { name: /consumer credit risk/i })).toHaveAttribute(
      "href",
      "/fintech/credit-risk",
    );
    // Nothing to link to: there is no checkpoint behind an untrained model.
    expect(screen.queryByRole("link", { name: /card attrition/i })).not.toBeInTheDocument();
  });

  it("offers exactly the industries in the registry as filter options", () => {
    // The options come from lib/industries.ts, not from the rows, so a stale
    // vertical cannot reappear in the dropdown after being deleted.
    render(<DashboardTable rows={FIXTURE} />);
    const filter = screen.getByLabelText(/industry filter/i) as HTMLSelectElement;
    expect([...filter.options].map((o) => o.value)).toEqual(["all", "fintech"]);
  });

  it("filters rows to the selected industry", async () => {
    const user = userEvent.setup();
    const withOther: DashboardRow[] = [
      ...FIXTURE,
      { ...makeRow("legacy", "Legacy Model", "ready", 0.5), industrySlug: "retired", industryTitle: "Retired" },
    ];
    render(<DashboardTable rows={withOther} />);

    expect(screen.getByText("Legacy Model")).toBeInTheDocument();
    await user.selectOptions(screen.getByLabelText(/industry filter/i), "fintech");

    expect(screen.getByText("Payment Fraud")).toBeInTheDocument();
    expect(screen.queryByText("Legacy Model")).not.toBeInTheDocument();
  });

  it("sorts ready rows by metric descending, untrained rows last", async () => {
    const user = userEvent.setup();
    render(<DashboardTable rows={FIXTURE} />);
    await user.click(screen.getByRole("button", { name: /sort by metric/i }));

    const titles = screen
      .getAllByRole("row")
      .slice(1)
      .map((r) => within(r).getByTestId("model-name").textContent);
    expect(titles[0]).toBe("Payment Fraud"); // 0.47
    expect(titles[1]).toBe("Consumer Credit Risk"); // 0.31
    expect(titles[2]).toBe("Card Attrition"); // no metric
  });

  it("shows an empty state rather than a bare table when there are no rows", () => {
    render(<DashboardTable rows={[]} />);
    expect(screen.getByText(/no models to display/i)).toBeInTheDocument();
  });
});
