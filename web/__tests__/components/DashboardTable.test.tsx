/** A.9.8 — DashboardTable component tests.
 *
 * Pure data-in / DOM-out: accepts a DashboardRow[] prop and renders
 * a sortable, filterable table. The data join + fetch lives in
 * lib/dashboard.ts (A.9.9). Here we only verify rendering, sort,
 * filter, and empty-state behavior with a hand-crafted fixture.
 */
import { describe, it, expect } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { DashboardTable } from "@/components/DashboardTable";
import type { DashboardRow } from "@/lib/dashboard";

const FIXTURE: DashboardRow[] = [
  {
    industrySlug: "fintech",
    industryTitle: "Fintech",
    modelSlug: "credit-risk",
    modelTitle: "Credit Risk Scoring",
    modelDescription: "XGBoost loan-default classifier.",
    status: "ready",
    href: "/fintech/credit-risk",
    keyMetric: { name: "test_auc_roc", label: "AUC-ROC", value: 0.85, higherIsBetter: true },
    lastTrained: "2026-04-12T13:40:00",
    history: [0.81, 0.83, 0.84, 0.85],
  },
  {
    industrySlug: "fintech",
    industryTitle: "Fintech",
    modelSlug: "fraud",
    modelTitle: "Fraud Detection",
    modelDescription: "Autoencoder + isolation forest.",
    status: "ready",
    href: "/fintech/fraud",
    keyMetric: {
      name: "test_autoencoder_auc_roc",
      label: "AUC-ROC",
      value: 0.91,
      higherIsBetter: true,
    },
    lastTrained: "2026-04-23T05:00:00",
    history: [0.88, 0.9, 0.91],
  },
  {
    industrySlug: "real-estate",
    industryTitle: "Real Estate",
    modelSlug: "price",
    modelTitle: "Price Prediction",
    modelDescription: "LightGBM home-price regressor.",
    status: "ready",
    href: "/real-estate/price",
    keyMetric: { name: "test_r2", label: "R²", value: 0.78, higherIsBetter: true },
    lastTrained: "2026-04-12T13:50:00",
    history: [0.72, 0.75, 0.78],
  },
  {
    industrySlug: "logistics",
    industryTitle: "Logistics",
    modelSlug: "damage-risk",
    modelTitle: "Shipment Damage Risk",
    modelDescription: "XGBoost risk classifier.",
    status: "not_built",
    href: null,
    keyMetric: null,
    lastTrained: null,
    history: [],
  },
];

describe("DashboardTable", () => {
  it("renders one row per DashboardRow plus a header row", () => {
    render(<DashboardTable rows={FIXTURE} />);
    const table = screen.getByRole("table");
    const rows = within(table).getAllByRole("row");
    // 1 header + 4 data rows
    expect(rows.length).toBe(5);
    expect(within(table).getByText("Credit Risk Scoring")).toBeInTheDocument();
    expect(within(table).getByText("Fraud Detection")).toBeInTheDocument();
    expect(within(table).getByText("Price Prediction")).toBeInTheDocument();
    expect(within(table).getByText("Shipment Damage Risk")).toBeInTheDocument();
  });

  it("links the model name when status='ready' and renders it as plain text otherwise", () => {
    render(<DashboardTable rows={FIXTURE} />);
    // Ready row has a link to its page
    const link = screen.getByRole("link", { name: /credit risk scoring/i });
    expect(link).toHaveAttribute("href", "/fintech/credit-risk");
    // Not-built row has no link
    expect(screen.queryByRole("link", { name: /shipment damage risk/i })).not.toBeInTheDocument();
  });

  it("filters rows when an industry is selected from the filter dropdown", async () => {
    const user = userEvent.setup();
    render(<DashboardTable rows={FIXTURE} />);
    const filter = screen.getByLabelText(/industry filter/i) as HTMLSelectElement;
    await user.selectOptions(filter, "real-estate");
    // After filter: only Price Prediction visible
    expect(screen.getByText("Price Prediction")).toBeInTheDocument();
    expect(screen.queryByText("Credit Risk Scoring")).not.toBeInTheDocument();
    expect(screen.queryByText("Fraud Detection")).not.toBeInTheDocument();
    expect(screen.queryByText("Shipment Damage Risk")).not.toBeInTheDocument();
  });

  it("sorts ready rows by metric value descending when the metric header is clicked", async () => {
    const user = userEvent.setup();
    render(<DashboardTable rows={FIXTURE} />);
    const metricHeader = screen.getByRole("button", { name: /sort by metric/i });
    await user.click(metricHeader);
    const dataRows = screen.getAllByRole("row").slice(1);
    // First three rows are the "ready" ones; assert their order by metric desc:
    // Fraud (0.91) > Credit Risk (0.85) > Price (0.78). Not-built rows last.
    const orderedTitles = dataRows.map((r) => within(r).getByTestId("model-name").textContent);
    expect(orderedTitles[0]).toBe("Fraud Detection");
    expect(orderedTitles[1]).toBe("Credit Risk Scoring");
    expect(orderedTitles[2]).toBe("Price Prediction");
  });

  it("shows an empty-state row when given an empty rows array", () => {
    render(<DashboardTable rows={[]} />);
    expect(screen.getByText(/no models to display/i)).toBeInTheDocument();
  });
});
