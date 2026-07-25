/**
 * The industry registry is the catalogue a reviewer scans first.
 *
 * These tests exist because the previous registry listed 21 models, 10 of which
 * were permanent placeholders with literal `TODO(copy)` taglines. Rows that never
 * resolve read as an abandoned project.
 */
import { describe, expect, it } from "vitest";

import { ALL_MODELS, INDUSTRIES, getIndustry } from "@/lib/industries";

describe("industry registry", () => {
  it("has exactly one industry", () => {
    expect(INDUSTRIES).toHaveLength(1);
    expect(INDUSTRIES[0].slug).toBe("fintech");
  });

  it("has exactly three models, all under fintech", () => {
    expect(ALL_MODELS.map((m) => m.slug).sort()).toEqual([
      "churn",
      "credit-risk",
      "fraud",
    ]);
    expect(ALL_MODELS.every((m) => m.industrySlug === "fintech")).toBe(true);
  });

  it("marks every catalogued model as ready", () => {
    // An unbuilt model does not get a catalogue row.
    expect(ALL_MODELS.every((m) => m.ready)).toBe(true);
  });

  it("contains no placeholder copy anywhere", () => {
    const serialised = JSON.stringify(INDUSTRIES);
    for (const marker of ["TODO", "PLACEHOLDER", "Coming soon", "Lorem"]) {
      expect(serialised).not.toContain(marker);
    }
  });

  it("names the real dataset in every model description", () => {
    // The description is the only place a scanning reviewer learns that these
    // are real datasets rather than generated ones.
    const datasets = ["IEEE-CIS", "LendingClub", "10,127"];
    const joined = ALL_MODELS.map((m) => m.description).join(" ");
    for (const dataset of datasets) {
      expect(joined).toContain(dataset);
    }
  });

  it("derives modelCount from the model list rather than hardcoding it", () => {
    for (const industry of INDUSTRIES) {
      expect(industry.modelCount).toBe(industry.models.length);
    }
  });

  it("builds hrefs that match the app/ route structure", () => {
    expect(ALL_MODELS.map((m) => m.href).sort()).toEqual([
      "/fintech/churn",
      "/fintech/credit-risk",
      "/fintech/fraud",
    ]);
  });

  it("resolves a known slug and returns undefined for an unknown one", () => {
    expect(getIndustry("fintech")?.title).toBe("Fintech");
    expect(getIndustry("healthcare")).toBeUndefined();
  });
});
