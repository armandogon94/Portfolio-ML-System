/** A.2.8 — industries registry tests. */
import { describe, it, expect } from "vitest";

import { INDUSTRIES } from "@/lib/industries";

describe("INDUSTRIES registry", () => {
  it("exposes all 6 industries", () => {
    expect(INDUSTRIES).toHaveLength(6);
    expect(INDUSTRIES.map((i) => i.slug).sort()).toEqual(
      ["dental", "fintech", "healthcare", "legal", "logistics", "real-estate"].sort(),
    );
  });

  it("each industry has the fields the landing page needs", () => {
    for (const industry of INDUSTRIES) {
      expect(industry.slug).toMatch(/^[a-z-]+$/);
      expect(industry.title).toBeTruthy();
      expect(industry.tagline).toBeTruthy();
      expect(industry.href).toBe(`/${industry.slug}`);
      expect(typeof industry.icon).toBe("object"); // lucide icon component (ForwardRef)
      expect(industry.modelCount).toBeGreaterThan(0);
    }
  });
});
