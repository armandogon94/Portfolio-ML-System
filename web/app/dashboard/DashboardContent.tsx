/**
 * DashboardContent — synchronous, data-in renderer for the
 * /dashboard page. Split out from page.tsx so the async
 * Server Component layer is a thin wrapper around the fetch
 * + this component handles purely deterministic rendering.
 *
 * Vitest-friendly: tests can render `<DashboardContent rows={...} />`
 * directly without grappling with async RSC mechanics. The page
 * itself only needs a smoke check that it compiles.
 *
 * Pure SSR-safe: no `"use client"` directive, no hooks at this
 * level. Children (`DashboardTable`) are themselves Client
 * Components and will be hydrated automatically.
 */

import { DashboardTable } from "@/components/DashboardTable";
import { IndustrySummaryTile } from "@/components/IndustrySummaryTile";
import { INDUSTRIES } from "@/lib/industries";
import type { DashboardRow } from "@/lib/dashboard";

export function DashboardContent({ rows }: { rows: DashboardRow[] }) {
  return (
    <main className="container mx-auto max-w-7xl px-4 py-8">
      <header className="mb-8 space-y-1">
        <h1 className="text-3xl font-bold tracking-tight">Model Dashboard</h1>
        <p className="text-sm text-muted-foreground">
          Live status across {rows.length} models in {INDUSTRIES.length} industries.
        </p>
      </header>

      {/* Industry summary tiles — one per industry, joined client-side. */}
      <section
        aria-labelledby="industries-heading"
        className="mb-10 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3"
      >
        <h2 id="industries-heading" className="sr-only">
          Industry summaries
        </h2>
        {INDUSTRIES.map((ind) => (
          <IndustrySummaryTile
            key={ind.slug}
            industry={ind}
            rows={rows.filter((r) => r.industrySlug === ind.slug)}
          />
        ))}
      </section>

      <section aria-labelledby="catalog-heading" className="space-y-3">
        <h2 id="catalog-heading" className="text-xl font-semibold tracking-tight">
          Model catalog
        </h2>
        <p className="text-sm text-muted-foreground">
          Sortable table of every model in the registry. Click a header to sort;
          use the dropdown to filter to one industry.
        </p>
        <DashboardTable rows={rows} />
      </section>
    </main>
  );
}
