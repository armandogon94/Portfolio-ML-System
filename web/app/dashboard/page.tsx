/**
 * /dashboard — async Server Component that fetches the joined data
 * and hands it to the synchronous DashboardContent for rendering.
 *
 * ISR via `export const revalidate = 30` — both the FastAPI /models
 * call and the MLflow REST calls cache for 30 seconds. This keeps
 * the dashboard responsive (no per-request fan-out to MLflow on
 * every page load) while still surfacing fresh metrics within a
 * minute of a new training run completing.
 *
 * Pure server-side fetching — no client-side data loading lib,
 * no TanStack Query at this layer. Read-only surface, low
 * cardinality, perfect fit for Server Components + ISR.
 */

import { DashboardContent } from "@/app/dashboard/DashboardContent";
import { getDashboardRows } from "@/lib/dashboard";

// 30-second ISR window. See lib/dashboard.ts:fetchModelInfo for the
// matching `next: { revalidate: 30 }` on the inner fetch.
export const revalidate = 30;

export default async function DashboardPage() {
  const rows = await getDashboardRows();
  return <DashboardContent rows={rows} />;
}
