# Architecture Decision Records

Each ADR captures a significant architectural decision: the context that
forced the call, the alternatives considered, the chosen approach, and
the consequences accepted by choosing it. Older ADRs are not deleted
when superseded — the historical reasoning is the point.

| # | Title | Status | Date |
|---|-------|--------|------|
| [ADR-001](ADR-001-gradio-to-nextjs.md) | Replace Gradio with Next.js + shadcn/ui for the demo UI | Accepted | 2026-04-25 |

## Conventions

- Sequential numbering, zero-padded to 3 digits (`ADR-001`, `ADR-042`, …).
- File name: `ADR-<NNN>-<short-kebab-slug>.md`.
- One decision per ADR. Keep them small enough to read in 5 minutes.
- Status lifecycle: **Proposed → Accepted → (Superseded by ADR-XXX | Deprecated)**.
- When a decision changes, write a new ADR that references and supersedes
  the old one. Don't edit the old one in place beyond bumping its Status.
