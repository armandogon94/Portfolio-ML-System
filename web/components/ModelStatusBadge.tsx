/**
 * ModelStatusBadge: colored pill summarizing a model's lifecycle state.
 *
 * 3-value enum (ready / training / not_built) → green / amber / muted
 * palette. Used by the dashboard table's "Status" column. Inlined
 * rather than abstracted as a generic shadcn Badge primitive because
 * it currently has exactly one caller and the styling is tightly
 * coupled to the status semantics.
 */

import { cn } from "@/lib/utils";

export type ModelStatus = "ready" | "training" | "not_built";

const STATUS_CLASSES: Record<ModelStatus, string> = {
  ready:
    "bg-green-100 text-green-800 border-green-300 " +
    "dark:bg-green-900/30 dark:text-green-300 dark:border-green-700",
  training:
    "bg-amber-100 text-amber-800 border-amber-300 " +
    "dark:bg-amber-900/30 dark:text-amber-300 dark:border-amber-700",
  not_built:
    "bg-muted text-muted-foreground border-muted-foreground/20",
};

const STATUS_LABELS: Record<ModelStatus, string> = {
  ready: "Ready",
  training: "Training",
  not_built: "Not built",
};

export function ModelStatusBadge({ status }: { status: ModelStatus }) {
  return (
    <span
      data-testid="model-status-badge"
      className={cn(
        "inline-flex items-center rounded-md border px-2 py-0.5 text-xs font-medium",
        STATUS_CLASSES[status],
      )}
    >
      {STATUS_LABELS[status]}
    </span>
  );
}
