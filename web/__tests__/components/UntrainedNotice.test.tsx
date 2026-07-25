/**
 * A fresh clone has no trained checkpoints. That state must read as "not trained
 * yet", never as "something went wrong" — the API is behaving exactly as
 * documented and its 503 detail carries the commands that fix it.
 */
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { UntrainedNotice } from "@/components/UntrainedNotice";

const DETAIL = [
  "No checkpoint for 'fraud' at /app/checkpoints/fraud.",
  "Create it with:",
  "  uv run python scripts/download_data.py --dataset all",
  "  uv run python scripts/train.py --model fraud",
].join("\n");

describe("UntrainedNotice", () => {
  it("says the model is untrained, not that the API is broken", () => {
    render(<UntrainedNotice detail={DETAIL} />);
    expect(screen.getByText(/No trained model yet/i)).toBeInTheDocument();
    expect(screen.getByText(/The API is healthy/i)).toBeInTheDocument();
  });

  it("renders the API's remediation commands verbatim", () => {
    render(<UntrainedNotice detail={DETAIL} />);
    const block = screen.getByTestId("untrained-notice");
    expect(block).toHaveTextContent("scripts/download_data.py --dataset all");
    expect(block).toHaveTextContent("scripts/train.py --model fraud");
  });
});
