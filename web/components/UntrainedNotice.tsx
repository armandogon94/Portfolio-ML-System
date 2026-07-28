/**
 * UntrainedNotice: what a reviewer sees when no checkpoint exists.
 *
 * A fresh clone has no trained models, and the API returns 503 with the exact
 * commands that fix it. Rendering that message verbatim is the honest behaviour:
 * the alternative is a generic "something went wrong", which suggests a bug in a
 * system that is working exactly as documented.
 *
 * This is also the state the demo is in until the datasets are downloaded; see
 * docs/PROGRESS.md.
 */

import { AlertCircle } from "lucide-react";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export function UntrainedNotice({ detail }: { detail: string }) {
  return (
    <Card data-testid="untrained-notice" className="border-amber-300 dark:border-amber-700">
      <CardHeader>
        <div className="flex items-center gap-2">
          <AlertCircle aria-hidden="true" className="h-4 w-4 text-amber-600" />
          <CardTitle className="text-base">No trained model yet</CardTitle>
        </div>
        <CardDescription>
          The API is healthy. This model simply has not been trained on real data
          yet, so there is nothing to score with.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <pre className="overflow-x-auto whitespace-pre-wrap rounded-md bg-muted p-3 text-xs">
          {detail}
        </pre>
      </CardContent>
    </Card>
  );
}
