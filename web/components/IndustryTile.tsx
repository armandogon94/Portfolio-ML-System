/**
 * IndustryTile: landing-page card linking to one industry's model index.
 *
 * Kept intentionally minimal: shadcn Card + lucide icon + title + tagline
 * + a "Try models →" CTA. The whole card becomes a single clickable
 * target by wrapping it in a <Link/>; the CTA arrow is decorative.
 */
import Link from "next/link";
import { ArrowRight, type LucideIcon } from "lucide-react";

import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";

type IndustryTileProps = {
  href: string;
  title: string;
  tagline: string;
  icon: LucideIcon;
  modelCount: number;
};

export function IndustryTile({
  href,
  title,
  tagline,
  icon: Icon,
  modelCount,
}: IndustryTileProps) {
  return (
    <Link href={href} className="group block focus:outline-none">
      <Card className="h-full transition-colors hover:border-primary/60 focus-within:border-primary">
        <CardHeader>
          <div className="flex items-center gap-3">
            <span
              aria-hidden="true"
              className="inline-flex h-9 w-9 items-center justify-center rounded-md bg-primary/10 text-primary"
            >
              <Icon className="h-5 w-5" />
            </span>
            <CardTitle>{title}</CardTitle>
          </div>
          <CardDescription>{tagline}</CardDescription>
        </CardHeader>
        <CardContent className="flex items-center justify-between pt-2">
          <span className="text-sm text-muted-foreground">
            {modelCount} model{modelCount === 1 ? "" : "s"}
          </span>
          <span className="inline-flex items-center gap-1 text-sm font-medium text-primary">
            Try models
            <ArrowRight
              aria-hidden="true"
              className="h-4 w-4 transition-transform group-hover:translate-x-0.5"
            />
          </span>
        </CardContent>
      </Card>
    </Link>
  );
}
