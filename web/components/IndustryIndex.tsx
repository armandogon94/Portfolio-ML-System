/**
 * IndustryIndex: shared layout for every /<industry> stub page.
 *
 * The six industry-level pages (`app/<slug>/page.tsx`) delegate to
 * this component so we have a single source of truth for how each
 * industry's model list renders. As A.3–A.8 ship model pages and
 * flip `ready: true` in lib/industries.ts, this component picks up
 * the change automatically, with no edits here required.
 */
import Link from "next/link";
import { notFound } from "next/navigation";
import { ArrowLeft } from "lucide-react";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { getIndustry } from "@/lib/industries";

export function IndustryIndex({ slug }: { slug: string }) {
  const industry = getIndustry(slug);
  if (!industry) return notFound();

  const { title, tagline, icon: Icon, models } = industry;

  return (
    <main className="container mx-auto max-w-4xl px-4 py-10">
      <Link
        href="/"
        className="mb-4 inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
      >
        <ArrowLeft className="h-4 w-4" />
        All industries
      </Link>

      <header className="mb-8 flex items-start gap-4">
        <span
          aria-hidden="true"
          className="inline-flex h-12 w-12 items-center justify-center rounded-md bg-primary/10 text-primary"
        >
          <Icon className="h-6 w-6" />
        </span>
        <div>
          <h1 className="text-3xl font-semibold tracking-tight">{title}</h1>
          <p className="text-muted-foreground">{tagline}</p>
        </div>
      </header>

      <section aria-label="Models" className="space-y-3">
        {models.map((model) => {
          const href = `/${slug}/${model.slug}`;
          const inner = (
            <Card
              className={
                model.ready
                  ? "transition-colors hover:border-primary/60"
                  : "opacity-60"
              }
            >
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle>{model.title}</CardTitle>
                  <span className="text-xs font-medium text-muted-foreground">
                    {model.ready ? "Try it →" : "Coming soon"}
                  </span>
                </div>
                <CardDescription>{model.description}</CardDescription>
              </CardHeader>
              <CardContent className="text-xs text-muted-foreground">
                /{slug}/{model.slug}
              </CardContent>
            </Card>
          );
          return model.ready ? (
            <Link key={model.slug} href={href} className="block focus:outline-none">
              {inner}
            </Link>
          ) : (
            <div key={model.slug} aria-disabled="true">
              {inner}
            </div>
          );
        })}
      </section>
    </main>
  );
}
