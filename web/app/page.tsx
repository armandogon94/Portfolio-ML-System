/**
 * Landing page — grid of 6 industry tiles.
 *
 * Server Component (no interactivity needed). Responsive grid:
 *   1 column @ 375px, 2 @ 768px, 3 @ 1280px.
 */
import { IndustryTile } from "@/components/IndustryTile";
import { INDUSTRIES } from "@/lib/industries";

export default function Home() {
  return (
    <main className="container mx-auto max-w-6xl px-4 py-10">
      <header className="mb-10 space-y-2">
        <h1 className="text-3xl font-semibold tracking-tight md:text-4xl">
          Portfolio ML System
        </h1>
        <p className="max-w-2xl text-muted-foreground">
          Industry-specific demos across six verticals. Pick a domain to see
          the models in action — each one accepts real inputs, returns a live
          prediction, and shows you what drove it.
        </p>
      </header>

      <section
        aria-label="Industries"
        className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3"
      >
        {INDUSTRIES.map((industry) => (
          <IndustryTile
            key={industry.slug}
            href={industry.href}
            title={industry.title}
            tagline={industry.tagline}
            icon={industry.icon}
            modelCount={industry.modelCount}
          />
        ))}
      </section>
    </main>
  );
}
