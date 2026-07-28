/** Fintech landing page for the three implemented model demos. */
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
          Explore payment fraud, consumer credit risk, and card attrition: the
          three fintech problems implemented in this system.
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
