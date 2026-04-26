/**
 * Nav — top bar for every page.
 *
 * Layout: project title on the left (links to /), Dashboard link in
 * the middle (added in A.9.9), ThemeToggle on the right. Sticky to
 * the top with a subtle border + scroll-under blur. Server
 * Component (ThemeToggle is the only client bit).
 */
import Link from "next/link";

import { ThemeToggle } from "@/components/ThemeToggle";

export function Nav() {
  return (
    <header className="sticky top-0 z-50 w-full border-b border-border bg-background/80 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto flex h-14 max-w-6xl items-center justify-between gap-4 px-4">
        <Link
          href="/"
          className="text-base font-semibold tracking-tight transition-colors hover:text-primary"
        >
          Portfolio ML System
        </Link>
        <nav className="flex items-center gap-4 text-sm">
          <Link
            href="/dashboard"
            className="text-muted-foreground transition-colors hover:text-foreground"
          >
            Dashboard
          </Link>
        </nav>
        <ThemeToggle />
      </div>
    </header>
  );
}
