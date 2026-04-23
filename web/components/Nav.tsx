/**
 * Nav — top bar for every page.
 *
 * Simple layout: project title on the left (links to /), ThemeToggle
 * on the right. Sticky to the top, with a subtle border and semi-
 * transparent background for the scroll-under effect. Server
 * Component (ThemeToggle is the only client bit).
 */
import Link from "next/link";

import { ThemeToggle } from "@/components/ThemeToggle";

export function Nav() {
  return (
    <header className="sticky top-0 z-50 w-full border-b border-border bg-background/80 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto flex h-14 max-w-6xl items-center justify-between px-4">
        <Link
          href="/"
          className="text-base font-semibold tracking-tight transition-colors hover:text-primary"
        >
          Portfolio ML System
        </Link>
        <ThemeToggle />
      </div>
    </header>
  );
}
