"use client";

/**
 * ThemeToggle — flips between light and dark themes via next-themes.
 *
 * next-themes handles localStorage persistence + the <html class>
 * update automatically; we just call setTheme with the opposite of
 * the current resolved value. During SSR/pre-hydration resolvedTheme
 * is undefined — we render an aria-labeled button regardless so the
 * markup is stable (prevents hydration mismatch).
 */
import { useTheme } from "next-themes";
import { Moon, Sun } from "lucide-react";

import { Button } from "@/components/ui/button";

export function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();
  const isDark = resolvedTheme === "dark";

  return (
    <Button
      variant="ghost"
      size="icon"
      aria-label={`Toggle theme (currently ${resolvedTheme ?? "loading"})`}
      onClick={() => setTheme(isDark ? "light" : "dark")}
    >
      {/* Render both icons; CSS hides the wrong one based on resolvedTheme.
          Showing the icon of the *target* theme is the common UX: user
          sees Moon when in light mode (meaning "click to go dark"). */}
      <Sun
        aria-hidden="true"
        className="h-4 w-4 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0"
      />
      <Moon
        aria-hidden="true"
        className="absolute h-4 w-4 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100"
      />
    </Button>
  );
}
