"use client";

/**
 * Client-side provider tree for the app.
 *
 * Wraps children in:
 * 1. QueryClientProvider: exposes the singleton TanStack Query client
 *    to every component via context. Mutations + queries read from the
 *    same cache.
 * 2. ThemeProvider (next-themes): class-based light/dark theme with
 *    localStorage persistence. `attribute="class"` toggles `.dark` on
 *    <html>, matching our tailwind.config.ts `darkMode: ["class"]`.
 *    `disableTransitionOnChange` avoids a flash of mid-transition
 *    colors when the theme switches.
 *
 * Kept as a client component so the QueryClient can live in client
 * memory; the root layout (Server Component) just imports + wraps.
 */

import type { ReactNode } from "react";
import { QueryClientProvider } from "@tanstack/react-query";
import { ThemeProvider } from "next-themes";

import { queryClient } from "@/lib/query-client";
import { Toaster } from "@/components/ui/sonner";

export function Providers({ children }: { children: ReactNode }) {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider
        attribute="class"
        defaultTheme="system"
        enableSystem
        disableTransitionOnChange
      >
        {children}
        {/* Toast portal, mounted once at the provider tree root so any
            page can call `toast.error(...)` (sonner) without worrying
            about local containers. */}
        <Toaster richColors closeButton position="top-right" />
      </ThemeProvider>
    </QueryClientProvider>
  );
}
