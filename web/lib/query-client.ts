import { QueryClient } from "@tanstack/react-query";

/**
 * App-wide singleton QueryClient.
 *
 * Importing from anywhere should yield the same instance. QueryClient
 * holds the in-memory cache, retry policy, and mutation defaults, and
 * duplicating it would fragment the cache across React trees. The
 * Providers tree references this singleton exactly once.
 *
 * Defaults are intentionally conservative for a portfolio demo:
 * - retry: 1 (one retry on flaky local FastAPI is enough; more masks bugs)
 * - refetchOnWindowFocus: false (demo, not dashboard, so don't re-hit the API
 *   every time the user tabs away and back)
 * - staleTime: 0 (always refetch on mount; demos want fresh predictions)
 */
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
      staleTime: 0,
    },
    mutations: {
      retry: 0,
    },
  },
});
