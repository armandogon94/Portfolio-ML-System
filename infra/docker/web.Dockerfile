# web.Dockerfile — production image for the Next.js frontend.
#
# Uses Next.js `output: "standalone"` (see web/next.config.mjs) so the
# runner stage only carries the files `next start` actually needs.
# Multi-stage keeps the final image small (<250 MB expected): a deps
# stage for cached pnpm install, a builder stage for `next build`, and
# a lean runner stage with a non-root user.
#
# Build context is the repo root so the image can also reach files like
# .env.example if ever needed; the actual COPY scope is web/.

# ───── 1. Base image ─────────────────────────────────────────────────
FROM node:20-alpine AS base
# corepack ships with node 20 — pin the pnpm version matching
# web/package.json's packageManager field so local + container agree.
RUN apk add --no-cache libc6-compat \
 && corepack enable \
 && corepack prepare pnpm@10.33.0 --activate
WORKDIR /app

# ───── 2. Install dependencies (cached layer) ────────────────────────
FROM base AS deps
# Only copy lockfile + manifest so dep installs cache across rebuilds
# when source changes but deps don't.
COPY web/package.json web/pnpm-lock.yaml ./web/
WORKDIR /app/web
RUN pnpm install --frozen-lockfile

# ───── 3. Build (Next.js standalone output) ──────────────────────────
FROM base AS builder
COPY --from=deps /app/web/node_modules ./web/node_modules
COPY web ./web
WORKDIR /app/web
ENV NEXT_TELEMETRY_DISABLED=1
RUN pnpm build

# ───── 4. Runner (minimal production image) ──────────────────────────
FROM node:20-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3070 \
    HOSTNAME=0.0.0.0

# Non-root user — standard Next.js convention (uid 1001).
RUN addgroup --system --gid 1001 nodejs \
 && adduser --system --uid 1001 nextjs

# Copy only the Next.js standalone output + static + public assets.
# The standalone bundle already includes the minimal node_modules the
# server needs, so we don't run pnpm install here.
COPY --from=builder --chown=nextjs:nodejs /app/web/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/web/.next/static ./.next/static
COPY --from=builder --chown=nextjs:nodejs /app/web/public ./public

USER nextjs
EXPOSE 3070

# Liveness: fetch the landing page. wget is built into busybox on
# alpine so no extra package needed. --spider discards the body.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD wget --quiet --spider http://localhost:3070/ || exit 1

CMD ["node", "server.js"]
