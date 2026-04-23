/** @type {import('next').NextConfig} */

// `output: "standalone"` keeps the Docker image small (`next start`
// can ship with only the files Next.js actually uses — see
// Dockerfile.web landing in A.2.10). `rewrites` proxies /api/* to
// the FastAPI backend server-side, so the browser never leaves the
// Next.js origin and we don't need CORS middleware on the Python side.
//
//   - INTERNAL_API_URL is the server-only destination:
//       * local dev:  http://localhost:8070
//       * Docker:     http://ml-api:8000
//   - NEXT_PUBLIC_API_URL is reserved for future direct client-side
//     calls that opt out of the proxy. Currently unused by lib/api.ts.

const INTERNAL_API_URL = process.env.INTERNAL_API_URL ?? "http://localhost:8070";

const nextConfig = {
  output: "standalone",
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${INTERNAL_API_URL}/:path*`,
      },
    ];
  },
};

export default nextConfig;
