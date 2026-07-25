import type { Metadata } from "next";
import "./globals.css";

import { Providers } from "@/app/providers";
import { Nav } from "@/components/Nav";

export const metadata: Metadata = {
  title: "Portfolio ML System",
  description:
    "Fintech ML demos for payment fraud, consumer credit risk, and card attrition",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    // suppressHydrationWarning: next-themes toggles the `class` attribute
    // on <html> before React hydrates; without this the server-rendered
    // class ("") mismatches the client's ("dark" or "light") and React
    // emits a warning for every page load.
    <html lang="en" suppressHydrationWarning>
      <body className="antialiased">
        <Providers>
          <Nav />
          {children}
        </Providers>
      </body>
    </html>
  );
}
