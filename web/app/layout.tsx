import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";

import { Providers } from "@/app/providers";
import { Nav } from "@/components/Nav";

// Local Geist fonts bundled by create-next-app@14. Next 15's
// next/font/google `Geist` helper doesn't exist yet on 14.
const geistSans = localFont({
  src: "./fonts/GeistVF.woff",
  variable: "--font-geist-sans",
  weight: "100 900",
});
const geistMono = localFont({
  src: "./fonts/GeistMonoVF.woff",
  variable: "--font-geist-mono",
  weight: "100 900",
});

export const metadata: Metadata = {
  title: "Portfolio ML System",
  description: "Industry-specific ML demos across 6 domains",
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
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased`}>
        <Providers>
          <Nav />
          {children}
        </Providers>
      </body>
    </html>
  );
}
