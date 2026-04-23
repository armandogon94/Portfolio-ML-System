import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";

// Local Geist fonts bundled by create-next-app@14. Next 15's
// next/font/google `Geist` helper doesn't exist yet on 14, so we stick
// with next/font/local. A.2.3 rewrites this layout with providers; this
// minimal layout only needs to render without type errors.
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
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased`}>
        {children}
      </body>
    </html>
  );
}
