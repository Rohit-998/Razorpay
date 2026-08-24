import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PayRevive — Recovery intelligence",
  description: "AI-powered payment recovery command center",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
