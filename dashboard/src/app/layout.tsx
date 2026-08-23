import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PayRevive — AI Payment Recovery Dashboard",
  description: "Real-time payment failure recovery powered by ML + LLM intelligence",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="min-h-screen">{children}</body>
    </html>
  );
}
