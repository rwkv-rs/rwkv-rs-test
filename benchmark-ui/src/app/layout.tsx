import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RWKV Benchmark Explorer",
  description: "Interactive Task 5 throughput explorer for RWKV inference backends."
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
