import type { Metadata } from "next";
import "./globals.css";
import Nav from "@/components/Nav";

export const metadata: Metadata = {
  title: "Uplift Targeting | HCP Call Planner",
  description: "Rep-facing HCP call planner ranked by incremental visit response, with verified reasons and model diagnostics.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Nav />
        <main className="container">{children}</main>
      </body>
    </html>
  );
}
