import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MediBot - MediAssist Health Network",
  description: "Role-aware internal assistant for MediAssist Health Network",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
