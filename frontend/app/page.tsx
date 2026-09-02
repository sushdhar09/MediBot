"use client";

import { useEffect, useState } from "react";
import ChatPanel from "@/components/ChatPanel";
import LoginScreen from "@/components/LoginScreen";
import Sidebar from "@/components/Sidebar";
import { CollectionInfo, Session, fetchCollections } from "@/lib/api";

const STORAGE_KEY = "medibot.session";

export default function Home() {
  const [session, setSession] = useState<Session | null>(null);
  const [access, setAccess] = useState<CollectionInfo | null>(null);
  const [restored, setRestored] = useState(false);

  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) setSession(JSON.parse(stored) as Session);
    setRestored(true);
  }, []);

  useEffect(() => {
    if (!session) {
      setAccess(null);
      return;
    }
    localStorage.setItem(STORAGE_KEY, JSON.stringify(session));
    fetchCollections(session.role).then(setAccess).catch(() => setAccess(null));
  }, [session]);

  function logout() {
    localStorage.removeItem(STORAGE_KEY);
    setSession(null);
  }

  if (!restored) return null;
  if (!session) return <LoginScreen onLogin={setSession} />;

  return (
    <main className="shell">
      <Sidebar session={session} access={access} onLogout={logout} />
      <ChatPanel session={session} onLogout={logout} />
    </main>
  );
}
