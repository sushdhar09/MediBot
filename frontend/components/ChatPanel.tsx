"use client";

import { useEffect, useRef, useState } from "react";
import { ChatResponse, Session, Source, askQuestion } from "@/lib/api";

type Message = {
  id: number;
  author: "user" | "bot";
  text: string;
  retrievalType?: ChatResponse["retrieval_type"];
  sources?: Source[];
  blocked?: boolean;
};

const SUGGESTIONS: Record<string, string[]> = {
  doctor: [
    "What is the first-hour sepsis management protocol?",
    "Show me the antibiotic dosing in the drug formulary",
    "How many days of casual leave do I get?",
  ],
  nurse: [
    "What are the five moments of hand hygiene?",
    "IV cannula size for a paediatric patient under 5kg",
    "Show me all insurance billing codes",
  ],
  billing_executive: [
    "How many claims are still pending?",
    "What is the total claimed amount per department?",
    "What documents are required for a cashless claim?",
  ],
  technician: [
    "What does fault code F-05 mean on the infusion pump?",
    "What is the calibration schedule for ventilators?",
    "List the ICU nursing procedures",
  ],
  admin: [
    "Which equipment category has the most open maintenance tickets?",
    "How many claims were submitted in December 2024?",
    "Summarise the code of conduct",
  ],
};

export default function ChatPanel({
  session,
  onLogout,
}: {
  session: Session;
  onLogout: () => void;
}) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, busy]);

  async function send(question: string) {
    const text = question.trim();
    if (!text || busy) return;

    const id = Date.now();
    setMessages((prev) => [...prev, { id, author: "user", text }]);
    setDraft("");
    setBusy(true);
    try {
      const response = await askQuestion(session.token, text);
      setMessages((prev) => [
        ...prev,
        {
          id: id + 1,
          author: "bot",
          text: response.answer,
          retrievalType: response.retrieval_type,
          sources: response.sources,
          blocked: response.access_denied,
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: id + 1,
          author: "bot",
          text: err instanceof Error ? err.message : "Something went wrong.",
          blocked: true,
        },
      ]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="chat">
      <header className="chat-header">
        <div>
          <strong>MediBot assistant</strong>
          <div className="subtitle" style={{ margin: 0 }}>
            Answers are drawn only from documents your role may access.
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span className="role-badge">{session.role.replace("_", " ")}</span>
          <button className="link-btn" style={{ width: "auto", marginTop: 0 }} onClick={onLogout}>
            Sign out
          </button>
        </div>
      </header>

      <div className="chat-log" ref={logRef}>
        {messages.length === 0 && (
          <div className="empty-state">
            <h2 style={{ marginBottom: 8 }}>Ask MediBot anything</h2>
            <p>
              Signed in as <b>{session.display_name}</b>. You can search the{" "}
              <b>{session.collections.join(", ")}</b> collections.
            </p>
          </div>
        )}

        {messages.map((message) => (
          <div
            key={message.id}
            className={`msg ${message.author} ${message.blocked ? "blocked" : ""}`}
          >
            {message.author === "bot" && (
              <span
                className={`tag ${
                  message.blocked
                    ? "blocked"
                    : message.retrievalType === "sql_rag"
                      ? "sql"
                      : "hybrid"
                }`}
              >
                {message.blocked
                  ? "Access restricted"
                  : message.retrievalType === "sql_rag"
                    ? "SQL RAG"
                    : "Hybrid RAG"}
              </span>
            )}
            <div>{message.text}</div>

            {message.sources && message.sources.length > 0 && (
              <div className="sources">
                <strong>Sources</strong>
                {message.sources.map((source, index) => (
                  <div className="source-item" key={`${source.source_document}-${index}`}>
                    [{index + 1}] <b>{source.source_document}</b> - {source.section_title}{" "}
                    <i>({source.collection})</i>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}

        {busy && <div className="msg bot">Searching your permitted documents...</div>}
      </div>

      <div className="suggestions">
        {(SUGGESTIONS[session.role] ?? []).map((suggestion) => (
          <button key={suggestion} onClick={() => send(suggestion)} disabled={busy}>
            {suggestion}
          </button>
        ))}
      </div>

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          send(draft);
        }}
      >
        <textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send(draft);
            }
          }}
          placeholder="Ask about protocols, policies, equipment, billing..."
        />
        <button disabled={busy || !draft.trim()}>Send</button>
      </form>
    </section>
  );
}
