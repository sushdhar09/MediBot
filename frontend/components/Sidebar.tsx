"use client";

import { CollectionInfo, Session } from "@/lib/api";

export default function Sidebar({
  session,
  access,
  onLogout,
}: {
  session: Session;
  access: CollectionInfo | null;
  onLogout: () => void;
}) {
  return (
    <aside className="sidebar">
      <div className="brand" style={{ fontSize: 20 }}>
        Medi<span>Bot</span>
      </div>
      <p className="subtitle" style={{ marginBottom: 14 }}>
        MediAssist Health Network
      </p>

      <div className="role-badge">{session.role.replace("_", " ")}</div>
      <p className="subtitle" style={{ margin: "10px 0 0" }}>
        {session.display_name}
        <br />
        {session.department}
      </p>

      <h3>Accessible collections</h3>
      {(access?.collections ?? session.collections.map((name) => ({ name, description: "" }))).map(
        (collection) => (
          <div className="collection-pill allowed" key={collection.name}>
            {collection.name}
            {collection.description && <small>{collection.description}</small>}
          </div>
        )
      )}

      {access && access.restricted_collections.length > 0 && (
        <>
          <h3>Restricted</h3>
          {access.restricted_collections.map((name) => (
            <div className="collection-pill denied" key={name}>
              {name}
              <small>blocked at the vector store</small>
            </div>
          ))}
        </>
      )}

      <h3>Database analytics</h3>
      <div className={`collection-pill ${access?.sql_rag_enabled ? "allowed" : "denied"}`}>
        SQL RAG
        <small>
          {access?.sql_rag_enabled
            ? "claims + maintenance_tickets"
            : "restricted to billing & admin"}
        </small>
      </div>

      <button className="link-btn" onClick={onLogout}>
        Sign out
      </button>
    </aside>
  );
}
