"use client";

import { useEffect, useState } from "react";
import { DemoUser, Session, fetchDemoUsers, login } from "@/lib/api";

export default function LoginScreen({
  onLogin,
}: {
  onLogin: (session: Session) => void;
}) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [demoUsers, setDemoUsers] = useState<DemoUser[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    fetchDemoUsers()
      .then(setDemoUsers)
      .catch(() =>
        setError("Cannot reach the MediBot API. Is the backend running on :8000?")
      );
  }, []);

  async function submit(user: string, pass: string) {
    setBusy(true);
    setError(null);
    try {
      onLogin(await login(user, pass));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="login-wrap">
      <div className="login-card">
        <form
          className="login-form"
          onSubmit={(e) => {
            e.preventDefault();
            submit(username, password);
          }}
        >
          <div className="brand">
            Medi<span>Bot</span>
          </div>
          <p className="subtitle">MediAssist Health Network - staff sign in</p>

          <label className="field">
            <span>Username</span>
            <input
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              autoComplete="username"
              placeholder="dr.mehta"
            />
          </label>
          <label className="field">
            <span>Password</span>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="current-password"
              placeholder="********"
            />
          </label>

          <button className="primary-btn" disabled={busy || !username || !password}>
            {busy ? "Signing in..." : "Sign in"}
          </button>
          {error && <p className="error">{error}</p>}
        </form>

        <aside className="login-demo">
          <h3 style={{ marginTop: 0, fontSize: 14 }}>Demo accounts</h3>
          <p className="subtitle" style={{ marginBottom: 18 }}>
            One per role - click to sign in and compare what each can access.
          </p>
          {demoUsers.map((user) => (
            <button
              key={user.username}
              className="demo-user"
              type="button"
              onClick={() => {
                setUsername(user.username);
                setPassword(user.password);
                submit(user.username, user.password);
              }}
            >
              {user.username} / {user.password}
              <small>
                {user.display_name} - {user.role} ({user.department})
              </small>
            </button>
          ))}
        </aside>
      </div>
    </div>
  );
}
