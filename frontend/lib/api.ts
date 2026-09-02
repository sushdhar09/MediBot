export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";

export type Session = {
  token: string;
  username: string;
  display_name: string;
  role: string;
  department: string;
  collections: string[];
};

export type Source = {
  source_document: string;
  section_title: string;
  collection: string;
};

export type ChatResponse = {
  answer: string;
  sources: Source[];
  retrieval_type: "hybrid_rag" | "sql_rag";
  role: string;
  access_denied: boolean;
  debug?: Record<string, unknown> | null;
};

export type DemoUser = {
  username: string;
  password: string;
  role: string;
  display_name: string;
  department: string;
};

export type CollectionInfo = {
  role: string;
  department: string;
  collections: { name: string; description: string }[];
  restricted_collections: string[];
  sql_rag_enabled: boolean;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!response.ok) {
    const detail = await response
      .json()
      .then((body) => body.detail as string)
      .catch(() => response.statusText);
    throw new Error(detail || `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export const login = (username: string, password: string) =>
  request<Session>("/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });

export const fetchDemoUsers = () => request<DemoUser[]>("/demo-users");

export const fetchCollections = (role: string) =>
  request<CollectionInfo>(`/collections/${role}`);

export const askQuestion = (token: string, question: string) =>
  request<ChatResponse>("/chat", {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
    body: JSON.stringify({ question }),
  });
