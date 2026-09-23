"""Streamlit UI for MediBot - calls the FastAPI backend."""
from __future__ import annotations

import os
import requests
import streamlit as st
from fastapi import FastAPI
app = FastAPI()
API_BASE = os.getenv("MEDIBOT_API_BASE", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(
    page_title="MediBot - MediAssist Health Network",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
/* Overall app background */
.stApp {
    background-color: #f9fafb;
    font-family: "Segoe UI", Roboto, sans-serif;
}

/* Inputs */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background-color: #ffffff;
    color: #1e293b;
    border: 1px solid #d1d5db;
    border-radius: 8px;
    padding: 8px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #0ea5e9, #0284c7);
    color: #ffffff;
    font-weight: 600;
    border-radius: 8px;
    padding: 8px 16px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    transition: 0.2s ease-in-out;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #0284c7, #0369a1);
}
.stButton > button:disabled {
    opacity: 0.5;
}

/* Chat messages */
.stChatMessage {
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 12px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
.stChatMessage.user {
    background-color: #e0f2fe;
    border-color: #bae6fd;
}
.stChatMessage.assistant {
    background-color: #ffffff;
}

/* Sidebar */
.stSidebar {
    background-color: #ffffff;
    border-right: 1px solid #e5e7eb;
    padding: 16px;
}
.stSidebar .stMarkdown {
    font-size: 0.95rem;
}

/* Titles and headers */
h1, h2, h3 {
    color: #0f172a;
    font-weight: 700;
}
h1 {
    border-bottom: 2px solid #e5e7eb;
    padding-bottom: 8px;
    margin-bottom: 16px;
}

/* Footer */
footer {
    text-align: center;
    font-size: 0.8rem;
    color: #6b7280;
    margin-top: 24px;
}
</style>
""",
    unsafe_allow_html=True,
)



def api_get(path: str) -> dict:
    resp = requests.get(f"{API_BASE}{path}", timeout=30)
    resp.raise_for_status()
    return resp.json()


def api_post(path: str, json: dict, headers: dict | None = None) -> dict:
    resp = requests.post(f"{API_BASE}{path}", json=json, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()


def login_screen() -> None:
    st.title("MediBot")
    try:
        demo_users = api_get("/demo-users")
    except requests.RequestException:
        st.error("Cannot reach the MediBot API. Is the backend running on :8000?")
        return

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### MediAssist Health Network - staff sign in")
        username = st.text_input("Username", key="login_username", placeholder="dr.mehta")
        password = st.text_input("Password", type="password", key="login_password", placeholder="********")
        if st.button("Sign in", use_container_width=True):
            try:
                session = api_post("/login", {"username": username, "password": password})
                st.session_state["session"] = session
                st.rerun()
            except requests.HTTPError as e:
                st.error(e.response.json().get("detail", "Login failed"))

    with col2:
        st.markdown("### Demo accounts")
        st.markdown("One per role - click to sign in and compare what each can access.")
        for user in demo_users:
            if st.button(
                f"{user['username']} / {user['password']}",
                key=f"demo_{user['username']}",
                use_container_width=True,
            ):
                try:
                    session = api_post("/login", {"username": user["username"], "password": user["password"]})
                    st.session_state["session"] = session
                    st.rerun()
                except requests.HTTPError as e:
                    st.error(e.response.json().get("detail", "Login failed"))


def main_app() -> None:
    session = st.session_state["session"]
    token = session["token"]
    role = session["role"]

    try:
        access = api_get(f"/collections/{role}")
    except requests.RequestException:
        st.error("Failed to fetch collection access info")
        access = None

    with st.sidebar:
        st.markdown("### MediBot")
        st.markdown("MediAssist Health Network")

        st.markdown(f"**Role:** {role.replace('_', ' ')}")
        st.markdown(f"**Name:** {session['display_name']}")
        st.markdown(f"**Department:** {session['department']}")

        st.markdown("---")
        st.markdown("### Accessible collections")
        for collection in (access or {}).get("collections", []):
            st.markdown(f"- {collection['name']}")
            if collection.get("description"):
                st.caption(collection["description"])

        if access and access.get("restricted_collections"):
            st.markdown("### Restricted")
            for name in access["restricted_collections"]:
                st.markdown(f"- ~~{name}~~")
                st.caption("blocked at the vector store")

        st.markdown("---")
        st.markdown("### Database analytics")
        sql_enabled = (access or {}).get("sql_rag_enabled", False)
        st.markdown(f"SQL RAG: {'✓' if sql_enabled else '✗'}")
        if sql_enabled:
            st.caption("claims + maintenance_tickets")
        else:
            st.caption("restricted to billing & admin")

        if st.button("Sign out", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    st.title("MediBot assistant")
    st.caption("Answers are drawn only from documents your role may access.")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                st.markdown("**Sources:**")
                for idx, source in enumerate(msg["sources"], start=1):
                    st.markdown(f"[{idx}] **{source['source_document']}** - {source['section_title']} *({source['collection']})*")
            if msg.get("retrieval_type"):
                st.caption(f"Retrieval: {msg['retrieval_type']}")
            if msg.get("access_denied"):
                st.warning("⚠️ Access restricted - this query targets documents outside your permitted collections.")

    if prompt := st.chat_input("Ask about protocols, policies, equipment, billing..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching your permitted documents..."):
                try:
                    response = api_post(
                        "/chat",
                        {"question": prompt},
                        headers={"Authorization": f"Bearer {token}"},
                    )
                except requests.HTTPError as e:
                    st.error(e.response.json().get("detail", "Request failed"))
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": "Request failed.", "access_denied": True}
                    )
                    st.stop()

            st.markdown(response["answer"])
            if response["sources"]:
                st.markdown("**Sources:**")
                for idx, source in enumerate(response["sources"], start=1):
                    st.markdown(f"[{idx}] **{source['source_document']}** - {source['section_title']} *({source['collection']})*")
            st.caption(f"Retrieval: {response['retrieval_type']}")
            if response["access_denied"]:
                st.warning("⚠️ Access restricted - this query targets documents outside your permitted collections.")

            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": response["answer"],
                    "sources": response["sources"],
                    "retrieval_type": response["retrieval_type"],
                    "access_denied": response["access_denied"],
                }
            )


def main() -> None:
    if "session" not in st.session_state:
        login_screen()
    else:
        main_app()


if __name__ == "__main__":
    main()
