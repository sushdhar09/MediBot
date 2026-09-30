# Agent notes

- Python: `.venv\Scripts\python.exe` (3.12). Run everything from `backend/`.
- Tests (offline, no network): `..\.venv\Scripts\python.exe -m pytest -q`. `-m live` tests need Groq.
- RAGAS eval: `python -m scripts.ragas_eval` (needs Groq; stop the API first - embedded Qdrant is single-process).
- `ragas==0.4.3` requires `langchain-community==0.4.1`; 0.4.2 removed a module ragas imports.
- On the Dell corporate network, api.groq.com is blocked by the firewall, so live LLM calls fail with a 503.
- The README has an unresolved merge-conflict block in "Tool choices and substitutions".
