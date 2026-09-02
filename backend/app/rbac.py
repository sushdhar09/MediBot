"""Role-based access control matrix.

This module is the single source of truth for "who may see what". The ingestion
pipeline stamps every chunk with `access_roles` derived from here, and every
retrieval query filters on that same field inside Qdrant.
"""
from __future__ import annotations

ROLES = ("doctor", "nurse", "billing_executive", "technician", "admin")

# collection -> roles allowed to retrieve chunks from it
COLLECTION_ACCESS: dict[str, list[str]] = {
    "general": ["doctor", "nurse", "billing_executive", "technician", "admin"],
    "clinical": ["doctor", "admin"],
    "nursing": ["nurse", "doctor", "admin"],
    "billing": ["billing_executive", "admin"],
    "equipment": ["technician", "admin"],
}

COLLECTION_DESCRIPTIONS: dict[str, str] = {
    "general": "HR handbook, leave policy, code of conduct, general FAQs",
    "clinical": "treatment protocols, drug formulary, diagnostic reference",
    "nursing": "ICU nursing procedures, infection control guidelines",
    "billing": "insurance billing codes, claim submission guide",
    "equipment": "equipment operation, calibration and maintenance manuals",
}

# Only roles with analytical responsibilities may query the operational database.
SQL_RAG_ROLES = frozenset({"billing_executive", "admin"})

ROLE_DEPARTMENTS: dict[str, str] = {
    "doctor": "Clinical",
    "nurse": "Clinical",
    "billing_executive": "Billing & Insurance",
    "technician": "Medical Equipment",
    "admin": "Executive / IT",
}

# Keyword hints used *only* to phrase a helpful refusal message. They never
# influence what is retrieved - that is decided by the Qdrant metadata filter.
_TOPIC_HINTS: dict[str, tuple[str, ...]] = {
    "billing": (
        "billing", "invoice", "insurer", "insurance", "claim", "reimburse",
        "cashless", "icd", "copay", "co-pay", "deductible", "tariff", "pre-auth",
        "preauth", "denial", "tpa", "settlement",
    ),
    "clinical": (
        "drug", "formulary", "dosage", "dose", "mg", "prescri", "treatment protocol",
        "diagnos", "contraindicat", "therapy", "antibiotic", "titration",
    ),
    "nursing": (
        "nursing", "nurse", "icu", "cannula", "catheter", "infection control",
        "hand hygiene", "bedside", "vitals", "ppe", "wound",
    ),
    "equipment": (
        "equipment", "calibrat", "maintenance", "ventilator", "infusion pump",
        "defibrillat", "fault code", "servicing", "sterilis", "steriliz", "manual",
    ),
    "general": (
        "leave", "holiday", "hr ", "handbook", "code of conduct", "faq",
        "attendance", "payroll", "grievance", "dress code",
    ),
}


def collections_for_role(role: str) -> list[str]:
    """Collections a role may retrieve from, in a stable display order."""
    return [c for c, roles in COLLECTION_ACCESS.items() if role in roles]


def roles_for_collection(collection: str) -> list[str]:
    try:
        return list(COLLECTION_ACCESS[collection])
    except KeyError as exc:
        raise ValueError(f"Unknown collection: {collection!r}") from exc


def can_use_sql_rag(role: str) -> bool:
    return role in SQL_RAG_ROLES


def guess_topic_collection(question: str) -> str | None:
    """Best-effort guess of which collection a question is aiming at.

    Used to turn a blocked query into an informative message rather than a
    generic "no results found".
    """
    text = question.lower()
    scores = {
        collection: sum(1 for kw in keywords if kw in text)
        for collection, keywords in _TOPIC_HINTS.items()
    }
    best = max(scores, key=lambda c: scores[c])
    return best if scores[best] > 0 else None


def access_denied_message(role: str, blocked_collection: str | None = None) -> str:
    allowed = collections_for_role(role)
    allowed_str = ", ".join(allowed)
    label = role.replace("_", " ")
    article = "an" if label[0] in "aeiou" else "a"
    if blocked_collection and blocked_collection not in allowed:
        return (
            f"As {article} {label}, you don't have access to {blocked_collection} documents "
            f"({COLLECTION_DESCRIPTIONS[blocked_collection]}). "
            f"I can only answer questions from the {allowed_str} collections."
        )
    return (
        f"I couldn't find anything in the documents you're authorised to read. "
        f"As {article} {label}, your access is limited to the {allowed_str} collections."
    )


def sql_denied_message(role: str) -> str:
    label = role.replace("_", " ")
    article = "an" if label[0] in "aeiou" else "a"
    return (
        f"That's an analytical question answered from the operational database "
        f"(claims and maintenance tickets). As {article} {label}, you don't have "
        f"access to it - database analytics are restricted to billing executives "
        f"and admins. I can still answer document questions from the "
        f"{', '.join(collections_for_role(role))} collections."
    )
