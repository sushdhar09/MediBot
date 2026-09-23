"""Deterministic, zero-cost guardrail checks.

These run first, before any LLM judge, and produce one of three outcomes per
rule set:

  * a `block` signal  - unambiguous, decided here, no LLM call is made
  * a `suspect` signal - inconclusive, escalated to the OpenEvals judge
  * nothing            - the text is clean on this axis

Regexes are the floor, not the ceiling: they catch the cheap, high-volume
attacks instantly and reliably, and hand the genuinely ambiguous cases to the
judge in `judge.py`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .. import rbac

Severity = str  # "block" | "suspect"


@dataclass(frozen=True)
class Signal:
    category: str
    severity: Severity
    detail: str


@dataclass(frozen=True)
class Rule:
    category: str
    severity: Severity
    pattern: re.Pattern[str]
    detail: str


def _rule(
    category: str, severity: Severity, pattern: str, detail: str, flags: int = re.IGNORECASE
) -> Rule:
    return Rule(category, severity, re.compile(pattern, flags), detail)


# --- Input rules ------------------------------------------------------------

INPUT_RULES: tuple[Rule, ...] = (
    # Prompt injection: overriding or extracting the system prompt.
    _rule(
        "prompt_injection", "block",
        r"\b(?:ignore|disregard|forget|override)\s+(?:all\s+|any\s+|the\s+)*"
        r"(?:previous|prior|above|earlier|preceding|your|these)\s+"
        r"(?:instructions?|rules?|prompts?|directions?|guidelines?|restrictions?)",
        "instruction-override phrase",
    ),
    _rule(
        "prompt_injection", "block",
        r"\b(?:reveal|show|print|repeat|output|display|dump|leak)\s+(?:me\s+)?"
        r"(?:your|the)\s+(?:full\s+|initial\s+|original\s+|hidden\s+)?"
        r"(?:system\s+prompt|system\s+message|instructions?|prompt|rules)\b",
        "system prompt extraction",
    ),
    _rule(
        "prompt_injection", "block",
        r"\b(?:developer\s+mode|dan\s+mode|jailbreak|do\s+anything\s+now|"
        r"unrestricted\s+mode|god\s+mode|sudo\s+mode)\b",
        "jailbreak persona",
    ),
    _rule(
        "prompt_injection", "block",
        r"\byou\s+are\s+now\s+(?:a|an|in|no\s+longer)\b|"
        r"\bfrom\s+now\s+on\s+you\s+(?:are|will|must)\b|"
        r"\bpretend\s+(?:to\s+be|you(?:'re|\s+are|\s+have))\b|"
        r"\bact\s+as\s+(?:an?\s+)?(?:admin|administrator|root|superuser|system|developer|dba)\b",
        "persona reassignment",
    ),
    _rule(
        "prompt_injection", "block",
        r"</?\s*(?:system|assistant|instructions?)\s*>|\[\s*(?:system|inst|/inst)\s*\]|"
        r"<\|[^|]{0,40}\|>|\bBEGIN\s+SYSTEM\b|#{2,}\s*(?:system|new)\s+instructions?",
        "delimiter / role-tag injection",
    ),
    _rule(
        "prompt_injection", "block",
        r"\b(?:system|admin(?:istrator)?|security)\s+(?:override|bypass|mode)\b",
        "fake system directive",
    ),
    # Role and access-control override.
    _rule(
        "access_override", "block",
        r"\b(?:bypass|override|disable|turn\s+off|switch\s+off|circumvent|skip|ignore|"
        r"disregard|forget|drop|lift|relax|suspend)\s+"
        r"(?:the\s+|your\s+|all\s+)?(?:access\s+control|access\s+restrictions?|rbac|"
        r"permissions?|restrictions?|authori[sz]ation|role\s+check|security|guardrails?|filters?)",
        "access-control override request",
    ),
    _rule(
        "access_override", "block",
        r"\b(?:regardless|irrespective)\s+of\s+(?:my\s+|the\s+)?"
        r"(?:role|permissions?|access|clearance|restrictions?)|"
        r"\beven\s+(?:if|though)\s+(?:it(?:'s|\s+is)\s+|i(?:'m|\s+am)\s+)?"
        r"(?:restricted|not\s+allowed|not\s+authori[sz]ed|unauthori[sz]ed|confidential|forbidden)",
        "explicit restriction bypass",
    ),
    _rule(
        "access_override", "block",
        r"\b(?:grant|give|elevate|escalate)\s+(?:me\s+)?(?:full\s+|admin\s+|root\s+)?"
        r"(?:access|privileges?|permissions?|rights?)|"
        r"\b(?:change|switch|promote|set)\s+my\s+role\b|"
        r"\btreat\s+me\s+as\s+(?:an?\s+)?(?:admin|administrator|doctor|superuser)\b",
        "privilege escalation request",
    ),
    _rule(
        "access_override", "block",
        r"\bi\s*(?:'m|\s+am)\s+(?:actually\s+|really\s+)?(?:an?\s+)?"
        r"(?:admin|administrator|the\s+cio|the\s+cto|your\s+developer|your\s+creator)\b",
        "false authority claim",
    ),
    # Abuse / unsafe requests that are off-topic for a hospital assistant.
    _rule(
        "unsafe_request", "block",
        r"\bhow\s+(?:to|do\s+i|can\s+i)\s+(?:make|build|synthesi[sz]e|obtain|acquire)\s+"
        r"(?:a\s+|an\s+)?(?:bomb|explosive|weapon|gun|meth|cocaine|poison)\b",
        "harmful how-to request",
    ),
    _rule(
        "unsafe_request", "block",
        r"\bhow\s+(?:to|do\s+i|can\s+i)\s+(?:kill|harm|hurt|poison|sedate|overdose)\s+"
        r"(?:a\s+|the\s+|my\s+|someone|him|her|them|patients?)",
        "request to cause harm",
    ),
    _rule(
        "abusive_language", "block",
        r"\b(?:fuck|fucking|shit|bitch|bastard|asshole|dickhead|retard)\b",
        "abusive language",
    ),
    # Inconclusive - escalate to the judge rather than guessing.
    _rule(
        "prompt_injection", "suspect",
        r"\bsystem\s+prompt\b|\byour\s+instructions\b|\bprompt\s+injection\b",
        "mentions the system prompt",
    ),
    _rule(
        "prompt_injection", "suspect",
        r"\bhypothetical(?:ly)?\b|\brole[-\s]?play\b|\blet's\s+play\b|\bimagine\s+you\b|"
        r"\bin\s+a\s+story\b|\bfor\s+a\s+novel\b",
        "hypothetical / roleplay framing",
    ),
    _rule(
        "access_override", "suspect",
        r"\bfor\s+(?:an?\s+)?(?:urgent\s+)?(?:audit|investigation|inspection|emergency)\b|"
        r"\bi\s+have\s+(?:clearance|authori[sz]ation|approval)\b|"
        r"\bmy\s+(?:supervisor|manager|hod|director)\s+(?:said|approved|asked)\b|"
        r"\btrust\s+me\b|\bthis\s+is\s+urgent\b",
        "social-engineering framing",
    ),
    _rule(
        "prompt_injection", "suspect",
        r"https?://|\bwww\.\w|\b[A-Za-z0-9+/]{60,}={0,2}\b",
        "external content or encoded payload",
    ),
)

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

# Anything a MediAssist staff member plausibly asks about. Used only to decide
# whether a question is *possibly* off-topic and therefore worth a judge call -
# never to decide the answer.
_IN_SCOPE_EXTRA = (
    "hospital", "patient", "ward", "shift", "roster", "staff", "doctor", "nurse",
    "pharmacy", "medicine", "medication", "surgery", "discharge", "admission",
    "appointment", "consent", "triage", "emergency", "radiology", "lab", "sample",
    "consultant", "department", "compliance", "audit", "sop", "report", "record",
    "mediassist", "medibot", "salary", "shift", "uniform", "training", "safety",
    "incident", "escalation", "insurance", "device", "machine", "sterile", "oxygen",
)
_GREETINGS = frozenset(
    {"hi", "hello", "hey", "thanks", "thank you", "ok", "okay", "bye", "good morning",
     "good evening", "good afternoon"}
)


def scan(text: str, rules: tuple[Rule, ...]) -> list[Signal]:
    return [
        Signal(rule.category, rule.severity, rule.detail)
        for rule in rules
        if rule.pattern.search(text)
    ]


def structural_problem(text: str) -> str | None:
    """Cheap sanity checks that do not need any pattern matching."""
    if not text.strip():
        return "empty question"
    if _CONTROL_CHARS.search(text):
        return "control characters in question"
    if len(text) > 2000:
        return f"question too long ({len(text)} chars)"
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if len(text) > 40 and non_ascii / len(text) > 0.4:
        return "predominantly non-ASCII payload"
    return None


def is_greeting(text: str) -> bool:
    stripped = text.strip().strip("!.?,").lower()
    return stripped in _GREETINGS


def looks_in_scope(text: str) -> bool:
    """True if the question uses vocabulary this assistant is actually for."""
    lowered = text.lower()
    if any(rbac.topic_hits(lowered).values()):
        return True
    return any(word in lowered for word in _IN_SCOPE_EXTRA)


# --- Output rules -----------------------------------------------------------

# High-severity identifiers: an answer containing these is withheld entirely.
PII_BLOCK_RULES: tuple[Rule, ...] = (
    _rule("pii_leak", "block", r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b", "Aadhaar-format number"),
    # PAN is upper-case by definition; matching case-insensitively would flag
    # ordinary words followed by digits.
    _rule("pii_leak", "block", r"\b[A-Z]{5}\d{4}[A-Z]\b", "PAN-format identifier", flags=0),
    _rule("pii_leak", "block", r"\b\d{3}-\d{2}-\d{4}\b", "SSN-format number"),
    _rule(
        "credential_leak", "block",
        r"\b(?:api[_-]?key|secret[_-]?key|access[_-]?key|password|passwd|token)\s*[:=]\s*\S{6,}",
        "credential assignment",
    ),
    _rule("credential_leak", "block", r"\bgsk_[A-Za-z0-9]{20,}\b", "Groq API key"),
    _rule("credential_leak", "block", r"\bBearer\s+[A-Za-z0-9._\-]{20,}\b", "bearer token"),
)

# Lower-severity identifiers: masked in place so a useful answer survives.
PII_REDACT_RULES: tuple[Rule, ...] = (
    _rule("pii_contact", "redact", r"[\w.+-]+@[\w-]+\.[\w.]{2,}", "email address"),
    _rule("pii_contact", "redact", r"(?:\+91[\s-]?)?\b[6-9]\d{9}\b", "phone number"),
    _rule(
        "pii_patient", "redact",
        r"\b(?:MRN|UHID|patient[\s_-]?id)\s*[:#-]?\s*[A-Z]{0,3}\d{3,}\b",
        "patient identifier",
    ),
)

_REDACTION_LABELS = {
    "email address": "[redacted-email]",
    "phone number": "[redacted-phone]",
    "patient identifier": "[redacted-patient-id]",
}

# Internal detail that must never reach a staff member.
SYSTEM_LEAK_RULES: tuple[Rule, ...] = (
    _rule(
        "system_prompt_leak", "block",
        r"You are MediBot, the internal assistant|You are MediBot, reporting figures|"
        r"You translate questions about a hospital operations database",
        "verbatim system prompt",
    ),
    _rule("system_prompt_leak", "block", r"\baccess_roles\b|\bQDRANT_COLLECTION\b|\bGROQ_API_KEY\b",
          "internal configuration identifier"),
    _rule(
        "system_prompt_leak", "suspect",
        r"\bmy\s+(?:system\s+)?instructions\s+(?:are|say|state)\b|"
        r"\bI\s+was\s+(?:told|instructed)\s+to\b",
        "paraphrased system prompt",
    ),
    _rule("internal_detail_leak", "suspect", r"\bSELECT\b[\s\S]{0,200}\bFROM\b",
          "raw SQL in the answer"),
)

# Phrases that signal the model reached outside the retrieved context.
UNGROUNDED_HINT_RULES: tuple[Rule, ...] = (
    _rule(
        "fabrication", "suspect",
        r"\b(?:generally|typically|usually|in\s+most\s+hospitals|common\s+practice|"
        r"standard\s+practice|it\s+is\s+known|widely\s+accepted|as\s+an\s+ai|"
        r"i\s+believe|i\s+think|probably|presumably|to\s+my\s+knowledge)\b",
        "external-knowledge phrasing",
    ),
)

_CITATION_RE = re.compile(r"\[\d+\]")
# Numbers worth verifying: multi-digit values, decimals and anything with a unit.
_NUMBER_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|ml|mcg|g|kg|%|hours?|hrs?|days?|weeks?|months?|years?|"
    r"mmhg|bpm|inr|rs\.?|units?)\b|\b\d{2,}(?:\.\d+)?\b",
    re.IGNORECASE,
)
_LIST_MARKER_RE = re.compile(r"(?m)^\s*\d+[.)]\s")


_CARD_CANDIDATE_RE = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")


def _luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for index, char in enumerate(digits):
        value = int(char)
        if index % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def card_numbers(text: str) -> list[str]:
    """Digit runs that pass the Luhn check, i.e. plausible payment cards.

    Luhn keeps this from firing on invoice numbers and long claim ids, which a
    length-only regex would flag constantly.
    """
    found = []
    for match in _CARD_CANDIDATE_RE.finditer(text):
        digits = re.sub(r"\D", "", match.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            found.append(match.group(0).strip())
    return found


def redact_pii(text: str) -> tuple[str, list[Signal]]:
    """Mask low-severity identifiers, returning the cleaned text and what was hit."""
    signals: list[Signal] = []
    cleaned = text
    for rule in PII_REDACT_RULES:
        label = _REDACTION_LABELS.get(rule.detail, "[redacted]")
        cleaned, count = rule.pattern.subn(label, cleaned)
        if count:
            signals.append(Signal(rule.category, "redact", f"{rule.detail} x{count}"))
    return cleaned, signals


def unsupported_numbers(answer: str, context: str) -> list[str]:
    """Numeric claims in the answer that do not appear in the retrieved context.

    A dosage or a code the context never mentioned is the single most dangerous
    failure mode for a clinical assistant, so it is worth an explicit check
    before spending a judge call.
    """
    if not context:
        return []
    stripped = _LIST_MARKER_RE.sub("", _CITATION_RE.sub("", answer))
    context_digits = re.sub(r"[^\d.]", "", context)
    missing = []
    for match in _NUMBER_RE.finditer(stripped):
        digits = re.sub(r"[^\d.]", "", match.group(0))
        if digits and digits not in context_digits:
            missing.append(match.group(0).strip())
    return missing


def has_citations(answer: str) -> bool:
    return bool(_CITATION_RE.search(answer))


def restricted_topic_hits(answer: str, role: str) -> dict[str, int]:
    """Keyword hits in the answer belonging to collections this role cannot read."""
    allowed = set(rbac.collections_for_role(role))
    return {
        collection: count
        for collection, count in rbac.topic_hits(answer.lower()).items()
        if collection not in allowed and count
    }
