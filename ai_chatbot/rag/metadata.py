"""Hybrid metadata inference for Module objects.

Provides `infer_metadata(module_name, subsections_text)` which returns a dict:
    { 'category': <FOUNDATION|TACTICAL|OPERATIONS|LEGAL>, 'description': <short summary> }

Behavior:
- Use lightweight keyword heuristics to pick a category and produce a short description.
- If heuristics are low-confidence, call the project's ChatGoogleGenerativeAI model (Gemini)
  with a strict structured output request (pydantic) to get validated JSON.
- All failures are logged and the function falls back gracefully.

ID: 7mn3zn
"""
import logging
import re
from typing import Optional, Tuple

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class _ModuleMetadataModel(BaseModel):
    category: str
    description: str


_CATEGORIES = ["FOUNDATION", "TACTICAL", "OPERATIONS", "LEGAL"]

# Lightweight keyword maps for heuristic classification
_KEYWORDS = {
    "FOUNDATION": ["foundation", "foundational", "overview", "introduction", "basics", "principle"],
    "TACTICAL": ["tactic", "tactical", "procedure", "steps", "method", "practice", "guideline"],
    "OPERATIONS": ["operation", "operational", "operations", "maintenance", "process", "workflow", "runbook"],
    "LEGAL": ["law", "legislation", "legal", "statute", "regulation", "offence", "offense", "compliance", "court"],
}


def _heuristic_classify(module_name: str, text: str, name_weight: float = 2.0) -> Tuple[Optional[str], float]:
    """Return (best_category, confidence) where confidence is top_count/total_count.

    Returns (None, 0.0) when no keywords matched.
    """
    hay = ((module_name or "") + "\n" + (text or "")).lower()
    counts = {c: 0 for c in _CATEGORIES}
    for cat, kws in _KEYWORDS.items():
        for kw in kws:
            # simple substring count (word boundaries when reasonable)
            counts[cat] += len(re.findall(r"\b" + re.escape(kw) + r"\b", hay))

    # boost module-name-only matches
    if module_name:
        name_lower = module_name.lower()
        for cat, kws in _KEYWORDS.items():
            for kw in kws:
                if re.search(r"\b" + re.escape(kw) + r"\b", name_lower):
                    counts[cat] += int(name_weight)

    total = sum(counts.values())
    if total == 0:
        return None, 0.0

    best = max(counts.items(), key=lambda kv: kv[1])
    best_cat, best_count = best[0], best[1]
    confidence = float(best_count) / float(total) if total > 0 else 0.0
    return best_cat, confidence


def _short_description(text: str, max_chars: int = 300) -> str:
    if not text:
        return ""
    t = re.sub(r"\s+", " ", text).strip()
    # try to return first sentence-like fragment
    m = re.search(r"([A-Z][^\.\n]{10,}?[\.\?\!])", t)
    if m:
        s = m.group(1).strip()
        return s[:max_chars]
    return t[:max_chars]


def infer_metadata(module_name: str, subsections_text: str) -> dict:
    """Infer metadata for a Module using heuristics followed by an LLM fallback.

    Steps:
    - Run keyword heuristics; if confidence >= threshold, return heuristic result.
    - Otherwise call Gemini via ChatGoogleGenerativeAI with a strict pydantic schema.
    - On any failure, return a safe fallback (category FOUNDATION, short description).

    Configurable via Django settings (optional):
    - MODULE_METADATA_HEURISTIC_THRESHOLD (float, default 0.6)
    - METADATA_GEMINI_MODEL (str), METADATA_GEMINI_TEMPERATURE (float), METADATA_GEMINI_MAX_OUTPUT_TOKENS (int)
    - GEMINI_API_KEY (used by the LLM client)
    """
    # Prefer not to import Django settings at module import time in case this helper
    # is used in contexts where settings are not configured. Access settings lazily.
    try:
        from django.conf import settings
    except Exception:
        settings = None

    try:
        threshold = float(getattr(settings, "MODULE_METADATA_HEURISTIC_THRESHOLD", 0.6)) if settings is not None else 0.6
    except Exception:
        threshold = 0.6

    try:
        best_cat, conf = _heuristic_classify(module_name or "", subsections_text or "")
    except Exception as e:
        logger.debug("Heuristic classification failed: %s", e)
        best_cat, conf = None, 0.0

    # Force LLM generation for metadata to ensure AI-based descriptions
    # if best_cat and conf >= threshold:
    #     desc = _short_description(subsections_text or module_name or "")
    #     logger.info("Heuristic metadata chosen for '%s': %s (conf=%.2f)", module_name, best_cat, conf)
    #     return {"category": best_cat, "description": desc}

    # Low confidence -> attempt LLM
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage
    except Exception as e:
        logger.debug("LLM integration not available: %s", e)
        return {"category": best_cat or "FOUNDATION", "description": _short_description(subsections_text or module_name or "")}

    for attempt in range(3):
        try:
            model = getattr(settings, "METADATA_GEMINI_MODEL", "gemini-2.5-flash") if settings is not None else "gemini-2.5-flash"
            temp = float(getattr(settings, "METADATA_GEMINI_TEMPERATURE", 0.0)) if settings is not None else 0.0
            max_tokens = int(getattr(settings, "METADATA_GEMINI_MAX_OUTPUT_TOKENS", 512)) if settings is not None else 512
            api_key = getattr(settings, "GEMINI_API_KEY", None) if settings is not None else None

            llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=temp, max_output_tokens=max_tokens)

            prompt = f"""
Return a JSON object ONLY with two fields: category and description.
category must be one of: {', '.join(_CATEGORIES)}.
description must be a short (one-sentence) summary of the module (max 200 characters).

IMPORTANT: You must return valid JSON. Do not use unescaped double quotes inside the description.

Module name: {module_name}

Module content (truncated):
{(subsections_text or '')[:4000]}

Respond with strict JSON only, for example:
{{"category": "LEGAL", "description": "Short module summary"}}
"""

            response = llm.invoke([HumanMessage(content=prompt)])
            txt = getattr(response, "content", str(response))
            
            import json
            import re
            
            obj = None
            m = re.search(r"\{.*\}", txt, re.DOTALL)
            if m:
                try:
                    obj = json.loads(m.group(0))
                except Exception:
                    # Fallback for minor JSON errors like unescaped quotes
                    import ast
                    try:
                        obj = ast.literal_eval(m.group(0))
                    except Exception:
                        pass
            
            result = obj or {}

            category = (result.get("category") or "").upper()
            if category not in _CATEGORIES:
                raise ValueError(f"Invalid category from LLM: {category}")

            description = str(result.get("description") or "").strip()
            if not description:
                raise ValueError("LLM returned empty description")

            logger.info("LLM metadata for '%s': %s (Attempt %d)", module_name, category, attempt + 1)
            return {"category": category, "description": description}

        except Exception as e:
            logger.warning("Metadata LLM attempt %d failed for module '%s': %s", attempt + 1, module_name, e)
            if attempt == 2:
                # Final graceful fallback after all retries
                return {"category": best_cat or "FOUNDATION", "description": _short_description(subsections_text or module_name or "")}
            import time

