import sys
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()

MODEL = "gemini-3.5-flash-lite"

CATEGORIES = [
    "Fire, Lightning, Explosion",
    "Storm",
    "Escape of Water",
    "Flood",
    "Landslip, Subsidence, Earthquake",
    "Theft",
    "Impact",
    "Riot, Malicious Damage",
    "Accidental Damage",
    "Power Surge",
    "Special Perils Extension",
    "Alternative Accommodation",
    "Loss of Rent",
    "Tracing and Access",
    "Replacement of Locks",
    "Buildings or Contents Sum Insured",
    "Hospitalisation, Inpatient Benefits",
    "Pre-hospitalisation, Post-hospitalisation",
    "Outpatient Treatment",
    "Co-insurance",
    "Overall Annual Limit",
    "Waiting Period",
    "Pre-existing Condition",
    "Government Hospital Daily Cash Allowance",
    "Premium Payment, Grace Period, Lapse",
    "Premium Deferment Facility",
    "Reinstatement",
    "Cancellation",
    "Tier Upgrade or Downgrade",
    "Claims Process, Documents Required",
    "Other, General",
]
_CATEGORY_LOOKUP = {c.lower(): c for c in CATEGORIES}
TIERS = ["Silver", "Gold", "Platinum"]

CLASSIFY_PROMPT = f"""You classify a customer's plain-language insurance question into exactly ONE category from this fixed list, using your own general knowledge of how insurance policies describe things (e.g. a burst pipe causing water damage is usually called "Escape of Water"). You do NOT have access to this specific policy's documents.

Categories (copy one EXACTLY as written):
{chr(10).join('- ' + c for c in CATEGORIES)}

Also identify the plan tier (Silver, Gold, or Platinum) if the question mentions or clearly implies one; otherwise "none".

Respond in EXACTLY this format, nothing else, no explanation:
category: <one category copied exactly from the list>
tier: <Silver, Gold, Platinum, or none>"""


def classify_question(question, model=MODEL, retries=4):
    for attempt in range(retries):
        try:
            client = genai.Client()
            response = client.models.generate_content(
                model=model,
                contents=question,
                config=types.GenerateContentConfig(
                    system_instruction=CLASSIFY_PROMPT,
                    max_output_tokens=60,
                    temperature=0,
                    thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )
            return _parse(response.text or "")
        except (errors.ClientError, errors.APIError) as e:
            code = getattr(e, "code", None)
            transient = code in (429, 500, 502, 503)
            if not transient or attempt == retries - 1:
                print(f"      classify_question() failed for {question[:50]!r}: {type(e).__name__} {code}", file=sys.stderr)
                return {"category": None, "tier": None}
            wait = min(35, 12 * (attempt + 1))
            print(f"      classify_question() rate-limited/transient, retry in {wait}s ({attempt + 1}/{retries})", file=sys.stderr)
            time.sleep(wait)
        except Exception as e:
            if attempt == retries - 1:
                print(f"      classify_question() failed for {question[:50]!r}: {type(e).__name__}: {e}", file=sys.stderr)
                return {"category": None, "tier": None}
            wait = min(35, 12 * (attempt + 1))
            print(f"      classify_question() network error ({type(e).__name__}), retry in {wait}s ({attempt + 1}/{retries})", file=sys.stderr)
            time.sleep(wait)
    return {"category": None, "tier": None}


def _parse(text):
    category, tier = None, None
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("category:"):
            raw = line.split(":", 1)[1].strip()
            category = _CATEGORY_LOOKUP.get(raw.lower())
        elif line.lower().startswith("tier:"):
            raw = line.split(":", 1)[1].strip().lower()
            tier = next((t for t in TIERS if t.lower() == raw), None)
    return {"category": category, "tier": tier}


def search_terms(classification):
    parts = [p for p in (classification.get("category"), classification.get("tier")) if p]
    return ", ".join(parts)
