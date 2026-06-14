"""
ContractCompass — FastAPI Backend
Analyzes legal contracts using Groq AI (Llama, Mixtral, etc.).
"""

import io
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, List, Literal

import openai
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ── Load environment ──────────────────────────────────────────────────────────
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
AI_MODEL = os.getenv("AI_MODEL", "llama-3.1-8b-instant")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. "
        "Add it to Backend/.env: GROQ_API_KEY=gsk_..."
    )

client = openai.OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
    default_headers={
        "Authorization": f"Bearer {GROQ_API_KEY}",
    },
)

# ── Pydantic Schemas for Structured Output ────────────────────────────────────

class RiskBreakdownItem(BaseModel):
    label: str = Field(description="Name of the risk area, e.g., 'Termination Rights', 'Liability Exposure'")
    risk: int = Field(description="Risk score for this specific area, 0-100")


class ClauseAnalysis(BaseModel):
    title: str = Field(description="Name or title of the clause")
    risk: Literal["high", "medium", "low"] = Field(description="Risk classification")
    original: str = Field(description="A concise or paraphrased excerpt of the problematic original clause text from the contract")
    rewritten: str = Field(description="A professionally rewritten, legal-grade replacement clause protecting the user")
    reason: str = Field(description="Brief explanation of why this original clause is risky")
    impact: str = Field(description="How the rewritten clause protects the user or reduces risk")


class SimplifiedClause(BaseModel):
    section: str = Field(description="Legal section name or heading from the contract")
    legalText: str = Field(description="Concise excerpt of the legal language from the contract")
    simplified: str = Field(description="Plain English, one-sentence explanation of what this means in simple terms")
    realWorldExample: str = Field(description="A short, concrete real-world scenario showing the practical effect of this clause")


class TimelineItem(BaseModel):
    title: str = Field(description="Milestone or risk event title")
    description: str = Field(description="What happens at this point or what the clause dictates")
    timeframe: str = Field(description="Concise timeframe description, e.g., 'Day 1', 'Month 3', 'Upon Termination'")
    severity: Literal["high", "medium", "low"] = Field(description="Severity of this event/milestone")
    clause: str = Field(description="Concise description or quote of the clause triggering this milestone")
    action: str = Field(description="Recommended action for the user")


class ComplianceCheck(BaseModel):
    regulation: str = Field(description="Regulation name, e.g., GDPR Article 28, HIPAA, or labor standards")
    category: Literal["GDPR", "Privacy", "Labor", "Consumer"] = Field(description="Category of the check")
    status: Literal["pass", "fail", "warning"] = Field(description="Compliance status")
    requirement: str = Field(description="What the regulation requires")
    finding: str = Field(description="What the contract currently says, doesn't say, or is missing in relation to this requirement")
    recommendation: Optional[str] = Field(description="What to fix, add, or negotiate (can be empty if status is pass)")
    reference: Optional[str] = Field(description="URL or reference to the regulation if known")


class ContractAnalysisResult(BaseModel):
    riskScore: int = Field(description="Overall contract risk score, 0-100, where higher is riskier")
    totalClauses: int = Field(description="Estimated number of distinct clauses in the contract")
    riskyClauseCount: int = Field(description="Number of clauses classified as risky or unfavorable")
    complianceIssues: int = Field(description="Number of non-compliant items or warnings found")
    estimatedReadTime: int = Field(description="Estimated minutes to read and fully review the contract")
    summary: str = Field(description="A concise 2-3 sentence executive summary of the contract's primary legal risks and overall posture")
    riskBreakdown: List[RiskBreakdownItem] = Field(description="Breakdown of risks across predefined dimensions")
    topIssues: List[str] = Field(description="Top 3 key issues or major risks found in the contract")
    strongPoints: List[str] = Field(description="Top 3 positive clauses, favorable terms, or strong protections for the user")
    aiVerdict: List[str] = Field(description="AI Verdict: first element is a headline recommendation (e.g. 'Negotiate Before Signing'), followed by 2 specific recommendations")
    clauses: List[ClauseAnalysis] = Field(description="Detailed analysis of 2 to 4 of the riskiest clauses")
    simplifiedClauses: List[SimplifiedClause] = Field(description="Simplified explanation of 2 to 4 key legal concepts in the contract")
    timeline: List[TimelineItem] = Field(description="Chronological timeline of 3 to 5 key milestones or risk events")
    complianceChecks: List[ComplianceCheck] = Field(description="Compliance assessment of 3 to 5 key regulations")


class MetricComparison(BaseModel):
    label: str = Field(description="Name of the compared metric, e.g., liability cap, notice period, payment terms")
    a: str = Field(description="Value for Contract A")
    b: str = Field(description="Value for Contract B")
    winner: Literal["A", "B", "tie"] = Field(description="Which contract has the more favorable value")
    unit: Optional[str] = Field(description="Optional unit, e.g. '%', 'days', '$'")
    aNumeric: Optional[float] = Field(description="Numeric value for A if applicable, else null")
    bNumeric: Optional[float] = Field(description="Numeric value for B if applicable, else null")


class PointComparison(BaseModel):
    category: str = Field(description="Comparison category, e.g., 'Termination', 'Liability'")
    aspect: str = Field(description="Specific aspect compared")
    contractA: str = Field(description="What Contract A specifies")
    contractB: str = Field(description="What Contract B specifies")
    advantage: Literal["A", "B", "tie"] = Field(description="Which contract has the advantage")
    impact: Literal["high", "medium", "low"] = Field(description="Impact of this difference")


class ContractComparisonResult(BaseModel):
    winner: Literal["A", "B", "tie"] = Field(description="Overall winner of the comparison")
    summary: str = Field(description="Concise 2-3 sentence summary explaining why one contract is better or if it is a tie")
    metrics: List[MetricComparison] = Field(description="Measurable metrics side-by-side comparison (5-8 items)")
    points: List[PointComparison] = Field(description="Key qualitative comparison points (5-8 items)")
    hiddenTradeoffs: List[str] = Field(description="3-5 non-obvious tradeoffs or observations")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="ContractCompass API",
    description="AI-powered legal contract analysis using Groq",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Text extraction ───────────────────────────────────────────────────────────

def extract_text_from_pdf(content: bytes) -> str:
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(content))
        return "\n\n".join(
            page.extract_text() or "" for page in reader.pages
        ).strip()
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse PDF: {e}")


def extract_text_from_docx(content: bytes) -> str:
    try:
        import docx
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        doc = docx.Document(tmp_path)
        os.unlink(tmp_path)
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse DOCX: {e}")


def extract_text(file: UploadFile, content: bytes) -> str:
    name = (file.filename or "").lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(content)
    if name.endswith(".docx") or name.endswith(".doc"):
        return extract_text_from_docx(content)
    try:
        return content.decode("utf-8", errors="replace").strip()
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read file: {e}")


# ── Groq AI helper ───────────────────────────────────────────────────────────

def _slim_schema(obj: Any) -> Any:
    """Recursively strip 'description' and 'title' fields from a JSON schema to reduce token count."""
    if isinstance(obj, dict):
        return {k: _slim_schema(v) for k, v in obj.items() if k not in ("description", "title")}
    if isinstance(obj, list):
        return [_slim_schema(i) for i in obj]
    return obj

def _strip_json_fence(raw: str) -> str:
    """Remove markdown code fences from the response."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def _repair_truncated_json(raw: str) -> str:
    """Attempt to repair JSON that was truncated mid-stream."""
    s = raw.rstrip()
    in_string = False
    escape_next = False
    stack = []

    for i, ch in enumerate(s):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
        elif not in_string:
            if ch in ('{', '['):
                stack.append(ch)
            elif ch in ('}', ']'):
                if stack:
                    stack.pop()

    if in_string:
        s += '"'

    s = re.sub(r',\s*$', '', s)

    for opener in reversed(stack):
        if opener == '{':
            s += '}'
        elif opener == '[':
            s += ']'

    return s


def call_ai(prompt: str, schema_class=None) -> Dict[str, Any]:
    """
    Call Groq and parse the JSON response.
    Retries up to 3 times with exponential backoff on rate-limit errors.
    """
    last_error: Exception = RuntimeError("Unknown error")
    raw = ""
    BACKOFF = [5, 15, 45]

    # Build system message with schema if provided
    # Strip descriptions/titles to save ~400-600 tokens on the Groq free tier
    if schema_class is not None:
        slim = _slim_schema(schema_class.model_json_schema())
        schema_json = json.dumps(slim, separators=(",", ":"))
        system_content = (
            "You are an expert legal AI assistant. "
            "Respond with ONE valid JSON object matching this schema exactly. "
            "No markdown, no extra text.\n"
            f"Schema:{schema_json}"
        )
    else:
        system_content = (
            "You are an expert legal AI assistant. "
            "Respond with a single valid JSON object only. No markdown, no extra text."
        )

    for attempt in range(1, 4):
        try:
            response = client.chat.completions.create(
                model=AI_MODEL,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1800,
                temperature=0.2,
            )

            raw = response.choices[0].message.content or ""
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = _strip_json_fence(cleaned)

            # First try: parse as-is
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                # Second try: repair truncated JSON
                repaired = _repair_truncated_json(cleaned)
                print(f"[Attempt {attempt}] JSON repair applied, retrying parse...")
                return json.loads(repaired)

        except json.JSONDecodeError as e:
            last_error = e
            print(f"[Attempt {attempt}/3] JSONDecodeError: {e}\nRaw (first 500):\n{raw[:500]}")
            continue

        except openai.RateLimitError as e:
            print(f"[Attempt {attempt}/3] Rate limit: {e}")
            if attempt < 3:
                wait = BACKOFF[attempt - 1]
                print(f"  Waiting {wait}s before retry {attempt + 1}/3...")
                time.sleep(wait)
                continue
            raise HTTPException(
                status_code=429,
                detail=(
                    "Groq rate limit exceeded. Please wait a moment and try again, "
                    "or check your usage at https://console.groq.com. "
                    f"Original error: {e}"
                ),
            )

        except openai.AuthenticationError as e:
            raise HTTPException(
                status_code=401,
                detail=f"Invalid Groq API key. Check GROQ_API_KEY in Backend/.env. Error: {e}",
            )

        except openai.APIError as e:
            err_str = str(e)
            print(f"[Attempt {attempt}/3] API error: {e}")
            if "429" in err_str or "rate" in err_str.lower():
                if attempt < 3:
                    wait = BACKOFF[attempt - 1]
                    time.sleep(wait)
                    continue
            raise HTTPException(status_code=502, detail=f"Groq API error: {e}")

        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Unexpected error: {e}")

    raise HTTPException(
        status_code=502,
        detail=(
            f"AI returned invalid JSON after 3 attempts: {last_error}. "
            "Please try again with a shorter contract or try later."
        ),
    )


# ── Prompts ───────────────────────────────────────────────────────────────────

ANALYZE_PROMPT = """
You are an expert legal AI assistant. Analyze the following contract and return a valid JSON object matching the requested schema.

CONTRACT TEXT:
---
{contract_text}
---

Your response MUST match the schema.
Important rules:
- Keep all original and rewritten text snippets, legal text extracts, and clause references concise (under 2-3 sentences max).
- The "clauses" array must have 2 to 4 items covering the most critical risky clauses.
- The "simplifiedClauses" array must have 2 to 4 items covering key legal concepts.
- The "timeline" array must have 3 to 5 key milestones or risk events.
- The "complianceChecks" array must have 3 to 5 essential regulatory items (GDPR, Privacy, Labor, Consumer).
- Base ALL analysis strictly on the actual contract text provided. Do not invent details.
"""

COMPARE_PROMPT = """
You are an expert legal AI assistant. Compare these two contracts and return a valid JSON object matching the requested schema.

CONTRACT A — "{name_a}":
---
{text_a}
---

CONTRACT B — "{name_b}":
---
{text_b}
---

Your response MUST match the schema.
Important rules:
- Keep all compared descriptions and aspect statements concise.
- "metrics" should compare 5 to 7 key measurable terms.
- "points" should compare 5 to 7 qualitative aspects (termination, liability, data, pricing, compliance, SLA, etc.).
- "hiddenTradeoffs" should contain 3 to 5 non-obvious tradeoffs or observations.
- Base ALL analysis on the actual contract texts provided. Do not invent details.
"""


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "model": AI_MODEL, "provider": "groq"}


@app.post("/api/analyze")
async def analyze_contract(file: UploadFile = File(...)):
    """
    Analyze a single contract file (PDF, DOCX, or TXT).
    Returns full structured analysis.
    """
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    contract_text = extract_text(file, content)

    if len(contract_text) < 50:
        raise HTTPException(
            status_code=422,
            detail="Contract text is too short to analyze. Please upload a real contract document."
        )

    # Free-tier TPM cap: 6,000 tokens total (input + output).
    # Schema ~300 tokens + system ~50 + contract + 1,800 output = budget ~3,850 for contract.
    # 10,000 chars ≈ 2,500 tokens — safely within limits.
    contract_text = contract_text[:10_000]

    prompt = ANALYZE_PROMPT.format(contract_text=contract_text)
    result = call_ai(prompt, schema_class=ContractAnalysisResult)

    result.setdefault("riskScore", 50)
    result.setdefault("totalClauses", 0)
    result.setdefault("riskyClauseCount", 0)
    result.setdefault("complianceIssues", 0)
    result.setdefault("estimatedReadTime", 5)
    result.setdefault("summary", "")
    result.setdefault("riskBreakdown", [])
    result.setdefault("topIssues", [])
    result.setdefault("strongPoints", [])
    result.setdefault("aiVerdict", [])
    result.setdefault("clauses", [])
    result.setdefault("simplifiedClauses", [])
    result.setdefault("timeline", [])
    result.setdefault("complianceChecks", [])

    return result


@app.post("/api/compare")
async def compare_contracts(
    file_a: UploadFile = File(...),
    file_b: UploadFile = File(...),
):
    """
    Compare two contract files side by side.
    Returns structured comparison with winner determination.
    """
    content_a = await file_a.read()
    content_b = await file_b.read()

    if not content_a or not content_b:
        raise HTTPException(status_code=400, detail="Both contract files must be non-empty.")

    text_a = extract_text(file_a, content_a)
    text_b = extract_text(file_b, content_b)

    text_a = text_a[:5_000]
    text_b = text_b[:5_000]

    prompt = COMPARE_PROMPT.format(
        name_a=file_a.filename or "Contract A",
        text_a=text_a,
        name_b=file_b.filename or "Contract B",
        text_b=text_b,
    )
    result = call_ai(prompt, schema_class=ContractComparisonResult)

    result.setdefault("winner", "tie")
    result.setdefault("summary", "")
    result.setdefault("metrics", [])
    result.setdefault("points", [])
    result.setdefault("hiddenTradeoffs", [])

    return result


# ── Serve built React frontend ────────────────────────────────────────────────
_STATIC_DIR = Path(__file__).parent / "static"

if _STATIC_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=_STATIC_DIR / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        """Catch-all: serve index.html for any non-API route (SPA routing)."""
        file_path = _STATIC_DIR / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(_STATIC_DIR / "index.html")
