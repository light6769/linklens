"""LinkLens - a local fake-news / claim checker.

Pipeline:
  1. Input is a headline, claim, article text, or a URL (article is fetched).
  2. The local LLM (Mistral 7B via Ollama) extracts a short search query.
  3. Evidence is gathered from Wikipedia and DuckDuckGo (grounding).
  4. The LLM judges the claim against that evidence and returns JSON.
  5. A regex layer scans for classic misinformation / clickbait patterns and
     can override or downgrade the model's verdict.

Run:  python app.py   ->   http://127.0.0.1:5000
Needs Ollama running locally with the model pulled:  ollama pull mistral
"""
import ipaddress
import json
import re
import socket
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, request

try:
    from ddgs import DDGS
except ImportError:  # older package name
    from duckduckgo_search import DDGS

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "mistral"
HEADERS = {"User-Agent": "LinkLens/1.0 (educational project)"}
MAX_TEXT = 3000  # chars of article text sent to the model

app = Flask(__name__)

# --------------------------------------------------------------------------
# Regex override layer
# --------------------------------------------------------------------------
# Hard patterns: strong misinformation markers -> force LIKELY FAKE
HARD_PATTERNS = {
    "miracle cure claim": r"miracle cure",
    "'doctors hate' clickbait": r"doctors (hate|don'?t want you)",
    "'they don't want you to know'": r"they don'?t want you to know",
    "'share before it's deleted'": r"share (this )?before (it'?s|it is) (deleted|removed|banned)",
    "'100% guaranteed' claim": r"100% (guaranteed|proven|cure)",
    "'media is hiding' claim": r"mainstream media (is )?(hiding|won'?t)",
}
# Soft patterns: sensational language -> only downgrade a 'real' verdict
SOFT_PATTERNS = {
    "'shocking'": r"\bshocking\b",
    "'you won't believe'": r"you won'?t believe",
    "'exposed'": r"\bexposed\b",
    "'secret'": r"\bsecrets?\b",
    "'wake up'": r"\bwake up\b",
    "repeated exclamation marks": r"!{2,}",
}


def regex_flags(text):
    hard = [name for name, p in HARD_PATTERNS.items() if re.search(p, text, re.I)]
    soft = [name for name, p in SOFT_PATTERNS.items() if re.search(p, text, re.I)]
    letters = [c for c in text if c.isalpha()]
    if len(letters) > 40 and sum(c.isupper() for c in letters) / len(letters) > 0.3:
        soft.append("excessive ALL CAPS")
    return hard, soft


def apply_overrides(result, hard, soft):
    result["regex_hard"], result["regex_soft"] = hard, soft
    result["overridden"] = False
    if hard:
        result.update(verdict="LIKELY FAKE",
                      confidence=max(result["confidence"], 85),
                      overridden=True)
    elif len(soft) >= 3 and result["verdict"] == "LIKELY REAL":
        result.update(verdict="UNVERIFIED",
                      confidence=min(result["confidence"], 50),
                      overridden=True)
    return result


# --------------------------------------------------------------------------
# LLM (Ollama)
# --------------------------------------------------------------------------
def ask_llm(system, user, as_json=False, max_tokens=400):
    payload = {
        "model": MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {"temperature": 0, "num_predict": max_tokens},
    }
    if as_json:
        payload["format"] = "json"
    r = requests.post(OLLAMA_URL, json=payload, timeout=180)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


QUERY_SYSTEM = (
    "Extract the single main factual claim from the text and turn it into a "
    "web search query of at most 10 words. Drop negations (not, no, never, "
    "isn't, doesn't) and question words; keep only the topic's key terms. "
    "Reply with the query only, no quotes."
)

JUDGE_SYSTEM = (
    "You are a careful fact-checking assistant. You must decide whether THE "
    "CLAIM SENTENCE, exactly as written, is TRUE or FALSE. Never judge the "
    "evidence instead of the claim, and never flip the claim's polarity: "
    'for the claim "The Earth is flat", FALSE is correct, not TRUE. '
    "- TRUE = the claim sentence, as written, is correct. "
    "- FALSE = the claim sentence, as written, is wrong (including when the "
    "truth is the opposite of what it says). "
    "- UNVERIFIED = the evidence is unrelated or insufficient; never guess. "
    "Set evidence_aligns to how the evidence relates to the claim sentence AS "
    "WRITTEN: supports = confirms it, contradicts = shows the opposite, "
    "unrelated = neither. Your verdict must match evidence_aligns exactly "
    "(supports -> TRUE, contradicts -> FALSE, unrelated -> UNVERIFIED). "
    'Reply with JSON only: {"verdict": "TRUE" | "FALSE" | "UNVERIFIED", '
    '"evidence_aligns": "supports" | "contradicts" | "unrelated", '
    '"confidence": <0-100>, "reasoning": "<2-3 sentences>", '
    '"red_flags": ["<short phrase>", ...]}'
)


def parse_verdict(raw):
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        try:
            data = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            data = {}
    label = str(data.get("verdict", "")).upper()
    align = str(data.get("evidence_aligns", "")).lower()
    if "FAKE" in label or "FALSE" in label:
        verdict = "LIKELY FAKE"
    elif "REAL" in label or "TRUE" in label:
        verdict = "LIKELY REAL"
    else:
        verdict = "UNVERIFIED"

    # Consistency guardrail: derive the verdict deterministically from the model's
    # own evidence_aligns reading (supports -> REAL, contradicts -> FAKE,
    # unrelated -> UNVERIFIED). 7B models commonly flip polarity on negated
    # claims or emit a verdict token that contradicts their own reasoning
    # (e.g. "The Earth is flat" -> REAL, or "The Moon is not made of cheese"
    # -> FALSE while the reasoning says it is true). Trust the alignment, not
    # the verdict token.
    flipped = False
    if align.startswith("support") and verdict != "LIKELY REAL":
        verdict, flipped = "LIKELY REAL", True
    elif align.startswith("contrad") and verdict != "LIKELY FAKE":
        verdict, flipped = "LIKELY FAKE", True
    elif align.startswith("unrelat") and verdict != "UNVERIFIED":
        verdict, flipped = "UNVERIFIED", True

    try:
        confidence = max(0, min(100, int(float(data.get("confidence", 50)))))
    except (TypeError, ValueError):
        confidence = 50
    if flipped:
        if verdict == "UNVERIFIED":
            confidence = min(confidence, 50)
        else:
            confidence = max(confidence, 85)
    flags = data.get("red_flags")
    return {
        "verdict": verdict,
        "confidence": confidence,
        "reasoning": str(data.get("reasoning", ""))[:800],
        "red_flags": [str(f) for f in flags][:6] if isinstance(flags, list) else [],
        "evidence_aligns": align or "unknown",
        "flipped": flipped,
    }


# --------------------------------------------------------------------------
# Fetching articles (with basic SSRF protection - this is a local tool,
# don't expose it to the public internet as-is)
# --------------------------------------------------------------------------
def is_safe_url(url):
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        return False
    try:
        for info in socket.getaddrinfo(p.hostname, None):
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False
    except (socket.gaierror, ValueError):
        return False
    return True


def fetch_article(url):
    for _ in range(4):  # follow a few redirects manually, re-checking each hop
        if not is_safe_url(url):
            raise ValueError("blocked or invalid URL")
        r = requests.get(url, headers=HEADERS, timeout=10, allow_redirects=False)
        if r.is_redirect and r.headers.get("Location"):
            url = urljoin(url, r.headers["Location"])
            continue
        r.raise_for_status()
        break
    else:
        raise ValueError("too many redirects")
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "aside"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else ""
    text = " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))
    return title, text[:MAX_TEXT]


# --------------------------------------------------------------------------
# Evidence retrieval (grounding)
# --------------------------------------------------------------------------
def search_wikipedia(query, n=2):
    try:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query", "generator": "search", "gsrsearch": query,
                "gsrlimit": n, "prop": "extracts", "exintro": 1,
                "explaintext": 1, "exchars": 500, "format": "json",
            },
            headers=HEADERS, timeout=10,
        )
        pages = r.json().get("query", {}).get("pages", {}).values()
        return [
            {
                "source": "Wikipedia",
                "title": p["title"],
                "snippet": p.get("extract", ""),
                "url": "https://en.wikipedia.org/wiki/" + p["title"].replace(" ", "_"),
            }
            for p in sorted(pages, key=lambda p: p.get("index", 99))
        ]
    except (requests.RequestException, ValueError, KeyError):
        return []


def search_web(query, n=4):
    try:
        with DDGS() as ddgs:
            return [
                {
                    "source": "Web",
                    "title": r.get("title", ""),
                    "snippet": r.get("body", "")[:300],
                    "url": r.get("href", ""),
                }
                for r in ddgs.text(query, max_results=n)
            ]
    except Exception:  # the DDG client raises several different error types
        return []


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
@app.post("/analyze")
def analyze():
    data = request.get_json(silent=True) or {}
    user_input = (data.get("input") or "").strip()
    if not user_input:
        return jsonify(error="Paste a headline, claim, article text, or link."), 400

    title, text = "", user_input
    if re.match(r"https?://", user_input):
        try:
            title, text = fetch_article(user_input)
        except (requests.RequestException, ValueError) as e:
            return jsonify(error=f"Couldn't fetch that link: {e}"), 400
    text = text[:MAX_TEXT]
    if not text.strip():
        return jsonify(error="No readable text found."), 400

    try:
        query = ask_llm(QUERY_SYSTEM, f"{title}\n{text[:800]}", max_tokens=30)
        query = query.split("\n")[0].strip(' "')[:120] or title or text[:100]

        evidence = search_wikipedia(query) + search_web(query)
        evidence_text = "\n".join(
            f"[{i + 1}] ({e['source']}) {e['title']}: {e['snippet']}"
            for i, e in enumerate(evidence)
        ) or "(no evidence found)"

        prompt = (
            f"CLAIM / ARTICLE:\n{title}\n{text}\n\n"
            f"EVIDENCE:\n{evidence_text}\n\nGive your verdict as JSON."
        )
        result = parse_verdict(ask_llm(JUDGE_SYSTEM, prompt, as_json=True))
    except requests.ConnectionError:
        return jsonify(error="Can't reach Ollama. Run `ollama serve` and "
                             f"`ollama pull {MODEL}`."), 503
    except requests.RequestException as e:
        return jsonify(error=f"Model request failed: {e}"), 502

    hard, soft = regex_flags(f"{title} {text}")
    result = apply_overrides(result, hard, soft)
    result["query"] = query
    result["sources"] = evidence
    return jsonify(result)


# --------------------------------------------------------------------------
# Front end (single page)
# --------------------------------------------------------------------------
PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LinkLens</title>
<style>
  :root{
    --bg:#070912;
    --surface:#0d1220;
    --surface-2:#11182a;
    --border:#202b42;
    --border-hover:#33415f;

    --text:#f4f7ff;
    --muted:#8995ab;

    --primary:#8b5cf6;
    --primary-light:#a78bfa;
    --cyan:#22d3ee;

    --real:#34d399;
    --fake:#fb7185;
    --unverified:#fbbf24;

    --tag:#182238;
    --link:#67e8f9;
  }

  *{
    box-sizing:border-box;
  }

  body{
    font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
    background:
      radial-gradient(
        circle at 50% -15%,
        rgba(139,92,246,.16),
        transparent 45%
      ),
      var(--bg);
    color:var(--text);
    max-width:760px;
    margin:40px auto;
    padding:0 16px;
    line-height:1.5;
  }

  h1{
    margin-bottom:4px;
    font-size:2rem;
    font-weight:750;
    letter-spacing:-.6px;
    background:linear-gradient(
      90deg,
      #ffffff,
      #c4b5fd,
      #67e8f9
    );
    -webkit-background-clip:text;
    background-clip:text;
    color:transparent;
  }

  .sub{
    color:var(--muted);
    margin-top:0;
  }

  textarea{
    width:100%;
    height:130px;
    background:rgba(13,18,32,.9);
    color:var(--text);
    border:1px solid var(--border);
    border-radius:12px;
    padding:13px;
    font:inherit;
    resize:vertical;
    outline:none;
    transition:
      border-color .2s ease,
      box-shadow .2s ease,
      background .2s ease;
  }

  textarea::placeholder{
    color:#5f6b80;
  }

  textarea:hover{
    border-color:var(--border-hover);
  }

  textarea:focus{
    border-color:var(--primary);
    background:#0f1626;
    box-shadow:
      0 0 0 3px rgba(139,92,246,.12),
      0 0 30px rgba(139,92,246,.06);
  }

  button{
    margin-top:10px;
    background:linear-gradient(
      135deg,
      #7c3aed,
      #8b5cf6 55%,
      #06b6d4
    );
    color:#fff;
    border:0;
    border-radius:9px;
    padding:10px 22px;
    font-size:1rem;
    font-weight:650;
    cursor:pointer;
    box-shadow:
      0 6px 22px rgba(124,58,237,.22);
    transition:
      transform .15s ease,
      box-shadow .2s ease,
      filter .2s ease;
  }

  button:hover{
    transform:translateY(-1px);
    filter:brightness(1.08);
    box-shadow:
      0 8px 28px rgba(124,58,237,.32);
  }

  button:active{
    transform:translateY(0);
  }

  button:disabled{
    opacity:.5;
    cursor:wait;
    transform:none;
    box-shadow:none;
  }

  .card{
    background:
      linear-gradient(
        145deg,
        rgba(17,24,42,.96),
        rgba(13,18,32,.96)
      );
    border:1px solid var(--border);
    border-radius:14px;
    padding:18px;
    margin-top:20px;
    box-shadow:
      0 16px 45px rgba(0,0,0,.28);
  }

  .verdict{
    font-size:1.5rem;
    font-weight:750;
    letter-spacing:-.3px;
  }

  .REAL{
    color:var(--real);
    text-shadow:
      0 0 18px rgba(52,211,153,.18);
  }

  .FAKE{
    color:var(--fake);
    text-shadow:
      0 0 18px rgba(251,113,133,.18);
  }

  .UNV{
    color:var(--unverified);
    text-shadow:
      0 0 18px rgba(251,191,36,.18);
  }

  .tag{
    display:inline-block;
    background:var(--tag);
    border:1px solid #293753;
    color:#b8c5da;
    border-radius:999px;
    padding:3px 10px;
    margin:3px;
    font-size:.82rem;
  }

  a{
    color:var(--link);
    text-decoration:none;
  }

  a:hover{
    color:#a5f3fc;
    text-decoration:underline;
  }

  li{
    margin:8px 0;
  }

  .muted{
    color:var(--muted);
    font-size:.9rem;
  }

  strong{
    color:#dce5f5;
  }
</style></head><body>
<h1>LinkLens</h1>
<p class="sub">Paste a headline, claim, article text, or a link. Runs locally with Mistral 7B.</p>
<textarea id="input" placeholder="e.g. https://example.com/article  or  'Scientists confirm the moon is hollow'"></textarea>
<button id="go">Analyze</button>
<div id="out"></div>
<script>
const out = document.getElementById("out"), btn = document.getElementById("go");
function el(tag, text, cls){const e=document.createElement(tag); if(text!==undefined) e.textContent=text; if(cls) e.className=cls; return e;}
btn.onclick = async () => {
  const input = document.getElementById("input").value;
  btn.disabled = true; out.replaceChildren(el("p","Analyzing... (first run can take a while)","muted"));
  try {
    const res = await fetch("/analyze", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({input})});
    const d = await res.json();
    if (!res.ok) { out.replaceChildren(el("div", d.error || "Something went wrong", "card")); return; }
    const card = el("div", undefined, "card");
    const cls = d.verdict === "LIKELY REAL" ? "REAL" : d.verdict === "LIKELY FAKE" ? "FAKE" : "UNV";
    card.append(el("div", d.verdict + " (" + d.confidence + "%)", "verdict " + cls));
    if (d.overridden) card.append(el("p", "Verdict adjusted by the pattern-check layer.", "muted"));
    if (d.flipped) card.append(el("p", "Verdict corrected: the model's own evidence reading disagreed with its verdict.", "muted"));
    card.append(el("p", d.reasoning));
    const flags = [...d.red_flags, ...d.regex_hard, ...d.regex_soft];
    if (flags.length) { const p = el("p"); flags.forEach(f => p.append(el("span", f, "tag"))); card.append(el("strong","Red flags"), p); }
    card.append(el("p", "Search query used: " + d.query, "muted"));
    if (d.sources.length) {
      card.append(el("strong", "Evidence"));
      const ul = el("ul");
      d.sources.forEach(s => { const li = el("li"); const a = el("a", "[" + s.source + "] " + s.title);
        if (/^https?:\\/\\//.test(s.url)) { a.href = s.url; a.target = "_blank"; a.rel = "noopener"; }
        li.append(a, el("div", s.snippet, "muted")); ul.append(li); });
      card.append(ul);
    }
    out.replaceChildren(card);
  } catch (e) { out.replaceChildren(el("div", "Request failed: " + e, "card")); }
  finally { btn.disabled = false; }
};
</script></body></html>
"""


@app.get("/")
def index():
    return PAGE


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)