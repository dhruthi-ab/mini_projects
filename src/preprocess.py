"""Patent text cleaning: lowercase, strip markup/claim numbering/figure refs,
tokenise, remove stopwords + patent boilerplate, light stemming."""
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

BOILERPLATE = {
    "claim", "claims", "claimed", "according", "wherein", "whereby", "comprising", "comprises",
    "comprise", "comprised", "thereof", "therein", "thereby", "herein", "said", "invention",
    "embodiment", "embodiments", "present", "fig", "figs", "figure", "figures", "preferably",
    "method", "apparatus", "device", "system", "including", "includes", "include", "having", "further",
}
STOP = set(ENGLISH_STOP_WORDS) | BOILERPLATE

try:
    from nltk.stem import PorterStemmer
    _stem = PorterStemmer().stem
except Exception:  # light suffix stripper fallback
    def _stem(w):
        for suf in ("ization", "ations", "ation", "ingly", "ings", "ing", "edly", "ed", "ies", "es", "s", "ly"):
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                return w[: -len(suf)] + ("y" if suf == "ies" else "")
        return w

_TAG = re.compile(r"<[^>]+>")
_CLAIMNUM = re.compile(r"(?<!\w)\d+\s*[\.\)]\s+(?=[A-Za-z])")   # "1. A device" -> "A device"
_REFNUM = re.compile(r"\(\s*\d+[a-z]?\s*\)|\b\d+[a-z]?\b")        # reference numerals / bare numbers
_TOKEN = re.compile(r"[a-z][a-z\-]{1,}")

def clean_text(text):
    text = _TAG.sub(" ", str(text or ""))
    text = text.lower()
    text = _CLAIMNUM.sub(" ", text)
    text = _REFNUM.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()

def tokenize(text, stem=True):
    toks = [t.strip("-") for t in _TOKEN.findall(clean_text(text))]
    toks = [t for t in toks if len(t) > 1 and t not in STOP]
    return [_stem(t) for t in toks] if stem else toks

def build_doc_text(rec, use_claims=True, max_claim_chars=3000):
    parts = [rec.get("title", ""), rec.get("abstract", "")]
    if use_claims:
        parts.append(rec.get("claims", "")[:max_claim_chars])
    return " . ".join(p for p in parts if p)
