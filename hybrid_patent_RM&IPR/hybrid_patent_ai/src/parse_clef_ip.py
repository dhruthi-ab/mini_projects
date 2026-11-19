"""Parse CLEF-IP XML patent documents into JSONL (id, title, abstract, claims)
and load topics / qrels.

CLEF-IP documents are EPO XML ("patent-document") with <invention-title lang=..>,
<abstract lang=..><p>, <claims lang=..><claim><claim-text>. Only English text is kept.
Parsing is namespace/format tolerant (matches on local tag names).
"""
import argparse, json, os, re
from lxml import etree

def _local(tag):
    return tag.split("}")[-1] if isinstance(tag, str) else ""

def _text(el):
    return re.sub(r"\s+", " ", " ".join(el.itertext())).strip()

def _lang_ok(el, lang="en"):
    l = (el.get("lang") or "").lower()
    return l in ("", lang)

def parse_patent_xml(path, lang="en"):
    try:
        root = etree.parse(path, etree.XMLParser(recover=True, huge_tree=True)).getroot()
    except Exception:
        return None
    if root is None:
        return None
    pid = root.get("ucid") or root.get("doc-id") or os.path.splitext(os.path.basename(path))[0]
    title, abstract, claims = "", [], []
    for el in root.iter():
        t = _local(el.tag)
        if t == "invention-title" and _lang_ok(el, lang) and not title:
            title = _text(el)
        elif t == "abstract" and _lang_ok(el, lang):
            abstract.append(_text(el))
        elif t == "claims" and _lang_ok(el, lang):
            subs = [c for c in el if _local(c.tag) == "claim"]
            claims += [_text(c) for c in subs] if subs else [_text(el)]
    return {"patent_id": pid, "title": title, "abstract": " ".join(abstract),
            "claims": " ".join(claims)}

def iter_xml(root_dir):
    for d, _, fs in os.walk(root_dir):
        for f in sorted(fs):
            if f.lower().endswith(".xml"):
                yield os.path.join(d, f)

def parse_dir(in_dir, out_jsonl, require_text=True):
    n = 0
    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as out:
        for p in iter_xml(in_dir):
            rec = parse_patent_xml(p)
            if not rec or (require_text and not (rec["title"] or rec["abstract"] or rec["claims"])):
                continue
            out.write(json.dumps(rec, ensure_ascii=False) + "\n"); n += 1
    return n

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def load_qrels(path):
    """Accepts TREC lines 'topic 0 doc rel' or 'topic doc [rel]'. Returns {topic: {doc: rel}}."""
    q = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) < 2 or line.startswith("#"):
                continue
            if len(p) >= 4: t, d, r = p[0], p[2], p[3]
            elif len(p) == 3: t, d, r = p[0], p[1], p[2]
            else: t, d, r = p[0], p[1], "1"
            try: r = int(float(r))
            except ValueError: r = 1
            if r > 0:
                q.setdefault(t, {})[d] = r
    return q

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CLEF-IP XML -> JSONL")
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print("wrote", parse_dir(a.in_dir, a.out), "records")
