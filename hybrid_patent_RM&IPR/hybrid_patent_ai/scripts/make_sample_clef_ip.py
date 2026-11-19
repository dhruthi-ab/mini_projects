"""Create a SYNTHETIC CLEF-IP-style sample (XML docs, topics, qrels) to smoke-test the pipeline.
NOT real patents. Results on it say nothing about real performance."""
import argparse, os, random
from xml.sax.saxutils import escape

def word(rng):
    c, v = "bcdfghklmnprstvz", "aeiou"
    return "".join(rng.choice(c) + rng.choice(v) for _ in range(rng.randint(3, 4)))

def xml(pid, title, abstract, claims):
    cl = "".join(f'<claim num="{i}"><claim-text>{i}. {escape(c)}</claim-text></claim>' for i, c in enumerate(claims, 1))
    return (f'<?xml version="1.0"?><patent-document ucid="{pid}"><bibliographic-data>'
            f'<invention-title lang="EN">{escape(title)}</invention-title></bibliographic-data>'
            f'<abstract lang="EN"><p>{escape(abstract)}</p></abstract><claims lang="EN">{cl}</claims></patent-document>')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sample_clef_ip")
    ap.add_argument("--families", type=int, default=60)
    ap.add_argument("--rel-per-topic", type=int, default=6)
    ap.add_argument("--distractors", type=int, default=800)
    a = ap.parse_args()
    rng = random.Random(1)
    for d in ("docs", "topics"):
        os.makedirs(f"{a.out}/{d}", exist_ok=True)
    domains = [[word(rng) for _ in range(40)] for _ in range(12)]
    filler = [word(rng) for _ in range(300)]
    def text(core, dom, n=45):
        ws = [rng.choice(core) for _ in range(n // 2)] + [rng.choice(dom) for _ in range(n // 4)] + \
             [rng.choice(filler) for _ in range(n // 4)]
        rng.shuffle(ws); return " ".join(ws)
    qrels, n = [], 0
    for f in range(a.families):
        dom = domains[f % len(domains)]
        slots = [(word(rng), word(rng)) for _ in range(8)]          # synonym pairs
        def core(side): return [s[side if rng.random() < 0.7 else 1 - side] for s in slots]
        tid = f"EP-T{f:04d}-A1"
        c = core(0)
        open(f"{a.out}/topics/{tid}.xml", "w").write(xml(tid, " ".join(c[:3]), text(c, dom), [text(c, dom, 30)]))
        for j in range(a.rel_per_topic):
            did = f"EP-{f:04d}{j:02d}-B1"; c = core(1)
            open(f"{a.out}/docs/{did}.xml", "w").write(xml(did, " ".join(c[:3]), text(c, dom), [text(c, dom, 30)]))
            qrels.append(f"{tid} 0 {did} 1")
    for i in range(a.distractors):
        dom = rng.choice(domains); c = [word(rng) for _ in range(6)]
        did = f"EP-D{i:05d}-B1"
        open(f"{a.out}/docs/{did}.xml", "w").write(xml(did, " ".join(c[:3]), text(c, dom), [text(c, dom, 30)]))
    open(f"{a.out}/qrels.txt", "w").write("\n".join(qrels) + "\n")
    print("sample written to", a.out)

if __name__ == "__main__":
    main()
