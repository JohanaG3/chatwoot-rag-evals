"""Run the eval set with one configuration and score every answer with the judge. Resumable.

Usage: uv run python evaluate.py <chunk_words> <prompt> [top_k]
  e.g. uv run python evaluate.py 200 v2-revised
Output: results/<words>_<prompt>_k<top_k>.json (+ traces and scores in Langfuse if configured).
Also prints agreement with the human labels in data/human_labels.json.
"""
import json
import statistics
import sys

from config import JUDGE_MODEL, ROOT, lf
from judge import judge
from rag import answer

WORDS, PROMPT = int(sys.argv[1]), sys.argv[2]
TOP_K = int(sys.argv[3]) if len(sys.argv) > 3 else 5
run = f"{WORDS}_{PROMPT}_k{TOP_K}"
out = ROOT / f"results/{run}.json"
out.parent.mkdir(exist_ok=True)
rows = json.loads(out.read_text()) if out.exists() else []
done = {r["id"] for r in rows}

for q in json.loads((ROOT / "data/questions.json").read_text()):
    if q["id"] in done:
        continue
    r = answer(q["question"], words=WORDS, prompt=PROMPT, top_k=TOP_K, tags=["eval"], session=run)
    s = judge(q["question"], r["context"], r["answer"], trace_id=r["trace_id"])
    rows.append({**q, "answer": r["answer"], "retrieved": [c["title"] for c in r["chunks"]],
                 "faithfulness": s.fidelidad, "relevance": s.relevancia, "judge_reason": s.razon,
                 "judge_model": JUDGE_MODEL})
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print(f"Q{q['id']}: F{s.fidelidad} R{s.relevancia} · {r['answer'][:70]}", flush=True)

if lf:
    lf.flush()
print(f"\n{run}: faithfulness {statistics.mean(r['faithfulness'] for r in rows):.2f} · "
      f"relevance {statistics.mean(r['relevance'] for r in rows):.2f}")

by_id = {r["id"]: r for r in rows}
labels = [h for h in json.loads((ROOT / "data/human_labels.json").read_text()) if h["chunk_words"] == WORDS]
if labels and PROMPT == "v1-strict":
    agree = sum((h["faithfulness"] == by_id[h["id"]]["faithfulness"]) + (h["relevance"] == by_id[h["id"]]["relevance"])
                for h in labels if h["id"] in by_id)
    print(f"judge vs human labels: {agree}/{2 * len(labels)} scores agree")
