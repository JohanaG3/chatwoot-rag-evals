"""Step 2: chunk the corpus, embed with Gemini, store in Chroma. Resumable.

Usage: uv run python index.py 200     (or 1000)  -> collection chatwoot_<words>
Free tier limits that shaped this: 30k tokens/minute and 1,000 embedding requests/day
(each chunk counts as one request). Batches stay under the per-minute budget; a rerun
skips chunks already stored, so a run cut off by the daily quota resumes the next day.
"""
import json
import sys
import time

import chromadb
from google.genai import types

from config import CHROMA, CORPUS, EMBED_DIM, EMBED_MODEL, gemini, retry

WORDS = int(sys.argv[1])
OVERLAP = WORDS // 7          # ~15 % overlap so an idea is not cut exactly at a border
TOKEN_BUDGET = 22_000         # per batch, estimated at ~1.5 tokens/word; one batch per minute
MAX_BATCH = 60


def chunk(text):
    """Pack whole lines (paragraphs) up to WORDS; the next chunk repeats the last OVERLAP words."""
    chunks, cur = [], []
    for line in text.split("\n"):
        cur += line.split() + ["\n"]
        if len(cur) >= WORDS:
            chunks.append(cur)
            cur = cur[-OVERLAP:]
    if len([w for w in cur if w != "\n"]) > OVERLAP or not chunks:
        chunks.append(cur)
    return [" ".join(c).replace(" \n ", "\n").strip() for c in chunks]


ids, docs, metas = [], [], []
for f in sorted(CORPUS.glob("*.json")):
    d = json.loads(f.read_text())
    for i, c in enumerate(chunk(d["text"])):
        ids.append(f"{f.stem}#{i}")
        docs.append(f"{d['title']}\n{c}")            # the title goes inside: context for the chunk
        metas.append({"url": d["url"], "title": d["title"], "words": len(c.split())})
print(f"{len(docs)} chunks of ~{WORDS} words")

col = chromadb.PersistentClient(path=str(CHROMA)).get_or_create_collection(
    f"chatwoot_{WORDS}", metadata={"hnsw:space": "cosine"})
done = set(col.get(include=[])["ids"])
todo = [k for k, x in enumerate(ids) if x not in done]
ids, docs, metas = [ids[k] for k in todo], [docs[k] for k in todo], [metas[k] for k in todo]
print(f"already indexed {len(done)}, pending {len(ids)}")

i = 0
while i < len(docs):
    j, tokens = i, 0
    while j < len(docs) and (j == i or tokens + len(docs[j].split()) * 1.5 <= TOKEN_BUDGET) and j - i < MAX_BATCH:
        tokens += len(docs[j].split()) * 1.5
        j += 1
    start = time.time()
    vectors = retry(lambda: [e.values for e in gemini.models.embed_content(
        model=EMBED_MODEL, contents=docs[i:j],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT", output_dimensionality=EMBED_DIM)).embeddings])
    col.add(ids=ids[i:j], documents=docs[i:j], metadatas=metas[i:j], embeddings=vectors)
    print(f"  {j}/{len(docs)}", flush=True)
    i = j
    if i < len(docs):
        time.sleep(max(0, 61 - (time.time() - start)))
print(f"done: chatwoot_{WORDS} has {col.count()} chunks")
