# chatwoot-rag-evals

A retrieval-augmented assistant over [Chatwoot](https://www.chatwoot.com)'s public documentation, built to answer one practical question: **how do you know a RAG system is actually good, and how do you prove a change made it better?**

It covers the whole loop: a corpus, chunking, embeddings, a vector store and grounded answers with citations. On top of that sit an eval set written from real user questions, a rubric, an LLM judge calibrated against human labels, tracing in Langfuse and a regression test on a prompt change.

> The assistant, the eval questions and the rubric are in **Spanish**, the language of the users it was designed for. Code and docs are in English.

## Pipeline

```
Chatwoot docs ──► chunks (200 / 1000 words) ──► Gemini embeddings ──► Chroma
                                                                        │
question ──► embed ──► top-5 chunks ──► answer with citations [n] ◄─────┘
                                              │
                         LLM judge (rubric) ──► faithfulness, relevance ──► Langfuse
```

| Piece | Choice |
|---|---|
| Corpus | 154 user-guide articles + 89 developer-doc pages (~137k words) |
| Chunking | paragraph-aware, ~200 or ~1000 words, ~15 % overlap, title prepended to each chunk |
| Embeddings | `gemini-embedding-001`, 768 dims, separate document/query task types |
| Vector store | Chroma (local, cosine) |
| Answer model | `gemini-3.5-flash-lite`, temperature 0, must cite chunk numbers |
| Judge | `gemini-3.5-flash-lite`, temperature 0, structured JSON output, fixed across runs |
| Observability | Langfuse: one trace per question (`retrieve` → `answer`), judge scores attached to the trace |

## The eval set

10 questions a Chatwoot admin or agent would really ask ([data/questions.json](data/questions.json)): easy ones, ones whose answer is spread across several pages, technical ones, and **one trap**. "Can I pay in bolívares?" has no answer in the docs, and the right behaviour is to say so rather than invent one.

Each question has the answer I *expected* before reading the docs. That turned out to matter (see what I learned below).

## Rubric and calibration

The rubric ([data/rubric.md](data/rubric.md)) scores two things separately:

- **Faithfulness**: is everything said backed by the retrieved chunks? An honest "I can't find it" is always faithful.
- **Relevance**: does it help with what was asked? "It's not there" scores 5 when the information really isn't there, and 1 when it was in the chunks.

Before trusting the judge, I scored 5 answers by hand ([data/human_labels.json](data/human_labels.json)). With the judge fixed to one model, it agreed with me on **10 of 10 scores**.

## Results

**Chunk size** (prompt `v1-strict`): neither size wins everywhere. 200-word chunks were better on pinpoint facts (architecture, macOS setup); 1000-word chunks were better when the answer is a procedure or needs several steps together (website widget, reports, grouping conversations).

**Regression test: one change, measured.** The strict prompt ("if it's not in the chunks, answer exactly *I can't find it*") never invented anything, but it also gave up on answerable questions. The revised prompt asks the model to weigh every chunk before giving up, while still refusing to invent.

| Run | Faithfulness | Relevance |
|---|---|---|
| 200 words · v1-strict | 5.0 | 4.6 |
| 200 words · **v2-revised** | 5.0 | **5.0** |
| 1000 words · v1-strict | 5.0 | 4.6 |
| 1000 words · **v2-revised** | 5.0 | **5.0** |

Relevance went up with no loss in faithfulness, and the trap question still got an honest "not in the documentation". Per-question answers, retrieved chunks and the judge's reasoning are in [results/](results/).

## What I learned

- **Retrieval failure ≠ generation failure.** In one case both chunk sizes retrieved the *identical* right chunk (the article fits in 200 words). One run still answered "I can't find it". The difference was the other four chunks around it and an ambiguous word in the question: a generation problem, not a retrieval one.
- **A judge that only sees the retrieved chunks can't catch retrieval misses.** If search didn't bring the answer, "it's not there" looks correct to the judge. Catching that needs a reference answer and a *context recall* metric.
- **Correct is not complete.** The revised prompt's answer to "how do we handle 10k+ customers?" got a 5 from the judge. I'd give it 3–4, because it leaves out automatic assignment. Next criterion to add: completeness against the expected answer.
- **The judge has variance too.** Swapping the judge model flipped one score from 1 to 5 under the same rubric. Calibrate against human labels, and never change the judge between runs you compare.
- **An eval set where everything scores 5 has stopped being useful.** The next step is harder questions, so that versions can be told apart.
- **Prompts get copied literally.** The revised prompt says *"di que no lo encuentras"* ("say that *you* can't find it"), and one answer came back as *"No lo encuentras"*. The prompts are kept verbatim as they were run.
- **Provider limits are part of the design.** Free tier: 30k tokens/min and 1,000 embedding requests/day. Indexing is batched under the per-minute budget and resumes where it stopped. Retries handle both 429 (quota) and 503 (overloaded). One model was retired mid-project and another had a 20 requests/day cap.

## Run it

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). A Gemini API key (free tier works); Langfuse keys are optional.

```bash
export GEMINI_API_KEY=...                  # required
export LANGFUSE_PUBLIC_KEY=... LANGFUSE_SECRET_KEY=...   # optional, enables tracing
export LANGFUSE_HOST=https://us.cloud.langfuse.com       # default

uv sync
git clone --depth 1 https://github.com/chatwoot/docs corpus-src
uv run python download.py          # builds corpus/
uv run python index.py 1000        # and/or 200 (resumable; the free tier may need two days for both)
uv run python evaluate.py 1000 v1-strict
uv run python evaluate.py 1000 v2-revised
```

Chatwoot's documentation is © Chatwoot (developer docs under MIT). It is downloaded at run time and not redistributed here.
