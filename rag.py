"""The RAG assistant. One question = one trace with two steps: retrieve -> answer."""
import chromadb
from google.genai import types

from config import ANSWER_MODEL, CHROMA, EMBED_DIM, EMBED_MODEL, gemini, lf, observe, retry, trace_attributes

db = chromadb.PersistentClient(path=str(CHROMA))

# Prompt versions. The regression test changes only this. Kept verbatim as they were run
# (v2 says "di que no lo encuentras", which once leaked into an answer as "No lo encuentras").
PROMPTS = {
    "v1-strict": """Eres el asistente de ayuda de Chatwoot. Responde en español, en 2-5 frases.
Usa SOLO la información de los fragmentos numerados. Cita cada afirmación con su número, por ejemplo [2].
Si los fragmentos no contienen la respuesta, responde exactamente: "No lo encuentro en la documentación."
No inventes nada que no esté en los fragmentos.""",
    "v2-revised": """Eres el asistente de ayuda de Chatwoot. Responde en español, en 2-5 frases. Cita cada afirmación con su número, por ejemplo [2].
Usa la información de los fragmentos, pero úsalos todos: evalúa el contexto que tiene cada uno, a ver si encuentras la respuesta.
En el caso de que no esté claro o definitivamente no haya nada, no lo inventes: di que no lo encuentras en la documentación.""",
}


def answer(question, words=1000, prompt="v1-strict", top_k=5, tags=(), session=None):
    col = db.get_collection(f"chatwoot_{words}")
    with observe("rag-chatwoot", as_type="chain", input={"question": question},
                 metadata={"chunk_words": words, "prompt": prompt, "top_k": top_k}) as trace, \
         trace_attributes(tags=[f"words-{words}", prompt, *tags], session_id=session, trace_name="rag-chatwoot"):

        with observe("retrieve", as_type="retriever", input=question) as step:
            qv = retry(lambda: gemini.models.embed_content(
                model=EMBED_MODEL, contents=question,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=EMBED_DIM))
            ).embeddings[0].values
            r = col.query(query_embeddings=[qv], n_results=top_k)
            chunks = [{"n": i + 1, "title": m["title"], "url": m["url"], "similarity": round(1 - d, 3), "text": doc}
                      for i, (doc, m, d) in enumerate(zip(r["documents"][0], r["metadatas"][0], r["distances"][0]))]
            step.update(output=[{k: c[k] for k in ("n", "title", "similarity", "url")} for c in chunks])

        context = "\n\n".join(f"[{c['n']}] ({c['title']})\n{c['text']}" for c in chunks)
        message = f"Fragmentos:\n{context}\n\nPregunta: {question}"
        with observe("answer", as_type="generation", model=ANSWER_MODEL, model_parameters={"temperature": 0},
                     input=[{"role": "system", "content": PROMPTS[prompt]}, {"role": "user", "content": message}]) as gen:
            resp = retry(lambda: gemini.models.generate_content(
                model=ANSWER_MODEL, contents=message,
                config=types.GenerateContentConfig(system_instruction=PROMPTS[prompt], temperature=0)))
            u = resp.usage_metadata
            gen.update(output=resp.text.strip(),
                       usage_details={"input": u.prompt_token_count or 0, "output": u.candidates_token_count or 0})

        trace.update(output=resp.text.strip())
        trace_id = lf.get_current_trace_id() if lf else None
    return {"answer": resp.text.strip(), "chunks": chunks, "context": context, "trace_id": trace_id}
