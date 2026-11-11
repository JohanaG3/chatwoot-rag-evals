"""LLM-as-judge: scores one answer against the human-written rubric (data/rubric.md), structured JSON output."""
import json

from google.genai import types
from pydantic import BaseModel, Field

from config import JUDGE_MODEL, ROOT, gemini, lf, observe, retry

RUBRIC = (ROOT / "data/rubric.md").read_text()


class Score(BaseModel):
    razon: str = Field(description="1-3 frases: por qué esas notas, citando el fragmento si aplica")
    fidelidad: int = Field(ge=1, le=5)
    relevancia: int = Field(ge=1, le=5)


SYSTEM = f"""Eres un evaluador estricto y justo de un asistente de ayuda basado en documentos.
Recibes la pregunta, los fragmentos que tenía el asistente y su respuesta.
Califica SOLO con esta rúbrica (puedes usar 2 y 4 para casos intermedios):

{RUBRIC}
Primero razona, luego pon las notas. Juzga contra los fragmentos, no contra lo que tú sepas de Chatwoot."""


def judge(question, context, answer, trace_id=None):
    message = f"PREGUNTA:\n{question}\n\nFRAGMENTOS:\n{context}\n\nRESPUESTA DEL ASISTENTE:\n{answer}"
    with observe("judge", as_type="evaluator", input=message, metadata={"model": JUDGE_MODEL}):
        resp = retry(lambda: gemini.models.generate_content(
            model=JUDGE_MODEL, contents=message, config=types.GenerateContentConfig(
                system_instruction=SYSTEM, temperature=0,
                response_mime_type="application/json", response_schema=Score)))
        score = Score(**json.loads(resp.text))
    if lf and trace_id:   # scores attach to the answer's trace in Langfuse
        for name, value in (("faithfulness", score.fidelidad), ("relevance", score.relevancia)):
            lf.create_score(trace_id=trace_id, name=name, value=value,
                            comment=f"judge {JUDGE_MODEL}: {score.razon}", data_type="NUMERIC")
    return score
