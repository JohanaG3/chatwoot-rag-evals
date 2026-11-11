"""Shared settings and clients. Keys come from env vars (or ~/.gemini_key, ~/.langfuse_* as a fallback)."""
import contextlib
import os
import pathlib
import time

from google import genai

ROOT = pathlib.Path(__file__).parent
CORPUS = ROOT / "corpus"
CHROMA = ROOT / "chroma"

EMBED_MODEL, EMBED_DIM = "gemini-embedding-001", 768
ANSWER_MODEL = "gemini-3.5-flash-lite"
JUDGE_MODEL = "gemini-3.5-flash-lite"   # fixed: never change the judge between runs you compare


def _secret(env, file):
    if os.environ.get(env):
        return os.environ[env].strip()
    path = pathlib.Path.home() / file
    return path.read_text().strip() if path.exists() else None


gemini = genai.Client(api_key=_secret("GEMINI_API_KEY", ".gemini_key"))

# Langfuse is optional: without keys everything runs, just without traces.
_pk, _sk = _secret("LANGFUSE_PUBLIC_KEY", ".langfuse_public"), _secret("LANGFUSE_SECRET_KEY", ".langfuse_secret")
if _pk and _sk:
    from langfuse import Langfuse, propagate_attributes
    lf = Langfuse(public_key=_pk, secret_key=_sk,
                  host=os.environ.get("LANGFUSE_HOST", "https://us.cloud.langfuse.com"))
else:
    lf = None


class _NoTrace:
    def update(self, **_):
        pass


@contextlib.contextmanager
def observe(name, **kwargs):
    """A Langfuse observation when tracing is on, a no-op otherwise."""
    if lf is None:
        yield _NoTrace()
    else:
        with lf.start_as_current_observation(name=name, **kwargs) as o:
            yield o


def trace_attributes(**kwargs):
    return propagate_attributes(**kwargs) if lf else contextlib.nullcontext()


def retry(fn, attempts=15):
    """Free tier: 429 (quota) and 503 (overloaded) are routine. Back off and try again."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            wait = 20 * (i + 1)
            print(f"  api error ({str(e)[:70]}), waiting {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError("gave up after retries")
