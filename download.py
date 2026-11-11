"""Step 1: build the corpus from Chatwoot's public docs.

- User guide: https://www.chatwoot.com/hc/user-guide/en (crawled: categories -> articles)
- Developer docs: https://github.com/chatwoot/docs (MIT), cloned into corpus-src/

Output: corpus/*.json with {url, title, text}. One paragraph per line.
Usage: git clone --depth 1 https://github.com/chatwoot/docs corpus-src && uv run python download.py
"""
import json
import re
import time

import httpx
from bs4 import BeautifulSoup

from config import CORPUS, ROOT

BASE = "https://www.chatwoot.com"
START = f"{BASE}/hc/user-guide/en"
CORPUS.mkdir(exist_ok=True)
http = httpx.Client(follow_redirects=True, timeout=30, headers={"User-Agent": "chatwoot-rag-evals/1.0"})


def links(html, pattern):
    return {a["href"].split("#")[0] for a in BeautifulSoup(html, "html.parser").find_all("a", href=True)
            if re.search(pattern, a["href"])}


# 1) user guide: home + categories -> articles
home = http.get(START).text
categories = links(home, r"/hc/user-guide/en/categories/")
articles = links(home, r"/hc/user-guide/articles/")
for c in sorted(categories):
    articles |= links(http.get(BASE + c).text, r"/hc/user-guide/articles/")
    time.sleep(0.3)
print(f"user guide: {len(categories)} categories, {len(articles)} articles")

for a in sorted(articles):
    url = BASE + a if a.startswith("/") else a
    soup = BeautifulSoup(http.get(url).text, "html.parser")
    title = (soup.find("h1") or soup.find("title")).get_text(strip=True)
    body = soup.find("article") or soup.find("main") or soup.body
    for t in body.find_all(["script", "style", "nav", "footer"]):
        t.decompose()
    # one paragraph per line: inline bold/links must not split a sentence across lines
    blocks = [b.get_text(" ", strip=True) for b in body.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre", "td"])
              if not b.find(["p", "li"])]
    text = "\n".join(re.sub(r"\s+([.,;:!?)])", r"\1", b) for b in blocks if b)
    text = re.sub(r"^.*?Last updated on [^\n]*\n", "", text, flags=re.S)   # drop the portal header
    slug = a.rstrip("/").split("/")[-1]
    (CORPUS / f"guide-{slug}.json").write_text(json.dumps({"url": url, "title": title, "text": text}, ensure_ascii=False))
    time.sleep(0.3)

# 2) developer docs (.mdx in the cloned repo)
n = 0
for f in (ROOT / "corpus-src").rglob("*.mdx"):
    raw = f.read_text()
    title = re.search(r"^title:\s*['\"]?(.+?)['\"]?\s*$", raw, re.M)
    text = re.sub(r"^---.*?---\s*", "", raw, flags=re.S)    # frontmatter
    text = re.sub(r"<[^>]+>", "", text)                       # JSX components
    path = str(f.relative_to(ROOT / "corpus-src")).removesuffix(".mdx")
    (CORPUS / f"dev-{path.replace('/', '__')}.json").write_text(json.dumps(
        {"url": f"https://developers.chatwoot.com/{path}", "title": title.group(1) if title else path, "text": text},
        ensure_ascii=False))
    n += 1
print(f"developer docs: {n} pages")
