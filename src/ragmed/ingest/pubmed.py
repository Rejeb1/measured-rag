"""PubMed ingestion via NCBI E-utilities.

Why this corpus: clinical abstracts are dense with exact identifiers (drug names,
dosages, HbA1c thresholds, NCT numbers, ICD codes) *and* heavy paraphrase (the same
mechanism described five different ways across five papers). That combination is what
makes the hybrid-retrieval ablation say something real rather than showing BM25 and
dense retrieval as interchangeable.

Structured abstracts also hand us genuine section labels (BACKGROUND / METHODS /
RESULTS / CONCLUSIONS), which is what makes structure-aware chunking testable against
fixed-size chunking instead of hypothetical.
"""

from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

import httpx

from ragmed.config import CorpusConfig
from ragmed.types import Document, Section

log = logging.getLogger(__name__)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# NCBI's published limits: 3 requests/sec anonymous, 10/sec with an API key.
# Staying under these is a courtesy requirement, not a suggestion - exceeding them
# gets the IP blocked.
RATE_NO_KEY = 3.0
RATE_WITH_KEY = 10.0

# efetch accepts a few hundred ids per request; 200 keeps responses parseable
# without hammering the endpoint.
EFETCH_BATCH = 200


class RateLimiter:
    def __init__(self, per_second: float):
        self.min_interval = 1.0 / per_second
        self._last = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        gap = now - self._last
        if gap < self.min_interval:
            time.sleep(self.min_interval - gap)
        self._last = time.monotonic()


def _text(el: ET.Element | None) -> str:
    """Flatten an element including nested markup (<i>, <sup>, <b> appear in titles)."""
    if el is None:
        return ""
    return " ".join(" ".join(el.itertext()).split())


@dataclass
class PubMedClient:
    cfg: CorpusConfig
    timeout_s: float = 30.0
    max_retries: int = 4

    def __post_init__(self) -> None:
        rate = RATE_WITH_KEY if self.cfg.api_key else RATE_NO_KEY
        self._limiter = RateLimiter(rate)
        self._client = httpx.Client(
            timeout=self.timeout_s,
            headers={"User-Agent": f"{self.cfg.tool}/0.1 (+https://github.com/)"},
        )
        if not self.cfg.email:
            log.warning(
                "no RAGMED_NCBI_EMAIL set; NCBI asks that automated clients identify "
                "themselves. Requests will still work but are rate-limited to %.0f/s.",
                RATE_NO_KEY,
            )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> PubMedClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _params(self, **kw: Any) -> dict[str, Any]:
        p: dict[str, Any] = {"db": "pubmed", "tool": self.cfg.tool}
        if self.cfg.email:
            p["email"] = self.cfg.email
        if self.cfg.api_key:
            p["api_key"] = self.cfg.api_key
        p.update(kw)
        return p

    def _get(self, endpoint: str, params: dict[str, Any]) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            self._limiter.wait()
            try:
                resp = self._client.get(f"{EUTILS}/{endpoint}", params=params)
                # 429 is a rate-limit signal; 5xx is NCBI having a moment. Both are
                # worth backing off on rather than failing the whole ingest.
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(
                        f"retryable status {resp.status_code}",
                        request=resp.request,
                        response=resp,
                    )
                resp.raise_for_status()
                return resp
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                backoff = 2.0**attempt
                log.warning(
                    "E-utilities %s failed (attempt %d/%d): %s; retrying in %.0fs",
                    endpoint,
                    attempt + 1,
                    self.max_retries,
                    exc,
                    backoff,
                )
                time.sleep(backoff)
        raise RuntimeError(f"E-utilities {endpoint} failed after {self.max_retries} attempts") from last_exc

    def search(self, term: str, retmax: int) -> list[str]:
        """Return PMIDs for a query, most relevant first."""
        pmids: list[str] = []
        # esearch caps retmax at 10000 per call; page through if asked for more.
        page_size = min(retmax, 10000)
        retstart = 0
        while len(pmids) < retmax:
            params = self._params(
                term=term,
                retmax=min(page_size, retmax - len(pmids)),
                retstart=retstart,
                retmode="json",
                sort="relevance",
            )
            data = self._get("esearch.fcgi", params).json()
            batch = data.get("esearchresult", {}).get("idlist", [])
            if not batch:
                break
            pmids.extend(batch)
            retstart += len(batch)
            if len(batch) < page_size:
                break
        return pmids[:retmax]

    def fetch(self, pmids: list[str]) -> list[Document]:
        docs: list[Document] = []
        for i in range(0, len(pmids), EFETCH_BATCH):
            batch = pmids[i : i + EFETCH_BATCH]
            params = self._params(id=",".join(batch), retmode="xml")
            xml = self._get("efetch.fcgi", params).text
            try:
                root = ET.fromstring(xml)
            except ET.ParseError as exc:
                log.error("efetch returned unparseable XML for %d ids: %s", len(batch), exc)
                continue
            for article in root.findall(".//PubmedArticle"):
                doc = _parse_article(article)
                if doc is not None:
                    docs.append(doc)
            log.info("fetched %d/%d records", min(i + EFETCH_BATCH, len(pmids)), len(pmids))
        return docs


def _parse_article(article: ET.Element) -> Document | None:
    pmid = _text(article.find(".//MedlineCitation/PMID"))
    if not pmid:
        return None

    art = article.find(".//MedlineCitation/Article")
    if art is None:
        return None

    title = _text(art.find("ArticleTitle"))
    sections = _parse_abstract(art.find("Abstract"))
    if not sections:
        # Title-only records are not retrievable text; the caller filters on this.
        return None

    journal = _text(art.find(".//Journal/Title"))
    year = _parse_year(art)
    doi = ""
    for aid in article.findall(".//ArticleIdList/ArticleId"):
        if aid.get("IdType") == "doi":
            doi = _text(aid)
            break

    pub_types = [_text(pt) for pt in art.findall(".//PublicationTypeList/PublicationType")]
    mesh = [
        _text(mh.find("DescriptorName"))
        for mh in article.findall(".//MeshHeadingList/MeshHeading")
    ]
    authors = []
    for a in art.findall(".//AuthorList/Author")[:12]:
        last, initials = _text(a.find("LastName")), _text(a.find("Initials"))
        if last:
            authors.append(f"{last} {initials}".strip())

    return Document(
        doc_id=f"pmid:{pmid}",
        title=title or "(untitled)",
        source_type="pubmed",
        sections=sections,
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        date=year,
        meta={
            "pmid": pmid,
            "journal": journal,
            "doi": doi,
            "publication_types": [p for p in pub_types if p],
            "mesh_terms": [m for m in mesh if m],
            "authors": authors,
        },
    )


def _parse_abstract(abstract: ET.Element | None) -> list[Section]:
    """Turn an <Abstract> into sections.

    Structured abstracts carry a Label per part; unstructured ones do not. Keeping the
    labels is what gives structure-aware chunking something to split on.
    """
    if abstract is None:
        return []
    sections: list[Section] = []
    for part in abstract.findall("AbstractText"):
        body = _text(part)
        if not body:
            continue
        label = part.get("Label") or part.get("NlmCategory")
        heading = label.strip().title() if label else None
        sections.append(Section(heading=heading, text=body))
    if len(sections) == 1 and sections[0].heading is None:
        sections[0].heading = "Abstract"
    return sections


def _parse_year(art: ET.Element) -> str | None:
    year = _text(art.find(".//Journal/JournalIssue/PubDate/Year"))
    if year:
        return year
    medline = _text(art.find(".//Journal/JournalIssue/PubDate/MedlineDate"))
    if medline:
        # MedlineDate looks like "2023 Jan-Feb" or "2019-2020"; take the leading year.
        head = medline.split()[0].split("-")[0]
        return head if head.isdigit() else None
    return None


def fetch_corpus(cfg: CorpusConfig) -> list[Document]:
    """Run every configured query and return deduplicated documents.

    The same paper legitimately matches several of our topic queries; deduping by PMID
    keeps the corpus honest. Without it, duplicated chunks inflate recall (two copies
    of the right answer means two chances to retrieve it) and the ablation table
    reports a system better than the one you built.
    """
    seen: dict[str, Document] = {}
    with PubMedClient(cfg) as client:
        for term in cfg.pubmed_queries:
            log.info("searching PubMed: %s", term)
            pmids = client.search(term, cfg.max_per_query)
            new = [p for p in pmids if f"pmid:{p}" not in seen]
            log.info("  %d hits, %d new", len(pmids), len(new))
            for doc in client.fetch(new):
                seen[doc.doc_id] = doc
    docs = list(seen.values())
    log.info("corpus: %d unique documents", len(docs))
    return docs
