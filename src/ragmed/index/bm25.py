"""Okapi BM25.

Hand-rolled rather than pulled from a library for one reason: the tokenizer is the
part that matters here, and off-the-shelf BM25 packages ship an English-web tokenizer
that destroys exactly the tokens this corpus depends on. ``HbA1c``, ``SGLT2``,
``ICD-10``, ``NCT01131676`` and ``25-hydroxyvitamin`` are the terms dense retrieval
smooths away, so they are precisely the terms BM25 has to get right - it is the entire
reason BM25 is in the pipeline.

The tokenizer therefore keeps alphanumeric compounds intact *and* emits their parts,
so a query for "ICD 10" still matches a document containing "ICD-10" and vice versa.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Keeps alphanumeric runs together, including internal hyphens, slashes and
# apostrophes: "hba1c", "icd-10", "mg/dl", "crohn's".
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'/][A-Za-z0-9]+)*")
_SPLIT_RE = re.compile(r"[-'/]")

# Deliberately short, and deliberately free of single characters. Stripping "a" would
# collapse "vitamin a" into "vitamin", and "d" would do the same to "vitamin d" - a
# distinction that matters more in this corpus than the handful of postings saved.
STOPWORDS = frozenset(
    """
    the be to of and in that have it for not on with he as you do at this but his by
    from they we say her she or an will my one all would there their what so up out
    if about who get which go me when make can like time no just him know take people
    into year your good some could them see other than then now look only come its
    over think also back after use two how our work first well way even new want
    because any these give day most us is are was were been has had does did doing
    """.split()
)


def tokenize(text: str, use_stopwords: bool = True) -> list[str]:
    """Lowercase, keep identifiers intact, and additionally emit their components."""
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text.lower()):
        term = match.group(0)
        if use_stopwords and term in STOPWORDS:
            continue
        tokens.append(term)
        # "icd-10" also contributes "icd" and "10" so a query written either way hits.
        if _SPLIT_RE.search(term):
            for part in _SPLIT_RE.split(term):
                if part and part != term and not (use_stopwords and part in STOPWORDS):
                    tokens.append(part)
    return tokens


class BM25Index:
    """Inverted index with flat-array postings.

    Postings are stored in one contiguous pair of arrays with per-term offsets rather
    than a dict of lists. That keeps scoring a handful of numpy slice operations and
    lets the whole index round-trip through .npz without pickle.
    """

    def __init__(
        self,
        doc_ids: list[str],
        terms: list[str],
        term_offsets: np.ndarray,
        postings_doc: np.ndarray,
        postings_tf: np.ndarray,
        doc_lengths: np.ndarray,
        k1: float,
        b: float,
        use_stopwords: bool,
    ):
        self.doc_ids = doc_ids
        self.terms = terms
        self._term_index = {t: i for i, t in enumerate(terms)}
        self.term_offsets = term_offsets
        self.postings_doc = postings_doc
        self.postings_tf = postings_tf
        self.doc_lengths = doc_lengths
        self.k1 = k1
        self.b = b
        self.use_stopwords = use_stopwords

        self.n_docs = len(doc_ids)
        self.avg_doc_len = float(doc_lengths.mean()) if self.n_docs else 0.0

        # Document frequency per term, derived from the offsets.
        df = np.diff(term_offsets).astype(np.float64)
        # Robertson/Sparck-Jones IDF with the +1 guard, so a term appearing in every
        # document scores 0 rather than going negative and actively penalising a hit.
        self.idf = np.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    @classmethod
    def build(
        cls,
        doc_ids: list[str],
        texts: list[str],
        k1: float = 1.2,
        b: float = 0.75,
        use_stopwords: bool = True,
    ) -> BM25Index:
        assert len(doc_ids) == len(texts)
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        doc_lengths = np.zeros(len(texts), dtype=np.int32)

        for i, text in enumerate(texts):
            toks = tokenize(text, use_stopwords)
            doc_lengths[i] = len(toks)
            counts: dict[str, int] = defaultdict(int)
            for t in toks:
                counts[t] += 1
            for term, tf in counts.items():
                postings[term].append((i, tf))

        terms = sorted(postings)
        term_offsets = np.zeros(len(terms) + 1, dtype=np.int64)
        total = sum(len(postings[t]) for t in terms)
        postings_doc = np.zeros(total, dtype=np.int32)
        postings_tf = np.zeros(total, dtype=np.float32)

        cursor = 0
        for idx, term in enumerate(terms):
            term_offsets[idx] = cursor
            for doc_i, tf in postings[term]:
                postings_doc[cursor] = doc_i
                postings_tf[cursor] = tf
                cursor += 1
        term_offsets[len(terms)] = cursor

        log.info(
            "BM25 index: %d docs, %d unique terms, %d postings, avg len %.1f tokens",
            len(doc_ids),
            len(terms),
            total,
            float(doc_lengths.mean()) if len(doc_lengths) else 0.0,
        )
        return cls(
            doc_ids, terms, term_offsets, postings_doc, postings_tf, doc_lengths, k1, b, use_stopwords
        )

    def search(self, query: str, top_k: int = 50) -> list[tuple[int, float]]:
        """Return (doc_index, score) pairs, best first."""
        if self.n_docs == 0:
            return []
        query_terms = tokenize(query, self.use_stopwords)
        if not query_terms:
            return []

        scores = np.zeros(self.n_docs, dtype=np.float32)
        # Length normalisation denominator is per-document and query-independent.
        norm = self.k1 * (1.0 - self.b + self.b * (self.doc_lengths / max(self.avg_doc_len, 1e-9)))

        # A repeated query term should not multiply its own contribution.
        for term in set(query_terms):
            ti = self._term_index.get(term)
            if ti is None:
                continue
            lo, hi = self.term_offsets[ti], self.term_offsets[ti + 1]
            docs = self.postings_doc[lo:hi]
            tfs = self.postings_tf[lo:hi]
            contribution = self.idf[ti] * (tfs * (self.k1 + 1.0)) / (tfs + norm[docs])
            scores[docs] += contribution.astype(np.float32)

        return _top_k(scores, top_k)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path / "bm25.npz",
            term_offsets=self.term_offsets,
            postings_doc=self.postings_doc,
            postings_tf=self.postings_tf,
            doc_lengths=self.doc_lengths,
        )
        (path / "bm25_meta.json").write_text(
            json.dumps(
                {
                    "doc_ids": self.doc_ids,
                    "terms": self.terms,
                    "k1": self.k1,
                    "b": self.b,
                    "use_stopwords": self.use_stopwords,
                }
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        arrays = np.load(path / "bm25.npz")
        meta = json.loads((path / "bm25_meta.json").read_text(encoding="utf-8"))
        return cls(
            doc_ids=meta["doc_ids"],
            terms=meta["terms"],
            term_offsets=arrays["term_offsets"],
            postings_doc=arrays["postings_doc"],
            postings_tf=arrays["postings_tf"],
            doc_lengths=arrays["doc_lengths"],
            k1=meta["k1"],
            b=meta["b"],
            use_stopwords=meta["use_stopwords"],
        )


def _top_k(scores: np.ndarray, top_k: int) -> list[tuple[int, float]]:
    """Top-k by score, dropping zeros.

    argpartition rather than a full sort: at k=50 over tens of thousands of chunks
    this is the difference between a sub-millisecond and a several-millisecond stage,
    and the latency table is supposed to reflect the retrieval work, not the sort.
    """
    nonzero = int(np.count_nonzero(scores))
    if nonzero == 0:
        return []
    k = min(top_k, nonzero)
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx], kind="stable")]
    return [(int(i), float(scores[i])) for i in idx if scores[i] > 0]
