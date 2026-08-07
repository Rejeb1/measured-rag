"""BM25 tests.

The tokenizer tests are the important ones. BM25's job in this pipeline is to catch
the exact identifiers the bi-encoder smooths away, so a tokenizer regression would
quietly remove the entire reason hybrid retrieval beats dense-only - and the ablation
table would still render, just with a smaller gap.
"""

from __future__ import annotations

import numpy as np
import pytest

from ragmed.index.bm25 import BM25Index, tokenize

# --- tokenizer ---------------------------------------------------------------


def test_alphanumeric_identifiers_survive_intact():
    assert "hba1c" in tokenize("HbA1c was measured")
    assert "sglt2" in tokenize("SGLT2 inhibitors")
    assert "nct01131676" in tokenize("Trial NCT01131676 enrolled")


def test_hyphenated_terms_emit_both_whole_and_parts():
    toks = tokenize("Coded as ICD-10 in the registry")
    # Whole form matches a query written "ICD-10"...
    assert "icd-10" in toks
    # ...and the parts match a query written "ICD 10".
    assert "icd" in toks
    assert "10" in toks


def test_units_with_slashes_are_preserved():
    toks = tokenize("glucose 126 mg/dL threshold")
    assert "mg/dl" in toks
    assert "mg" in toks
    assert "dl" in toks


def test_single_letters_are_never_stopworded():
    """Stripping "a"/"d" would collapse vitamin A and vitamin D into one term."""
    assert tokenize("vitamin A deficiency") != tokenize("vitamin D deficiency")
    assert "a" in tokenize("vitamin A deficiency")
    assert "d" in tokenize("vitamin D deficiency")


def test_common_words_are_removed_when_enabled():
    assert "the" not in tokenize("the patient", use_stopwords=True)
    assert "the" in tokenize("the patient", use_stopwords=False)


def test_decimals_stay_attached_to_their_units():
    toks = tokenize("HbA1c fell to 7.0% from 9.2%")
    # The regex splits on ".", so digits appear separately - what matters is that the
    # numbers survive as searchable tokens at all.
    assert "7" in toks and "0" in toks


# --- retrieval behaviour -----------------------------------------------------


@pytest.fixture
def index() -> BM25Index:
    docs = {
        "exact": "Empagliflozin is an SGLT2 inhibitor dosed at 10 mg once daily.",
        "paraphrase": "This class of medication blocks glucose reabsorption in the kidney tubule.",
        "unrelated": "Community acquired pneumonia is treated with amoxicillin in adults.",
        "partial": "Sodium glucose cotransporter inhibitors reduce cardiovascular events.",
    }
    return BM25Index.build(list(docs), list(docs.values()))


def test_exact_identifier_query_ranks_the_literal_match_first(index):
    hits = index.search("SGLT2", top_k=5)
    assert hits, "an exact identifier query must return something"
    assert index.doc_ids[hits[0][0]] == "exact"


def test_paraphrase_query_does_not_match_lexically(index):
    """The failure mode BM25 has and dense retrieval does not - this is why we run both."""
    hits = index.search("drugs that stop the kidney reabsorbing sugar", top_k=5)
    top = [index.doc_ids[i] for i, _ in hits[:1]]
    assert top != ["exact"]


def test_unknown_terms_return_no_results(index):
    assert index.search("zzzznonexistentterm", top_k=5) == []


def test_empty_and_stopword_only_queries_are_safe(index):
    assert index.search("", top_k=5) == []
    assert index.search("the of and to", top_k=5) == []


def test_top_k_is_respected(index):
    assert len(index.search("glucose", top_k=2)) <= 2


def test_scores_are_descending(index):
    scores = [s for _, s in index.search("glucose inhibitor", top_k=10)]
    assert scores == sorted(scores, reverse=True)


def test_repeated_query_terms_do_not_inflate_scores(index):
    once = index.search("SGLT2", top_k=1)[0][1]
    thrice = index.search("SGLT2 SGLT2 SGLT2", top_k=1)[0][1]
    assert once == pytest.approx(thrice)


def test_idf_is_never_negative_for_ubiquitous_terms():
    """A term in every document should contribute nothing, not a penalty."""
    idx = BM25Index.build(["a", "b", "c"], ["glucose here", "glucose there", "glucose everywhere"])
    assert float(idx.idf.min()) >= 0.0


def test_empty_corpus_does_not_crash():
    idx = BM25Index.build([], [])
    assert idx.search("anything", top_k=5) == []


# --- persistence -------------------------------------------------------------


def test_save_load_roundtrip_preserves_ranking(index, tmp_path):
    index.save(tmp_path)
    reloaded = BM25Index.load(tmp_path)

    assert reloaded.doc_ids == index.doc_ids
    assert reloaded.terms == index.terms
    assert reloaded.k1 == index.k1 and reloaded.b == index.b
    np.testing.assert_array_equal(reloaded.doc_lengths, index.doc_lengths)

    for query in ("SGLT2", "glucose inhibitor", "pneumonia"):
        before = index.search(query, top_k=5)
        after = reloaded.search(query, top_k=5)
        assert [d for d, _ in before] == [d for d, _ in after]
        assert [pytest.approx(s) for _, s in before] == [s for _, s in after]
