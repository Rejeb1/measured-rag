# ragmed — a *measured* RAG system over clinical literature

Hybrid retrieval (BM25 + dense) with reciprocal rank fusion, cross-encoder reranking,
and an evaluation layer that separates two things most RAG systems conflate:

- **Did the retriever find the right thing?** — recall@k, hit rate@k, MRR, NDCG@k.
  Deterministic, LLM-free, milliseconds.
- **Did the generator use it correctly?** — faithfulness, answer relevance, context
  precision, abstention accuracy. LLM-judged, and only trustworthy as far as the judge
  validation says.

Once those are measured separately you can say something precise. "recall@10 is 0.91
but faithfulness is 0.68" is a diagnosis — the retriever is fine, the generator is
ignoring its context. "My RAG works pretty well" is not.

---

## Architecture

```
                  ┌──────────────┐
   query ────────▶│ query rewrite│  (optional — off by default, see ablation)
                  └──────┬───────┘
                         │  1..n query variants
             ┌───────────┴───────────┐
             ▼                       ▼
    ┌─────────────────┐     ┌─────────────────┐
    │ BM25            │     │ dense bi-encoder│
    │ exact tokens:   │     │ paraphrase:     │
    │ HbA1c, SGLT2,   │     │ "stops kidney   │
    │ ICD-10, NCT ids │     │  reabsorbing    │
    │                 │     │  sugar"         │
    └────────┬────────┘     └────────┬────────┘
             │  top-50               │  top-50
             └───────────┬───────────┘
                         ▼
              ┌──────────────────────┐
              │ RRF  1/(60 + rank)   │   rank-based: no score normalisation
              └──────────┬───────────┘
                         ▼  50 candidates
              ┌──────────────────────┐
              │ cross-encoder rerank │   query+passage scored jointly
              └──────────┬───────────┘
                         ▼  top-5
              ┌──────────────────────┐
              │ assemble             │   dedup → edge-order → token budget
              └──────────┬───────────┘
                         ▼
              ┌──────────────────────┐
              │ generate + cite      │   or emit INSUFFICIENT_CONTEXT
              └──────────────────────┘
```

### Why each stage exists

**BM25 + dense in parallel.** They fail differently. Dense retrieval misses exact
identifiers — `SGLT2`, `NCT01131676`, `ICD-10`, `HbA1c` — because the embedding
smooths them away. BM25 misses paraphrase entirely. The BM25 tokenizer here is
hand-written for that job: it keeps alphanumeric compounds intact *and* emits their
parts, so a query for "ICD 10" matches a document containing "ICD-10". It also refuses
to stopword single characters, because stripping "a" would collapse *vitamin A* and
*vitamin D* into the same term.

**RRF for fusion.** Reciprocal Rank Fusion scores each document as `sum(1/(k + rank))`
with k=60. Because it consumes *ranks* rather than scores, it sidesteps the problem
that makes score fusion fragile: BM25 scores are an unbounded sum of IDF terms, cosine
similarity is bounded in [-1, 1], and any attempt to add them needs a normalisation
that is itself a tuning parameter. A `normalized_sum` fusion mode is implemented **as
a control**, not as an option — `tests/test_fusion.py` asserts that its scores shift
when an unrelated third candidate is retrieved, while RRF's do not.

**Cross-encoder reranking.** The bi-encoder never saw query and passage together — it
encoded each independently. A cross-encoder reads the pair jointly and is far more
accurate, but it is O(candidates) model calls per query. Hence retrieve-wide,
rerank-narrow: 50 cheap candidates, 5 expensive survivors.

**Context assembly.** Deduplicate (embedding cosine, falling back to token Jaccard so
it still works in the BM25-only ablation row), then order deliberately: models attend
most strongly to the beginning and end of a context window, so the two highest-scoring
chunks go at the *edges* rather than in descending order. Then enforce an explicit
token budget instead of letting context grow with candidate count.

**Chunking is a variable, not a constant.** Structure-aware chunking splits on section
boundaries first (structured PubMed abstracts hand us real `BACKGROUND / METHODS /
RESULTS / CONCLUSIONS` labels), packs whole sections up to the target size, and only
breaks a section apart if it exceeds the target alone. Fixed-size chunking is
implemented as the baseline it has to beat. Both pack *sentences*, never raw token
slices — a chunk that begins mid-sentence is unusable as a citation even when
retrieved correctly.

---

## Quickstart

```bash
# Pick the torch build that matches your machine, and install it FIRST so pip
# cannot resolve the wrong one as a transitive dependency:
pip install --index-url https://download.pytorch.org/whl/cu126 torch   # NVIDIA
pip install --index-url https://download.pytorch.org/whl/cpu   torch   # CPU only

pip install -r requirements.txt && pip install -e .
```

The reranker is ~98% of retrieval latency, so this choice matters more than any other
setup decision — see the latency budget below for both measurements. `device: auto` in
the config picks up CUDA with no further changes.

```bash
ragmed ingest      # fetch PubMed abstracts via E-utilities (no API key needed)
ragmed index       # chunk + build BM25 and dense indexes
ragmed ask "Which anticoagulant reduces major bleeding versus warfarin in AF?"
```

Generation, the golden-set builder and the LLM-judge need a local model:

```bash
ollama pull llama3.1:8b
ragmed golden      # generate + screen the golden set
ragmed eval --generation --label baseline
ragmed ablate --chunking
```

Serve it:

```bash
ragmed serve                      # or: docker compose up
curl localhost:8000/health
```

---

## The evaluation layer

### Golden set

**57 hand-written question / answer / source-chunk triples**
(`scripts/build_handwritten_golden.py`), each written after reading the chunk it is
labelled with. The builder validates every gold id against the live index and refuses to
write the file if one is missing — a stale id would score 0 recall forever and
masquerade as a retrieval failure. It also asserts multi-hop questions genuinely span
two documents.

**Why hand-written, when the generator is built and works.** The pipeline in
`eval/build_golden.py` (sample → generate → screen) is fully implemented, but it needs a
capable writer model, and `llama3.2:3b` — the largest that fits this machine's 4GB card
— is not one. Across three pilots it yielded 2/8 factoid and 1/4 multi-hop, producing
questions like *"What is the name of the drug used in the trial?"*: not self-contained,
useless as a standalone query. The screener correctly rejected them, which is the eval
layer working exactly as designed and still leaving no usable set. On a machine that can
run an 8B+ model, `ragmed golden` is the faster path.

Screening rejects anything not self-contained, whose answer is not in the source, or
that asks about document metadata rather than clinical content — and a screening
*failure* rejects the question rather than admitting it unscreened.

Four question types:

| Type | What it tests |
|---|---|
| `factoid` (30) | answerable from one chunk |
| `multi_hop` (8) | needs two chunks from **different documents** (asserted at build time) |
| `aggregation` (5) | "how many", "list all" — needs several chunks |
| `unanswerable` (14) | the answer is not in the corpus at all |

The last one matters most and is the one almost nobody includes. It is the only thing
that distinguishes a system that knows things from a system that will say anything. It
is measured deterministically: the generator emits a literal `INSUFFICIENT_CONTEXT`
sentinel, so abstention is detected by string comparison rather than by asking a second
model whether the first one refused.

Gold labels are **content-addressed chunk ids**, so re-running ingestion over an
unchanged corpus produces identical ids and a committed golden set stays valid across
re-indexing. Chunking ablations necessarily invalidate them — a 256-token re-chunk
means not one gold id still exists — so `remap_golden` re-projects gold labels onto the
new chunking by text containment before those rows are scored, and reports how many
labels it could not relocate. Without that step every chunking variant would report
0.0 recall and look like a catastrophic result rather than a broken measurement.

### Two metric layers, never blended

Retrieval metrics skip unanswerable items rather than scoring them zero — there is
nothing to retrieve, so counting them would drag every metric down in proportion to
how many unanswerable questions the set happens to contain, making sets of different
composition incomparable. `tests/test_retrieval_metrics.py` asserts this directly:
adding 20 unanswerable items must not move recall@10.

The judge is decomposed rather than holistic: it extracts atomic claims from an answer
and rules on each against the context. Asking a model for "a faithfulness score from 0
to 1" produces a number correlated with fluency; asking "is this specific sentence
supported by this specific text" produces a countable ratio. Parse failures are
recorded as **errors and excluded from the mean** — never coerced to a default, because
a coerced 0.5 looks exactly like a measurement.

### Judge validation

An unvalidated LLM-judge produces numbers, not measurements. `ragmed eval --generation`
writes `label_template.jsonl` with the judge's own verdict withheld (showing it would
anchor the labeller and the resulting "agreement" would measure suggestibility). Label
~50 by hand, then:

```bash
ragmed validate-judge --run runs/baseline --labels labels.jsonl
```

This reports agreement, a 95% interval that is honestly wide at n=50, mean absolute
error, and **bias direction** — a judge that is systematically generous inflates every
faithfulness number by a known amount, and a known bias can be stated alongside the
result.

---

## Results

Corpus as measured: **243 PubMed abstracts → 262 chunks**, mean 290 tokens,
5,431 BM25 terms, 384-dim embeddings (`BAAI/bge-small-en-v1.5`).

### Latency budget — measured, warm

10 clinical queries, 2 warmup passes discarded, `candidates=50`, GTX 1650 /
`torch 2.13.0+cu126`:

| Stage | p50 (ms) | p95 (ms) | % of retrieval p50 |
|---|---|---|---|
| embed_query (uncached) | 33.0 | 37.4 | 4.9% |
| embed_query (cached) | 0.0 | 0.1 | — |
| bm25_search | 0.9 | 1.3 | 0.1% |
| dense_search | 0.5 | 1.0 | 0.1% |
| fusion | 0.5 | 0.6 | 0.1% |
| **rerank** | **660.7** | **717.9** | **98.3%** |
| assemble_context | 0.3 | 0.6 | <0.1% |
| **TOTAL (retrieval)** | **672.1** | **758.3** | — |

**The headline number: reranking is 98% of retrieval latency.** Everything else —
both retrievers, fusion, dedup, ordering and budgeting — costs about 2ms combined.

Measured alternatives, and the same config on CPU-only torch for comparison:

| Configuration | p95 (GPU) | p95 (CPU) |
|---|---|---|
| Hybrid, no reranker | **2.5 ms** | 0.7 ms |
| Hybrid + reranker, 20 candidates | 356 ms | 2,219 ms |
| Hybrid + reranker, 50 candidates | 758 ms | 8,087 ms |

Two things worth reading off this table. Reranking scales **linearly in candidates** on
both devices — halving the pool halves the cost — so `rerank.candidates` is the single
most effective latency lever in the system. And the GPU buys an **11× speedup on the
reranker specifically** (7,876ms → 718ms p95) while changing nothing else, because
nothing else was ever the bottleneck.

This is what makes the quality columns below load-bearing rather than academic: if the
reranker does not buy a large NDCG gain, it has no business costing 98% of the latency
budget. The embedding cache is doing its job — a 72% hit rate across the benchmark
drove `embed_query` to 0.0ms on repeats.

The shipped latency gate is 1500ms, ~2× the measured p95, with a comment saying to
raise it to ~12000 on CPU-only torch. A gate that cannot be met on the hardware in use
stops being a gate.

### Ablation table

57 hand-written questions (43 answerable, 14 unanswerable), GTX 1650. Each row is a
**config patch against one base**, never a code change (`src/ragmed/eval/ablation.py`) —
if a row needed different code, the table would compare two systems that differ in more
ways than the label admits. `tests/test_end_to_end.py` asserts that disabling a stage
changes nothing else.

| Config | Recall@10 | NDCG@10 | MRR | p50 (ms) | p95 (ms) |
|---|---|---|---|---|---|
| **Dense only** | **0.909** | 0.833 | 0.821 | **12** | **18** |
| BM25 only | 0.864 | 0.802 | 0.810 | 1 | 1 |
| Hybrid (RRF) | 0.877 | 0.820 | 0.820 | 1 | 2 |
| Hybrid (normalized-sum) — *control* | 0.877 | 0.823 | 0.821 | 1 | 3 |
| Hybrid + reranker — *the default* | 0.903 | 0.846 | 0.844 | 713 | 837 |
| Hybrid + reranker, top-20 pool | 0.892 | 0.838 | 0.840 | 396 | 804 |
| Hybrid + reranker, top-100 pool | 0.911 | **0.849** | **0.848** | 2,742 | 4,025 |
| Hybrid + reranker + query rewrite | 0.903 | 0.847 | 0.846 | 13,270 | 35,190 |

**Three findings, two of which contradict the design this README argues for.**

**1. Dense-only beats the full pipeline on recall, at 1/46th the latency.**
0.909 vs 0.903 recall@10, 18ms vs 837ms p95. The reranker does buy real ranking quality
over hybrid (NDCG 0.820 → 0.846, MRR 0.820 → 0.844) — but it is buying back ground that
fusion gave away, and never overtakes the bi-encoder on recall.

**2. Adding BM25 *hurt*.** Hybrid RRF scores 0.877 recall against dense-only's 0.909 —
3.2 points worse. RRF fuses on rank, so a consistently weaker retriever (BM25 at 0.864)
drags down a stronger one's ordering. The rank-vs-score fusion control is a wash (0.877
both), so the loss comes from fusion *itself*, not from the fusion method.

**3. Query rewriting costs 10.9 seconds and buys 0.001 NDCG.** An 18× latency increase
for noise. Delete it. That number is only trustworthy because the silent no-op behind it
was found and fixed first — see finding 3 in the findings section below.

The candidate-pool rows behave as expected and are the real tuning knob: 20 → 50 → 100
candidates moves NDCG 0.838 → 0.846 → 0.849 while p95 goes 804 → 837 → 4,025ms.
Diminishing returns arrive well before the latency does.

**Rows excluded as invalid**, rather than quietly reported:

- *Sequential context order* — identical to baseline **by construction**. Retrieval
  metrics are computed on the ranked list, not the assembled context, so ordering cannot
  move them. It is a generation-metric row and belongs in that table.
- *Query rewrite, first run* — identical to baseline to three decimals because the LLM
  was never attached and rewriting silently no-opped. The row above is the corrected
  re-run, verified with `rewrite_active` true on 57/57 requests and 2.98 queries
  generated per request.

### Chunking

These rows re-index and re-project gold labels onto the new chunking (`remap_golden`).
Latency is omitted: they re-embed the corpus while evaluating, and GPU contention makes
the timings incomparable — the *same* config reads 713ms above and 2,719ms here.

| Config | Recall@10 | NDCG@10 | MRR | questions scored |
|---|---|---|---|---|
| Chunk 256 (structure) | 0.768 | 0.558 | 0.502 | **28 / 43** ⚠ |
| Chunk 512 (structure) — *the default* | 0.903 | 0.846 | 0.844 | 43 / 43 |
| Chunk 1024 (structure) | 0.911 | **0.889** | **0.908** | 43 / 43 |
| Chunk 512 (**fixed-size**) | **0.916** | 0.864 | 0.873 | 43 / 43 |

**Bigger chunks won, and structure-aware chunking lost to naive fixed-size** on recall
(0.903 vs 0.916) — a straightforward negative result for a design choice argued for at
length above. Abstracts are short and topically coherent, so section boundaries add
little, while structure-aware packing yields slightly smaller effective chunks.

**The 256-token row is not a fair comparison and must not be read as one.**
`remap_golden` relocates a gold label only when a new chunk contains ≥60% of the
original gold text. At 256 tokens a 489-token gold chunk splits into pieces that each
fall below that bar, so 15 of 43 questions were dropped. Its 0.768 is measured on a
different, smaller question set, and the containment threshold **systematically
penalises chunk sizes smaller than the one the golden set was built against**. Fixing it
properly means labelling gold at sentence level rather than chunk level.

### What the evidence would change

On this corpus and this question set, the data supports shipping **dense-only with a
reranker over a 50-candidate pool**, and dropping BM25 fusion and query rewriting
entirely — ~0.909 recall at ~730ms, or 18ms if the reranker goes too.

### The caveat that matters

**The golden set is biased toward dense retrieval, and I wrote it.** Questions were
deliberately paraphrased rather than copied from source wording — right for testing
semantic retrieval, wrong for testing BM25. It under-represents the exact-identifier
case (`NCT03057951`, `ICD-10`, `HbA1c`, `mg/dL`) that BM25 exists to catch.

So "hybrid hurts" is a real result **on this question set**, not a general claim about
hybrid retrieval. The honest fix is more identifier-anchored questions and a re-run —
not re-tuning until hybrid wins. Saying so is cheaper than being caught by it.

### Generation metrics

57 questions, `llama3.2:3b` as both generator and judge, ~4 hours on a GTX 1650.

| Metric | Value | Items scored | Trust |
|---|---|---|---|
| **Abstention accuracy** | **0.719** | 57 / 57 | **high** — no LLM involved |
| Faithfulness | 0.809 | 38 / 57 | low — 19 judge errors |
| Hallucination rate | 0.293 | 58 claims | low — under-decomposed |
| Answer relevance | 0.365 | 50 / 57 | **very low** — see below |
| Context precision | 0.374 | 43 / 57 | low — 14 judge errors |

**The judge failed on 20 of 57 items (35%).** Faithfulness is computed on two thirds of
the set; context precision on three quarters. Answer relevance is worse than incomplete
— its distribution is trimodal at exactly 0.25 (×25), 0.0 (×13) and 1.0 (×12), i.e. the
3B judge is collapsing onto rubric anchor points rather than discriminating. **0.365
should not be read as a property of the generator.** It is a property of the judge, and
`ragmed validate-judge` against human labels is the prerequisite for citing any row here
except the first.

That the first row is trustworthy and the rest are not is the whole argument for
measuring abstention deterministically instead of asking a model to grade a refusal.

#### The finding that matters: 28% of answers were not answers at all

**16 of 57 generated answers (28%) collapsed into repeated tokens**, e.g.

```
143 RCTs@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
{"question": "What is@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
```

This is not hallucination and not a prompt problem. It is the model failing on this
hardware. The evidence for that: the first collapse appears at question 9, the rate
climbs through the run, **9 of the last 14 answers are pure garbage**, and per-question
latency drifts from 270s to 386s over the same period. Sustained-load thermal
degradation on a 4GB laptop GPU fits; prompt content does not — the same prompts run
clean in a short fresh session (4/5 good answers in a five-question probe).

Breaking down the 14 unanswerable questions properly:

| Response | n | Reading |
|---|---|---|
| `@@@@` collapse | 9 | hardware failure, not a model judgement |
| Bare citation, no prose | 3 | degenerate in a second way |
| Correct refusal | 2 | correct |
| **Genuine hallucination** | **0** | — |

**The system never fabricated an answer to a question its corpus could not support.**
The refusal mechanism is undertested rather than broken: only 5 of 14 unanswerable
questions produced a coherent response at all, and 2 of those 5 correctly refused.

#### An earlier draft of this section was wrong

It read *"12 of 14 unanswerable questions were answered anyway — genuine hallucination
risk,"* and called it the most serious defect in the system. That claim came from
counting `abstained == False` without ever looking at the answer text. Nine of those
twelve were `@@@@@@@`.

The lesson is uncomfortable and worth keeping: **a metric computed over an unexamined
field will confidently describe something that never happened.** Abstention accuracy was
technically correct — those items genuinely did not contain the sentinel — and the
conclusion drawn from it was still false. Reading the raw outputs is not optional, and
no amount of metric hygiene substitutes for it.

Degenerate output is now detected (`is_degenerate`), excluded from every quality mean,
never sent to the judge, and reported as its own `degenerate_rate` — because a judge
handed `@@@@@@@` returns a faithfulness score for it, and that score would otherwise
launder a hardware failure into a quality metric. Pinned by `tests/test_degeneration.py`.

**Every generation number above is therefore computed on ~72% of the set, and the
underlying run is not a clean measurement of the generator.** A re-run on hardware that
can sustain the load is required before any of it is quoted.

#### A detector bug that inverted this result

The first run reported abstention accuracy **0.754** and **zero** refusals. Both were
wrong. The prompt asks the model to emit `INSUFFICIENT_CONTEXT`; the model emitted
`INSUFFICIENT CONTEXT`, and the exact-substring check missed every refusal.

| | broken detector | fixed detector |
|---|---|---|
| Refusals detected | 0 / 57 | 6 / 57 |
| Correct refusals (of 14 unanswerable) | 0 | 2 |
| Wrong refusals (of 43 answerable) | 0 | 4 |
| Abstention accuracy | 0.754 | **0.719** |

Note the direction: **fixing the bug lowered the score.** The broken detector was
hiding 4 incorrect refusals as well as 2 correct ones, and 0.754 was flattering.

Because abstention is a pure function of stored answer text, the fix was re-applied to
the completed run without repeating four hours of GPU time
(`scripts/rescore_abstention.py`). The matcher is now tolerant of spacing and case,
shared by the generator and the eval layer so they cannot disagree, and pinned by
`tests/test_abstention.py`.

The general lesson, which is the fourth instance of it in this project: **a contract
enforced by exact match against free-form model output is a contract a model will break
by doing something reasonable.**

### Failure analysis

`ragmed eval` attributes every failing question to the **earliest** stage that broke,
so categories are mutually exclusive and a generation failure downstream of a retrieval
failure is not counted as a generation failure:

| Category | Meaning | What fixes it |
|---|---|---|
| `retrieval` | gold chunk never appeared in the ranked list | chunking, embedding model, recall |
| `ranking` | retrieved, but below the context cutoff | the reranker |
| `abstention` | refused when it shouldn't, or answered when it shouldn't | prompt, threshold |
| `generation` | correct context, wrong answer | prompt, model |

Measured on the baseline run (57 questions):

| Failure mode | n | % of set |
|---|---|---|
| **abstention** | **16** | **28.1%** |
| generation | 8 | 14.0% |
| retrieval | 2 | 3.5% |
| ranking | 2 | 3.5% |
| clean | 29 | 50.9% |

Retrieval and ranking together account for **7% of failures**; abstention alone accounts
for **28%**. The retriever is not the problem with this system — the generator's
unwillingness to decline is. Without the split, all of this would have collapsed into a
single number around "50% clean" and pointed at nothing in particular.

(Abstention counts here are recomputed with the fixed detector: 12 unanswerable
questions answered plus 4 answerable ones refused. The console output from the original
run predates the fix.)

---

## Three real findings from building this

### 1. Ollama's default `num_ctx` breaks long prompts, and the symptom looks like a model defect

Ollama defaults `num_ctx` to **4096** regardless of what the model advertises —
`llama3.2` reports a 131,072-token window. Exceed the budget and generation does not
merely truncate: it fails, and it fails in ways that look like the model is broken.

The first golden-set pilot returned **0 kept questions out of 30**, with a mix of empty
responses, HTTP 500s, and degenerate output:

```
{"question": "What is@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
{"question": "How many patients with atr@@@@@@@@@@@@@@@@@@@@@@@
```

A controlled sweep — same prompts, same model, 5 runs per cell, varying only `num_ctx`
and device — settled it:

| `num_ctx` | prompt | tokens | OK / 5 | failures |
|---|---|---|---|---|
| 4096 | factoid (1 passage) | 595 | 5/5 | — |
| 4096 | multi-hop (2 passages) | 893 | 3/5 | 2× HTTP 500 |
| 4096 | aggregation (4 passages) | 1,264 | **0/5** | 1 degenerate, 4 empty |
| 8192 | factoid | 595 | 5/5 | — |
| 8192 | multi-hop | 893 | 5/5 | — |
| 8192 | aggregation | 1,264 | 5/5 | — |
| 8192 (CPU) | all three | — | 15/15 | — |

Failure rate tracks **prompt length at 4096** and disappears completely at 8192, on GPU
and CPU alike. The fix is one line — always send `num_ctx` — pinned by
`tests/test_llm.py::test_num_ctx_is_always_sent`.

**The detour is the interesting part.** The `@@@@@@` signature looks exactly like
numerical corruption, and the GTX 16xx series is Turing *without* tensor cores — a
documented source of garbage output from llama.cpp's F16 CUDA kernels. That hypothesis
was wrong: the GPU runs 15/15 clean at `num_ctx=8192` and is 1.7× faster than CPU
(6.7s vs 11.4s on the aggregation prompt). Adding `num_gpu=0` as an experimental arm
cost one extra column and killed a plausible, wrong story that would otherwise have
been written down as fact.

A second failure mode had a different cause: the pilot still produced 500s *after* the
`num_ctx` fix, because Ollama evicts an idle model after 5 minutes and an eval sweep has
natural gaps. Requests landing mid-reload fail. Hence `keep_alive`.

Two related hardening changes came out of the same pilot: transient empty responses,
5xx, and *truncated* JSON are now retried, and a persistent empty response raises rather
than returning `""` — which would have been indistinguishable from a refusal and would
have quietly inflated abstention accuracy.

### 2. BM25 returns nothing for an out-of-vocabulary query

BM25 returns an **empty result set**, not bad results. Asking
"Which antiplatelet agent is preferred after carotid endarterectomy?" of a corpus with
no carotid content produces zero candidates, not bad candidates.

Two consequences, both of which distort the abstention metric under a lexical-only
configuration:

1. Abstention becomes *trivially* correct — the system refuses because there was
   nothing to answer from, not because it judged the context inadequate. The metric
   ends up measuring the retriever while claiming to measure the generator.
2. The generator is never exercised on that question at all.

Dense retrieval always returns its top-k, so the hybrid configuration does not have
this hole. This is asserted, not hidden, in
`test_an_out_of_vocabulary_query_returns_nothing_under_lexical_only_retrieval` — along
with a companion test that a *topically adjacent* unanswerable question does retrieve,
so the generator genuinely has to decide to refuse. It is a good argument for writing
unanswerable questions that are close to the corpus rather than far from it.

### 3. Graceful degradation is a hazard in a measurement system

Query rewriting without a usable LLM falls back to the original query. That is exactly
right at request time — a rewrite failure must never take down a search. During an
ablation it is a disaster: the row runs, completes, and reports numbers **identical to
the baseline**, which reads as the clean publishable finding *"query rewriting doesn't
help"*. Nothing had been rewritten.

It survived two rounds of review because the evidence for it looked like evidence for
something else: metrics matching the baseline to three decimal places is precisely what
a genuine null result looks like. What gave it away was a `0.0ms` timing on a stage
whose whole job is an LLM round trip.

Fixed at the source rather than per call site — it had already appeared at two:

* `RetrievalPipeline` warns on construction when rewriting is enabled without a usable
  LLM, and exposes `rewrite_active`;
* every request's trace records `rewrite_active`, so the state is stated rather than
  inferred from a suspiciously round number;
* both CLI commands build the LLM when rewriting is on, not only under `--generation`.

With rewriting actually running, the real result is far more decisive than the fake one:
**10.9s p50 for +0.001 NDCG**.

This is the same shape as the other two findings — `num_ctx` returning empty responses,
BM25 returning an empty result set. In all three a component degraded politely and
quietly invalidated a measurement. For a system whose only product is trustworthy
numbers, the degraded path has to be **loud**, and that is now a design rule here rather
than a lesson learned three times.

---

## Production shape

- **FastAPI** with `/ask`, `/ask/stream` (SSE — sources first, then tokens),
  `/retrieve`, `/health`, `/metrics`.
- **Models load once at startup and are warmed** before the first request. Retrieval
  runs in a worker thread via `run_in_threadpool`: embedding, BM25 and cross-encoder
  inference are synchronous CPU work, and running them in the async handler would block
  the event loop — the service would look fine under `curl` and collapse under two
  concurrent users.
- **Structured JSON logs with a trace id per request**, propagated via `contextvars`
  and returned in the `x-trace-id` header, so a user-reported slow request can be
  attributed to a stage without reproducing it.
- **Embedding cache** (thread-safe LRU) keyed on model + prefix + text.
- **Docker**: CPU torch installed from its own index *before* everything else, so pip
  cannot pull the 2GB CUDA build transitively; models baked into the image so a cold
  start has no network dependency; index mounted as a volume.
- **CI**: `ruff` + a hermetic end-to-end test that builds a real index, runs the real
  pipeline and asserts the build gates with no network, no model download and no LLM.
  The full-corpus eval gate runs on a schedule, since it needs a PubMed fetch.

---

## Limitations

- **The golden set is machine-generated and machine-screened.** Screening removes the
  worst of it; it is not a substitute for hand review, and the README says so before
  any number derived from it.
- **The judge is a small local model.** Until `ragmed validate-judge` is run against
  human labels, generation numbers are unvalidated — treat them as directional.
- **Exact search, not ANN.** At 262 chunks a normalised matrix product is exact and
  sub-millisecond. An ANN index would add a *third* source of recall loss on top of
  chunking and the embedding model, and contaminate the ablation: a drop caused by the
  approximate index would look like a drop caused by the retriever. Revisit past ~10⁶
  chunks.
- **Corpus is abstracts, not full text.** PubMed abstracts are what E-utilities serves
  without a licence. Full guideline PDFs can be dropped into `data/raw/` and are
  ingested through the same path.
- **The generator and judge are a 3B model** (`llama3.2:3b`), chosen because it fits
  entirely in this machine's 4GB card — an 8B at q4 spills to CPU and turns a
  golden-set build into hours. This is now measured rather than assumed: the judge
  **errored on 35% of items** and collapsed answer relevance onto three rubric anchor
  points. Every generation number except abstention should be read as directional until
  `ragmed validate-judge` has been run against human labels. Abstention is exempt
  because no model is involved in scoring it.
- **The generation half of the eval is not a clean measurement.** 28% of answers
  collapsed into repeated tokens under sustained GPU load, so every quality figure is
  computed on the remaining ~72% and the refusal mechanism is barely exercised. The
  retrieval half is unaffected — it uses no LLM. Re-running generation on hardware that
  can sustain a four-hour load is the prerequisite for quoting any of it.
- **Highest-value next change:** a retrieval-score floor below which the system declines
  without calling the model at all. It removes an LLM round trip from the clearest
  refusal cases and makes abstention partly independent of generator quality.
- **Reranking is 98% of retrieval latency even on GPU.** The obvious lever is
  `rerank.candidates` (linear in cost); the ablation is what decides whether the
  quality justifies it.

---

## Layout

```
src/ragmed/
  config.py          every ablation knob; dotted-path patching
  types.py           Chunk/Document/GoldenItem + the abstention sentinel
  telemetry.py       per-stage spans, p50/p95 rollups, JSON logs
  ingest/            PubMed E-utilities, local files, chunking
  index/             BM25 (hand-rolled tokenizer), dense, bundled store
  retrieve/          fusion, rerank, assembly, pipeline
  eval/              metrics, golden set builder, ablation, failure analysis, judge validation
  generate.py        grounded prompt + citation extraction
  api.py             FastAPI service
  cli.py             ingest / index / golden / eval / ablate / ask / serve
tests/               139 tests, hermetic
```

Run the suite with `pytest -q`.
