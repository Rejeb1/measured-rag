# Eval run: rewrite_valid

Golden set: 57 questions (43 answerable, 14 unanswerable)

## Retrieval

```json
{'recall_at_k': {'1': 0.7093, '3': 0.8391, '5': 0.8798, '10': 0.9031, '20': 0.9442}, 'hit_rate_at_k': {'1': 0.7907, '3': 0.907, '5': 0.9302, '10': 0.9302, '20': 0.9767}, 'precision_at_k': {'1': 0.7907, '3': 0.3333, '5': 0.2186, '10': 0.114, '20': 0.0651}, 'mrr': 0.8464, 'ndcg_at_10': 0.8465, 'n_evaluated': 43, 'n_skipped_unanswerable': 14}
```

## By question type

| Question type | n | Recall@10 | Hit rate@10 | MRR | NDCG |
|---|---|---|---|---|---|
| aggregation | 5 | 0.167 | 0.400 | 0.163 | 0.120 |
| factoid | 30 | 1.000 | 1.000 | 0.944 | 0.959 |
| multi_hop | 8 | 1.000 | 1.000 | 0.906 | 0.880 |


## Latency budget

| Stage | p50 (ms) | p95 (ms) | mean (ms) | % of retrieval p50 |
|---|---|---|---|---|
| query_rewrite | 10863.6 | 31972.0 | 15908.3 | 81.9% |
| embed_query | 92.9 | 140.6 | 188.6 | 0.7% |
| bm25_search | 1.2 | 4.0 | 1.6 | 0.0% |
| dense_search | 0.3 | 1.1 | 0.5 | 0.0% |
| fusion | 0.5 | 0.9 | 0.6 | 0.0% |
| rerank | 3323.2 | 4595.7 | 3159.0 | 25.0% |
| assemble_context | 0.1 | 0.2 | 0.1 | 0.0% |
| **TOTAL_RETRIEVAL** | 13270.0 | 35190.1 | 19258.6 | — |
| **TOTAL** | 13270.0 | 35190.1 | 19258.6 | — |


## Failure analysis

| Failure mode | n | % of set | What it means |
|---|---|---|---|
| retrieval | 2 | 3.5% | gold chunk never retrieved — chunking/recall problem |
| ranking | 2 | 3.5% | retrieved but ranked too low — reranker problem |
| **clean** | 53 | 93.0% | — |

