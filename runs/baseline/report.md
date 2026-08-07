# Eval run: baseline

Golden set: 57 questions (43 answerable, 14 unanswerable)

## Retrieval

```json
{'recall_at_k': {'1': 0.7093, '3': 0.8333, '5': 0.8798, '10': 0.9031, '20': 0.9384}, 'hit_rate_at_k': {'1': 0.7907, '3': 0.8837, '5': 0.9302, '10': 0.9302, '20': 0.9767}, 'precision_at_k': {'1': 0.7907, '3': 0.3256, '5': 0.2186, '10': 0.114, '20': 0.064}, 'mrr': 0.8445, 'ndcg_at_10': 0.8455, 'n_evaluated': 43, 'n_skipped_unanswerable': 14}
```

## By question type

| Question type | n | Recall@10 | Hit rate@10 | MRR | NDCG |
|---|---|---|---|---|---|
| aggregation | 5 | 0.167 | 0.400 | 0.146 | 0.111 |
| factoid | 30 | 1.000 | 1.000 | 0.944 | 0.959 |
| multi_hop | 8 | 1.000 | 1.000 | 0.906 | 0.880 |


## Latency budget

| Stage | p50 (ms) | p95 (ms) | mean (ms) | % of retrieval p50 |
|---|---|---|---|---|
| embed_query | 41.0 | 298.8 | 168.8 | 1.0% |
| bm25_search | 0.9 | 3.6 | 2.9 | 0.0% |
| dense_search | 0.3 | 1.1 | 0.4 | 0.0% |
| fusion | 0.3 | 0.7 | 0.5 | 0.0% |
| rerank | 3839.5 | 5442.2 | 3496.6 | 96.8% |
| assemble_context | 0.2 | 1.8 | 0.8 | 0.0% |
| generate | 66985.7 | 167412.6 | 81002.1 | — |
| **TOTAL_RETRIEVAL** | 3966.7 | 5502.8 | 3670.1 | — |
| **TOTAL** | 71184.7 | 172649.5 | 84672.2 | — |


## Failure analysis

| Failure mode | n | % of set | What it means |
|---|---|---|---|
| retrieval | 2 | 3.5% | gold chunk never retrieved — chunking/recall problem |
| ranking | 2 | 3.5% | retrieved but ranked too low — reranker problem |
| abstention | 14 | 24.6% | refused when it shouldn't, or answered when it shouldn't |
| generation | 8 | 14.0% | correct context, wrong answer — prompt/model problem |
| **clean** | 31 | 54.4% | — |


## Generation

```json
{'faithfulness': 0.809, 'answer_relevance': 0.365, 'context_precision': 0.3744, 'abstention_accuracy': 0.7544, 'hallucination_rate': 0.2931, 'n_evaluated': 57, 'n_judge_errors': 20}
```
