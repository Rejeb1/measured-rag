## Ablation

| Config | Recall@10 | NDCG@10 | MRR | p50 (ms) | p95 (ms) |
|---|---|---|---|---|---|
| Dense only | 0.909 | 0.833 | 0.821 | 12 | 18 |
| BM25 only | 0.864 | 0.802 | 0.810 | 1 | 1 |
| Hybrid (RRF) | 0.877 | 0.820 | 0.820 | 1 | 2 |
| Hybrid (normalized-sum) | 0.877 | 0.823 | 0.821 | 1 | 3 |
| Hybrid + reranker | 0.903 | 0.846 | 0.844 | 713 | 837 |
| Hybrid + reranker + query rewrite | 0.903 | 0.846 | 0.844 | 733 | 880 |
| Hybrid + reranker, top-20 pool | 0.892 | 0.838 | 0.840 | 396 | 804 |
| Hybrid + reranker, top-100 pool | 0.911 | 0.849 | 0.848 | 2742 | 4025 |
| Sequential context order | 0.903 | 0.846 | 0.844 | 1667 | 2788 |
| Chunk 256 (structure) | 0.768 | 0.558 | 0.502 | 734 | 2214 |
| Chunk 512 (structure) | 0.903 | 0.846 | 0.844 | 2719 | 4581 |
| Chunk 1024 (structure) | 0.911 | 0.889 | 0.908 | 2501 | 4324 |
| Chunk 512 (fixed-size) | 0.916 | 0.864 | 0.873 | 4388 | 4617 |


## Latency budget — Hybrid + reranker

| Stage | p50 (ms) | p95 (ms) | mean (ms) | % of retrieval p50 |
|---|---|---|---|---|
| embed_query | 0.0 | 0.0 | 0.0 | 0.0% |
| bm25_search | 0.4 | 1.1 | 0.5 | 0.1% |
| dense_search | 0.2 | 0.6 | 0.2 | 0.0% |
| fusion | 0.2 | 0.6 | 0.3 | 0.0% |
| rerank | 712.8 | 834.5 | 827.6 | 99.9% |
| assemble_context | 0.1 | 0.4 | 0.1 | 0.0% |
| **TOTAL_RETRIEVAL** | 713.5 | 837.0 | 828.9 | — |
| **TOTAL** | 713.5 | 837.0 | 828.9 | — |

