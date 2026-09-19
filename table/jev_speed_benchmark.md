# Jev 1.13 speed benchmark

Run 2026-09-19T18:40:59+00:00 from this machine via OpenRouter; 100 abstracts per setting (seed 42, source `table/manual_review_by_design_50_each.csv`).
Latency is the client-observed round trip and includes network time. Cost is as reported by OpenRouter (input tokens only, $0.042/M).

| model | workers | n_abstracts | n_failed | wall_seconds | throughput_per_second | latency_ms_p50 | latency_ms_p90 | latency_ms_p99 | latency_ms_mean | input_tokens_mean | cost_usd_total | cost_usd_per_abstract | est_hours_for_corpus | est_cost_usd_for_corpus |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| typesafe/jev-1.13 | 1 | 100 | 0 | 31.36 | 3.189 | 291.1 | 401.2 | 512.5 | 312.2 | 1696.0 | 0.00712 | 7.12e-05 | 3.99 | 3.26 |
| typesafe/jev-1.13 | 4 | 100 | 0 | 8.03 | 12.455 | 291.1 | 385.9 | 808.7 | 311.8 | 1696.0 | 0.00712 | 7.12e-05 | 1.02 | 3.26 |
| typesafe/jev-1.13 | 8 | 100 | 0 | 4.96 | 20.16 | 306.2 | 600.0 | 1249.9 | 371.8 | 1696.0 | 0.00712 | 7.12e-05 | 0.63 | 3.26 |
| typesafe/jev-1.13 | 16 | 100 | 0 | 3.82 | 26.203 | 312.0 | 1145.5 | 1744.4 | 503.6 | 1696.0 | 0.00712 | 7.12e-05 | 0.49 | 3.26 |

## Extrapolation to the full corpus (n = 45,807)

| model | workers | throughput (abstracts/s) | est. hours | est. cost (USD) |
|---|---|---|---|---|
| deepseek-chat (DeepSeek V3.1) (README) | 5 | 1.27 | 10.0 | 3.00 |
| typesafe/jev-1.13 (best setting here) | 16 | 26.20 | 0.49 | 3.26 |

Speed-up over the DeepSeek run: about 21x at 16 workers. TypeSafe's published limit is 1,200 requests/minute (20/s), so throughput saturates around there.
