# Jev 1.13 speed benchmark (R implementation)

Run 2026-09-19T18:52:25+0000 from this machine via OpenRouter; 100 abstracts per setting (seed 42, source `table/manual_review_by_design_50_each.csv`).
Latency is the client-observed round trip (curl total time) and includes network time. Cost is as reported by OpenRouter (input tokens only, $0.042/M).

| model | workers | n_abstracts | n_failed | wall_seconds | throughput_per_second | latency_ms_p50 | latency_ms_p90 | latency_ms_p99 | latency_ms_mean | input_tokens_mean | cost_usd_total | cost_usd_per_abstract | est_hours_for_corpus | est_cost_usd_for_corpus |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| typesafe/jev-1.13 | 1 | 100 | 0 | 19.43 | 5.147 | 339.9 | 425 | 1077.5 | 367.9 | 1712.7 | 0.00719 | 7.19e-05 | 2.47 | 3.3 |
| typesafe/jev-1.13 | 4 | 100 | 0 | 7.4 | 13.521 | 303.3 | 408.8 | 837.9 | 329.3 | 1712.7 | 0.00719 | 7.19e-05 | 0.94 | 3.3 |
| typesafe/jev-1.13 | 8 | 100 | 0 | 4.68 | 21.386 | 307.4 | 449.5 | 935.2 | 357 | 1712.7 | 0.00719 | 7.19e-05 | 0.59 | 3.3 |
| typesafe/jev-1.13 | 16 | 100 | 0 | 5.17 | 19.33 | 402.7 | 1560.6 | 1916.9 | 732.7 | 1712.7 | 0.00719 | 7.19e-05 | 0.66 | 3.3 |

## Extrapolation to the full corpus (n = 45,807)

| model | workers | throughput (abstracts/s) | est. hours | est. cost (USD) |
|---|---|---|---|---|
| deepseek-chat (DeepSeek V3.1) (README) | 5 | 1.27 | 10.0 | 3.00 |
| typesafe/jev-1.13 (best setting here) | 8 | 21.39 | 0.59 | 3.30 |

Speed-up over the DeepSeek run: about 17x at 8 workers. TypeSafe's published limit is 1,200 requests/minute (20/s), so throughput saturates around there.
