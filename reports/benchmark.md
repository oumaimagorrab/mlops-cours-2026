# Benchmark de latence — german-credit, modèle v10

1000 appels par backend, après 10 appels de chauffe · x86_64 · Python 3.12.15

| Backend | P50 (ms) | P95 (ms) | P99 (ms) |
|---|---|---|---|
| sklearn_pipeline | 11.722 | 18.802 | 25.444 |
| onnx_python | 0.187 | 0.548 | 1.000 |
| onnx_batch_100 | 0.712 | 1.514 | 2.300 |

Accélération ONNX vs scikit-learn (P50, 1 ligne) : x62.6
