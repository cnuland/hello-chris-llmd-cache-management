# llm-d Cache Manager & Tiered KV Cache Offloading Demo

Animated web-based demo showcasing llm-d's distributed KV cache management for LLM inference — cache-aware routing, tiered offloading (GPU → CPU → Storage), and production deployment on OpenShift.

## Quick Start

```bash
open index.html
```

Or serve locally:
```bash
python3 -m http.server 8080
# then open http://localhost:8080
```

## Navigation

- **Arrow keys** (← →) or click **Prev / Next** to navigate scenes
- **Space** or click **Auto Play** for automatic scene advancement
- **Scene dots** at the bottom to jump to any section

## Demo Scenes

| # | Scene | What It Shows |
|---|-------|---------------|
| 1 | **Introduction** | Title + overview |
| 2 | **The Problem** | Animated: cache-blind routing wastes compute (3x redundant prefill) |
| 3 | **Block Hashing** | Animated: tokens → blocks → FNV-64a chained hashes with visual highlight |
| 4 | **KV-Cache Indexer** | Animated: KVEvents stream flowing from pods to indexer, index building live |
| 5 | **Cache-Aware Routing** | Animated: request → router → indexer query → pod scoring → route to best pod |
| 6 | **Tiered Offloading** | Animated: GPU fills → cascade to CPU → cascade to Storage, utilization bars |
| 7 | **Benchmarks** | Animated bar charts with published performance data (20-29% throughput gain) |
| 8 | **Summary** | Key takeaways grid |

## Reference Manifests

The `manifests/` directory contains annotated OpenShift/Kubernetes YAML:

- `vllm-serving.yaml` — vLLM InferenceService with CPU KV offloading enabled
- `kv-cache-indexer.yaml` — KV-Cache Indexer deployment + configuration
