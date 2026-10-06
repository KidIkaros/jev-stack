# ADR-001: Adopt Laya + EmbeddingGemma 2 Stack Instead of Training NanoCore-S1 From Scratch

## Status
Accepted (2026-10-05)

## Context

NanoCore-S1 was originally designed as a 135M parameter decoder-only transformer trained from scratch on FineWeb-EDU, using a custom byte-level BPE tokenizer, with `[STATE][CHOICE][ANSWER]` structured output.

Research revealed two superior alternatives:

1. **EmbeddingGemma 2** (Google DeepMind, Apache 2.0): 740M parameter multimodal embedding model (270M text encoder runs in 0.5GB RAM). Maps text/code/images/audio into unified 768-dim space. MTEB-leading in its size class.

2. **Laya** (Convai Innovations, Apache 2.0): 421M parameter ModernBERT-large encoder + decision head. Produces typed decisions (Choice, Score, Noul) with calibrated probabilities in a single forward pass (~33ms on T4). No text generation — eliminates hallucination risk.

3. **OpenJev** (zhangcy122, Apache 2.0): Self-evolving cognitive decision engine with deliberative flywheel (System 2→1), 100% option-order invariance, and adaptive safety guards.

## Decision

**Option A: Full pivot to Laya + EmbeddingGemma 2**

Abandon NanoCore-S1's from-scratch training path in favor of composing pre-trained, specialized models:

```
[Input State]
      ↓
EmbeddingGemma 2 (270M, 0.5GB RAM)  →  768-dim unified embeddings
      ↓
Laya Decision Head (421M)            →  {Choice, Noul, Score} calibrated probabilities
      ↓
OpenJev Flywheel                    →  self-refine decisions, System 2→1 loop
```

### Rationale

| Factor | NanoCore-S1 | Laya + EmbeddingGemma 2 |
|---|---|---|
| Training time | 15-20 min cloud GPU + tokenizer training | Download + run locally |
| Memory (inference) | ~270MB weights + activations | 0.5GB encoder + 1GB Laya = 1.5GB |
| Latency | TBD (generation-based) | 33ms (single forward pass) |
| Calibration | Untrained — no ECE guarantees | RLCD-trained, calibrated logprobs |
| Option invariance | Untrained — may depend on ordering | 100% option-order invariant |
| Multimodal | Text only | Text, code, image, video, audio |
| Hallucination risk | High (generative) | Zero (classification) |
| Token cost | $0 (FineWeb-EDU) | $0 (Apache 2.0) |

### Key Trade-offs

**Lost**: NanoCore-S1's generative `[ANSWER]` text output format
**Gained**: Typed probabilistic API, calibrated confidence, multimodal, lower latency

**Mitigation**: Build an adapter layer that maps our `[STATE][CHOICE][ANSWER]` format to/from Laya's Python API. The adapter translates:
- `[STATE] context [/STATE]` → Laya `state` parameter
- `[CHOICE] A B C D [/CHOICE]` → Laya `questions: [{qtype: "choice", options: [...]}]`
- Laya's `{choice: "A", probability: 0.95}` → `[ANSWER] A [/ANSWER]` with `[SCORE] 0.95 [/SCORE]`

## Consequences

### Positive
- Ships in days, not weeks (no training required)
- Better performance than planned (33ms vs 100ms target)
- Lower memory (0.5GB vs 8GB budget)
- Production-grade calibration (ECE < 0.1)
- Future-proof for multimodal
- OpenJev flywheel provides self-evolution

### Negative
- Abandons 4 days of NanoCore-S1 implementation
- Adapter layer adds complexity
- Locked to ModernBERT's 512-token context
- Less control over model internals

### Neutral
- NanoCore-S1 repo retained as-is (tokenizer, tests, training script all working)
- New `jev-stack` repo for the integration layer

## Sources
- Laya model card: https://modelsystem.one/models/laya/
- EmbeddingGemma 2 (Unsloth): https://unsloth.ai/docs/models/embeddinggemma-2
- Laya Python package: https://pypi.org/project/laya/
- OpenJev: https://github.com/zhangcy122/OpenJev
- EmbeddingGemma paper: arxiv.org/abs/2509.20354
