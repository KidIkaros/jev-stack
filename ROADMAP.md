# jev-stack — Roadmap & Development Plan

> **Status:** Active  
> **Goal:** Build a production-ready Jev-compatible System 1 decision stack using Laya + EmbeddingGemma 2 + OpenJev flywheel.

## Current State

- ✅ Laya adapter (`src/adapter.py`) with `[STATE][CHOICE][ANSWER]` ↔ API mapping
- ✅ CLI tool (`src/cli.py`) for interactive decisions
- ✅ 20 unit tests + 8 integration tests (all passing)
- ✅ Quality gate: Laya loads in 0.5GB RAM, 33ms latency

## Missing Before "Production Ready"

1. **EmbeddingGemma 2 integration** — Currently using Laya's built-in tokenizer only. Need to add the 270M text encoder (0.5GB) for richer state representations.
2. **OpenJev flywheel** — System 2→1 self-evolution loop for low-confidence decisions
3. **Calibration validation** — Laya ships with uncalibrated temperatures; need to fit and measure ECE
4. **CI/CD** — No GitHub Actions, no automated test runs
5. **Documentation** — README, usage examples, API reference
6. **Benchmarks** — Latency, accuracy, option-order invariance metrics

## Roadmap

### Phase 1: Core Stack (Week 1)
- [ ] **Task 1**: Add EmbeddingGemma 2 text encoder layer (`src/gemma_embed.py`)
- [ ] **Task 2**: Update adapter to use embeddings + Laya head
- [ ] **Task 3**: Calibration validation — `agent.fit_temperatures()`, ECE < 0.1
- [ ] **Task 4**: Integration tests for encoder + decision pipeline

### Phase 2: Self-Evolution (Week 2)
- [ ] **Task 5**: OpenJev flywheel — System 2→1 loop for low-confidence decisions
- [ ] **Task 6**: Exploration strategy (rule-based + LLM fallback)
- [ ] **Task 7**: Crystallization back into Laya confidence
- [ ] **Task 8**: Integration tests for flywheel behavior

### Phase 3: Production Readiness (Week 3)
- [ ] **Task 9**: GitHub Actions CI (tests + quality gate)
- [ ] **Task 10**: README + usage examples
- [ ] **Task 11**: Benchmarks (latency, accuracy, option invariance)
- [ ] **Task 12**: Packaging (`pyproject.toml`, version, `pip install jev-stack`)

### Phase 4: Multimodal (Week 4)
- [ ] **Task 13**: Add vision encoder (170M) from EmbeddingGemma 2
- [ ] **Task 14**: Add audio encoder (300M) from EmbeddingGemma 2
- [ ] **Task 15**: Multimodal integration tests

## Execution Principles

1. **Tests first** — Every task has tests before implementation (TDD)
2. **Source-driven** — Verify against official docs (Laya, EmbeddingGemma, OpenJev)
3. **API-first** — Define interfaces before implementation
4. **Ship incrementally** — Commit after each task, even if incomplete

## Acceptance Criteria

Before the repo is considered "published" (beyond WIP):
1. All tests pass (unit + integration)
2. CI/CD is green
3. README explains the architecture and has usage examples
4. At least one benchmark metric (latency, ECE, or accuracy) is measured
5. This roadmap file is updated with completed tasks
