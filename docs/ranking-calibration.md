# Ranking Constant Calibration

Date: 2026-08-23
Applies to: OMEGA Core 1.5.13
Instrument: LongMemEval_S (500 questions, Wang et al. 2024), retrieval-only

This document records how the 1.5.13 ranking constants were chosen. It exists
because the release specification requires the weights, caps, and near-tie
boundary to come from an offline evaluation rather than from intuition.

## What is being calibrated

| Constant | Value | Role |
| --- | --- | --- |
| `SEMANTIC_NEAR_TIE_DELTA` | 0.01 | width of the band inside which metadata may act |
| `PRIORITY_MAX_ADDITIVE` | 0.005 | maximum priority contribution |
| `ACCESS_MAX_ADDITIVE` | 0.0025 | maximum access contribution |
| `ACCESS_SCORING_CAP` | 3 | access count beyond which scoring stops counting |
| `QUALITY_MAX_ADDITIVE` | 0.0025 | maximum bundled type/feedback/Thompson contribution |
| `RECENCY_MAX_ADDITIVE` | 0.05 | maximum recency contribution, applied to every candidate |
| `CE_SPREAD_FULL_SCALE` | 1.0 | cross-encoder score spread, in raw units, at which the rerank boost reaches full strength |

## Method, and a correction to it

The evaluation runs LongMemEval in `--dry-run` mode: sessions are ingested and
retrieved, with no generation and no LLM judge. Retrieval quality is the signal
the ranking constants actually control, and it costs nothing to measure.

The harness's existing metric is "evidence session in top-K". **That metric is
unusable for this purpose.** It reported 100.0% for every configuration tested,
including the unmodified 1.5.12 baseline, because it only asks whether the
evidence appears anywhere in the returned set. The ranking constants change the
*order* within that set, not its membership.

Passing `--limit 1` does not fix this: the harness applies a global floor of
`k = max(args.limit, 20)`, so every run measured K>=20 regardless.

The harness was therefore extended to record the rank of the first
evidence-bearing result and to report MRR, recall@1/@3/@5, and mean evidence
rank. Those metrics are not saturated and do respond to ranking changes. This
addition lives in `scripts/longmemeval_official.py`, which is not part of the
published wheel.

**Sample.** 116 questions, stratified and deterministic (no RNG): 30
knowledge-update, 30 temporal-reasoning, 20 multi-session, and 12 each of
single-session-assistant, single-session-preference, and single-session-user.
The two recency-sensitive categories are deliberately over-weighted. All runs
used an isolated HOME, a fresh store per configuration, and the local ONNX
embedding model. No network call was made; the dataset was already cached.

## Result: recency

`CE_SPREAD_FULL_SCALE` held at 1.0.

| `RECENCY_MAX_ADDITIVE` | MRR | recall@1 | mean rank |
| --- | --- | --- | --- |
| baseline v1.5.12 | 0.8819 | 81.9% | 1.59 |
| 0.0 | 0.9093 | 85.3% | 1.41 |
| 0.02 | 0.9085 | 85.3% | 1.41 |
| **0.05 (chosen)** | **0.9085** | **85.3%** | **1.41** |
| 0.10 | 0.9085 | 85.3% | 1.41 |
| 0.20 | 0.9051 | 84.5% | 1.42 |

Two things follow.

First, the 1.5.13 ranking rework is an improvement on the shipped 1.5.12
release, not a regression: MRR rises from 0.8819 to 0.9085 and recall@1 from
81.9% to 85.3%.

Second, LongMemEval **cannot discriminate** between recency settings from 0.0
to 0.10. The spread across that range is 0.0008 MRR, which is one question
moving one rank. The benchmark does establish an upper bound: at 0.20 the
degradation is real and visible in both MRR and recall@1.

The benchmark therefore constrains the constant rather than selecting it. Within
the range it certifies as safe, the value is fixed by the behavioural invariant
the release must satisfy: among semantically equivalent records, the fresher one
must rank first. 0.05 satisfies that invariant end-to-end with margin and sits a
factor of four below the point where measurable degradation begins.

**Alternatives considered.** 0.0 scores marginally highest but is the defective
behaviour under repair — it is the setting that lets a 60-day-old record outrank
an equivalent fresh one. 0.02 satisfies the invariant with less margin against
reranker noise. 0.10 is equally safe on the benchmark but doubles recency's
influence for no measured gain. 0.20 is rejected on evidence.

Note that 0.20 is also, by direct computation, the smallest value that would
make the original `test_fresh_ranks_above_old` fixture pass. The benchmark
independently rejects that value. That fixture was corrected rather than
accommodated; see below.

## Result: cross-encoder confidence

`RECENCY_MAX_ADDITIVE` held at 0.05.

| `CE_SPREAD_FULL_SCALE` | MRR | recall@1 | mean rank |
| --- | --- | --- | --- |
| 0.0 (scaling disabled) | 0.9085 | 85.3% | 1.41 |
| 0.5 | 0.9085 | 85.3% | 1.41 |
| **1.0 (chosen)** | **0.9085** | **85.3%** | **1.41** |
| 2.0 | 0.9085 | 85.3% | 1.41 |

The scaling is benchmark-neutral, which is the expected and desired outcome. On
LongMemEval the candidate sets are large and their cross-encoder scores are
genuinely spread out, so the confidence factor is close to 1.0 and the reranker
keeps its full authority.

The change exists for the degenerate case the benchmark does not contain. The
reranker boost is driven by min-max normalised scores, which map the best
candidate to 1.0 and the worst to 0.0 *however close they are*. Measured on the
pair used in the regression tests:

- two records differing only by a trailing reference token: spread 0.0066
- two near-paraphrases with a real difference: spread 0.7118

Before this change both produced an identical full-strength 15% boost. That is
how a negligible reranker preference could override a 60-day age difference, and
it is why no bounded recency term could satisfy the invariant on its own. 1.0 was
chosen as the full-scale point because the measurements above place the boundary
between "effectively tied" and "genuinely preferred" an order of magnitude below
it, and because it is benchmark-neutral.

## MemoryStress: not run

The specification also names MemoryStress. **It could not be run, and this gate
is not satisfied.** The reason is a missing artifact, not a scheduling choice:

- `benchmarks/memorystress/` contains 13 Python modules — loader, grader,
  metrics, schema, adapters — and no dataset.
- `DatasetLoader` requires a path to a MemoryStress JSON dataset. The only
  construction of `BenchmarkData` in the tree is that loader parsing such a file.
- No dataset file has ever been committed under `benchmarks/memorystress/` in
  the repository's history.
- There is no generator and no driver or CLI entry point that would run it.

MemoryStress is not the wrong instrument for this work. Its metrics include
`recall_at_age`, which is precisely the recency measurement this release needs,
and it would strengthen the evidence above. The blocker is solely that the
dataset artifact does not exist locally or in the repository, and it also
requires an external LLM provider for grading. Supplying the dataset would let
this gate be closed properly.

## Honest limits of this evidence

- LongMemEval discriminates the recency constant only at its upper end. The
  chosen value rests on a benchmark-certified safe range plus a behavioural
  invariant, not on a benchmark optimum.
- Retrieval-only evaluation measures ranking, which is the right target here,
  but it does not measure end-to-end answer quality.
- MemoryStress is absent, so the age-bucketed recall evidence the specification
  asks for is missing.
- The stratified 116-question sample is a subset of the full 500.

## Gate for future ranking changes

Any change to scoring, ranking, or these constants must run the **complete**
Core test suite with the embedding model available, not a targeted subset, plus
`tests/test_ranking_regression.py`. The 1.5.13 recency regression was visible in
a single full-suite run and was missed because only targeted files were run on
the release branch. CI already runs the full suite and lint for pull requests
into `main`; release branches must meet the same bar before qualification.
