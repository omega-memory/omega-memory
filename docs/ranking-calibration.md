# Ranking Constant Calibration

Date: 2026-08-23
Applies to: OMEGA Core 1.5.13
Instrument: LongMemEval_S (500 questions, Wang et al. 2024), retrieval-only

This document records how the 1.5.13 ranking constants were chosen. It exists
because the release specification requires the weights, caps, and near-tie
boundary to come from an offline evaluation rather than from intuition.

**This is the second version of this document.** The first version drew
conclusions from a benchmark run that could not see the signal it claimed to be
measuring. Independent review disproved several of its claims. The withdrawn
claims are listed below rather than quietly dropped; the original text remains
in git history at commit `d6e2f56`.

> ## OPEN DEFECTS — this calibration is NOT final
>
> A second independent re-review found two defects that this document does not
> yet resolve. They are recorded here rather than left for a reader to discover.
>
> **1. The per-model scales were measured off the production path.** Every pair
> measurement below calls `cross_encoder_score(query, passages)` with no
> temporal metadata. The shipped pipeline always passes day-granular dates, so
> the reranker actually scores `"[Date: YYYY-MM-DD] {passage}"`. Re-measured
> through the production path, the `bge-reranker-v2-m3` window narrows to
> roughly (1.82, 2.70) and the shipped scale of **3.5 falls outside it**: a
> genuinely separated pair earns confidence 0.386, below the 0.5 the scale was
> chosen to guarantee. `ms-marco-MiniLM-L-6-v2` at 1.0 remains inside its
> production window. This is the failure the "Gate for future ranking changes"
> section below forbids, committed by this document itself.
>
> **2. A non-finite cross-encoder logit propagates NaN.** `ce_range` is tested
> with `> 0`, which `inf` satisfies, so `ce_norm` becomes NaN before the
> confidence factor is applied, and `NaN * 0.0` is NaN. The claim in
> `_query.py` that zero confidence leaves the ordering untouched does not hold
> for non-finite scores.
>
> Both are being handed to the owner as required corrective work. Treat the
> per-model scale table below as provisional.

## Withdrawn claims

| Withdrawn claim | What the evidence actually shows |
| --- | --- |
| "`0.20` is the smallest `RECENCY_MAX_ADDITIVE` that makes the original `test_fresh_ranks_above_old` fixture pass." | It depends entirely on which reranker is installed, and neither answer is 0.20. Under `bge-reranker-v2-m3` the fixture passes with recency switched **off**; under `ms-marco-MiniLM-L-6-v2` the threshold is between 0.17 (fails) and 0.18 (passes). |
| "At `0.20` the degradation is real and visible." | The run behind that sentence could not measure recency at all (see below). On the corrected instrument 0.20 does score lowest, but by a margin no paired test supports. |
| "LongMemEval **certifies** a safe recency range." | The benchmark constrains and informs; it certifies nothing. The word overstated a 116-question retrieval-only sweep. |
| "The recency sweep evaluates freshness." | It did not. Every candidate scored `decay_factor ≈ 1.0`, so the sweep varied a constant that was multiplied by an effectively fixed number. |
| "A single global `CE_SPREAD_FULL_SCALE` of 1.0 is correct." | Raw cross-encoder logits are model-scaled. One constant cannot satisfy both supported rerankers simultaneously. |
| "MemoryStress has no dataset, no generator, and no driver." | Factually wrong. A generator and a driver both exist and both run; only the dataset artifact is genuinely uncommitted. Corrected in the release specification. |
| "The scaling is benchmark-neutral, which is the expected and desired outcome." | Withdrawn. On the corrected harness, disabling the magnitude boost costs 4 questions of recall@1. Neutrality holds only within the evidence window. |
| MemoryStress "metrics include `recall_at_age`, which is precisely the recency measurement this release needs". | Withdrawn. `recall_at_age` buckets by session index, not calendar time, and the OMEGA adapter never writes `created_at`, which is the field decay reads. |
| "All runs used an isolated HOME." | False as written. The qualification environment deliberately retains the real HOME read-only so the production-preferred reranker is the one exercised; OMEGA state and TMPDIR are what is isolated. |
| "The two recency-sensitive categories are deliberately over-weighted" offered as support for the sweep. | Withdrawn as support. Both over-weighted categories are largely inert to the swept constant because of the harness confound disclosed under Method. |

The rows above paraphrase the withdrawn claims where the original wording was
long; the exact prior text is at `d6e2f56`.

## What is being calibrated

| Constant | Value | Role |
| --- | --- | --- |
| `SEMANTIC_NEAR_TIE_DELTA` | 0.01 | width of the band inside which metadata may act |
| `PRIORITY_MAX_ADDITIVE` | 0.005 | maximum priority contribution |
| `ACCESS_MAX_ADDITIVE` | 0.0025 | maximum access contribution |
| `ACCESS_SCORING_CAP` | 3 | access count beyond which scoring stops counting |
| `QUALITY_MAX_ADDITIVE` | 0.0025 | maximum bundled type/feedback/Thompson contribution |
| `RECENCY_MAX_ADDITIVE` | 0.05 | maximum recency contribution, applied to every candidate |
| `_CE_SPREAD_FULL_SCALE_BY_MODEL` | 1.0 / 3.5 | per-reranker cross-encoder spread, in that model's own raw units, at which the rerank boost reaches full strength |

## Method, and two corrections to it

The evaluation runs LongMemEval in `--dry-run` mode: sessions are ingested and
retrieved, with no generation and no LLM judge. Retrieval quality is the signal
the ranking constants actually control, and it costs nothing to measure.

**Correction 1: the metric was saturated.** The harness's original metric is
"evidence session in top-K". It reported 100.0% for every configuration tested,
including the unmodified 1.5.12 baseline, because it only asks whether the
evidence appears anywhere in the returned set. The ranking constants change the
*order* within that set, not its membership. Passing `--limit 1` does not fix
this: the harness applies a global floor of `k = max(args.limit, 20)`, so every
run measured K>=20 regardless. The harness was extended to record the rank of
the first evidence-bearing result and to report MRR, recall@1/@3/@5, and mean
evidence rank.

**Correction 2: the instrument could not see recency.** This is the defect that
invalidated the first version of this document. LongMemEval sessions carry 2023
dates, and the harness wrote them to `referenced_date`. Decay is computed from
`created_at`, which the harness left at ingest time — so every record was
"created" seconds ago and scored `decay_factor ≈ 1.0`. Writing the literal 2023
dates instead is equally blind in the opposite direction: at that age every
record sits on the `_DECAY_FLOOR_NEVER_ACCESSED = 0.15` floor.

The harness now backdates `created_at` to each session's own date **and** shifts
the whole timeline so the question's "present" is now, preserving the relative
spacing the dataset encodes.

Measured across the whole sample — 4999 haystack sessions over the 106 sampled
questions that carry dates — decay then takes 2728 distinct values spanning the
full 0.15 to 1.0 range, rather than sitting at a constant. It is a live signal.
It is not a uniformly live one: **18.4% of sessions are pinned on the 0.15
floor**, 4.8% sit at exactly 1.0 because they post-date the question, 3
questions have their entire haystack on the floor and 22 more have over half of
it there. An earlier version of this paragraph quoted "53 distinct values from
0.58 to 0.996", which was a probe of a **single** question presented as a
sample-wide property; that is corrected here rather than dropped.

Every number in the first version of this document was produced before this fix
and should be treated as measuring something else.

Both harness changes live in `scripts/longmemeval_official.py`, which is not
part of the published wheel.

**A confound this document must disclose.** The harness contains its own
recency mechanism, `_boost_recency` at `scripts/longmemeval_official.py:955`,
enabled by default and applied to **every knowledge-update question**. It
multiplies relevance by 1.0 to 1.5 by `referenced_date` and re-sorts. That
dwarfs a `RECENCY_MAX_ADDITIVE` of 0.05, and it runs on `referenced_date`, which
the harness fix deliberately leaves in 2023, rather than on the backdated
`created_at` the store's decay reads. The consequence is visible in the
per-category numbers: knowledge-update MRR moves by 0.0004 across the entire
recency sweep and temporal-reasoning by 0.0056, while single-session-preference
moves by 0.042. The sweep's signal comes from the categories the sample does
*not* over-weight. The sweep below should be read with that in mind, and a
re-run with `--no-recency-boost` is required before its shape is relied on.

**Sample.** 116 questions, stratified and deterministic (no RNG): 30
knowledge-update, 30 temporal-reasoning, 20 multi-session, and 12 each of
single-session-assistant, single-session-preference, and single-session-user.
The two categories intended as recency-sensitive are over-weighted; per the
confound above, they are also the two the harness's own boost renders least
sensitive to the constant under test. All runs
used isolated OMEGA state and TMPDIR, a fresh store per configuration, and the
local ONNX embedding model, with reranker auto-download disabled so a missing
model fails loudly instead of silently substituting a different one. No network
call was made; the dataset was already cached.

## Result: recency

Measured on the corrected instrument, cross-encoder scaling at the calibrated
per-model value.

| `RECENCY_MAX_ADDITIVE` | MRR | recall@1 | mean rank |
| --- | --- | --- | --- |
| baseline v1.5.12 | 0.8536 | 88/116 (75.9%) | 1.65 |
| 0.0 | 0.9052 | 98/116 (84.5%) | 1.42 |
| 0.02 | 0.9085 | 99/116 (85.3%) | 1.41 |
| **0.05 (chosen)** | **0.9085** | **99/116 (85.3%)** | **1.41** |
| 0.10 | 0.8998 | 97/116 (83.6%) | 1.44 |
| 0.20 | 0.8956 | 96/116 (82.8%) | 1.47 |

Two conclusions, of very different strength.

**The 1.5.13 ranking rework beats shipped 1.5.12, and that difference is
supported.** Per-question paired comparison of the two runs:

| Test | Counts | Exact two-sided p |
| --- | --- | --- |
| McNemar on recall@1 | baseline-only 1, candidate-only 12 | **0.0034** |
| Sign test on evidence rank | candidate better 16, worse 3, tied 97 | **0.0044** |

Both are exact (binomial) rather than asymptotic, because the discordant counts
are small. Both remain below 0.05 after a Bonferroni correction for the two
tests. On this 116-question sample the candidate's advantage over shipped
v1.5.12 is a **statistically significant improvement**, not merely a directional
one.

Three claim strengths are used deliberately in this document and should not be
read as interchangeable:

- **Statistically significant improvement** — the candidate over v1.5.12, above.
  This is the only comparison here that earns the phrase.
- **Measured directional improvement** — a difference visible in the aggregates
  with no paired test supporting it. The recency sweep's interior is at most
  this, and mostly not even that.
- **Non-regression** — configurations that score identically, so the evidence
  shows no harm and nothing more. The 1.7/3.5/5.0 cross-encoder scales are
  non-regressions with respect to each other.

**The interior of the sweep is not separable.** 0.02, 0.05 and 0.10 differ by at
most 2 questions out of 116, against 12 discordant questions for the comparison
that did reach significance. No paired test was run on these pairs; the point is
that a 2-question gap is the size of difference this sample cannot resolve, so
none of them is claimed as an improvement over another. The sweep is shaped
like an inverted U, and the shape is consistent with the mechanism — some
recency helps, too much starts overriding relevance — but this sample cannot
establish the peak's location. What it does support is a bound: the two largest
settings are the two worst-scoring, so `RECENCY_MAX_ADDITIVE` should not be
pushed into that region without stronger evidence.

The benchmark therefore constrains the constant rather than selecting it. Within
the range it does not distinguish, the value is fixed by the behavioural
invariant the release must satisfy: among semantically equivalent records, the
fresher one must rank first. 0.05 satisfies that invariant end-to-end under both
supported rerankers, with margin, and sits at the low end of the region the
sweep declines to separate.

**Alternatives considered.** 0.0 is the defective behaviour under repair: it is
the setting that lets a 60-day-old record outrank an equivalent fresh one, so it
is excluded on the invariant regardless of its score. 0.02 satisfies the
invariant with less margin against reranker noise. 0.10 and 0.20 are the two
lowest-scoring settings in the table and buy no measured benefit.

## Result: cross-encoder confidence

### Why the scale must be per model

The reranker boost is driven by min-max normalised cross-encoder scores, which
map the best candidate to 1.0 and the worst to 0.0 *however close they are*. For
an effectively tied candidate set that manufactures a full-strength boost out of
noise. That is how a negligible reranker preference could override a 60-day age
difference, and it is why no bounded recency term could satisfy the invariant on
its own.

The fix scales the boost by the observed spread. `cross_encoder_score` returns
raw logits, whose range is a property of the model, so the scale cannot be one
global constant. Measured over five tied pairs and five genuinely separated
pairs per model, on the same pairs for both:

| Reranker | max tied spread | min separated spread | ratio |
| --- | --- | --- | --- |
| `ms-marco-MiniLM-L-6-v2` | 0.04495 | 0.71181 | 15.8x |
| `bge-reranker-v2-m3` | 0.38548 | 2.48673 | 6.5x |

Each model separates its own two regimes cleanly, and the two models' regimes
are an order of magnitude apart from each other. A bare threshold could still be
placed between bge's largest tie (0.385) and ms-marco's smallest genuine
separation (0.712) — but the confidence factor is not a threshold. It is the
ratio `spread / full_scale`, so a single global `F` would have to satisfy both
`0.38548 / F < 0.227` (a bge tie must not decide the ranking) and
`0.71181 / F > 0.5` (an ms-marco separation must still count), that is
`F > 1.698` and `F < 1.424` simultaneously. No such `F` exists. The two
requirements are only 19% apart, which is why the single constant 1.0 looked
plausible while silently failing bge.

**Two scale-free transforms were tested and rejected**, rather than assumed to
work:

| Transform | ms-marco tied / separated | bge tied / separated | Verdict |
| --- | --- | --- | --- |
| sigmoid probability difference | 8.26e-6 / 8.84e-5 | 2.61e-3 / 1.28e-2 | **Rejected.** The regimes invert: bge's tied spread exceeds ms-marco's genuinely separated spread by 29x. No threshold separates them. |
| spread relative to larger score | 0.00521 / 0.07596 | 0.07439 / 0.36861 | **Rejected.** Only a 2% window between bge-tied 0.07439 and ms-marco-separated 0.07596, and the same ratio constraint as above has no solution. |

Sigmoid is the obvious candidate and it is the worst of the three: squashing
logits that are already deep in the saturated tail compresses exactly the
distinctions the boost depends on, and does so by a different amount per model.

### Choosing the per-model value

Two constraints bound the scale for each model:

- a tied pair must earn confidence below `_CE_DECISIVE_CONFIDENCE = 0.227`, the
  point at which a top-3 rerank boost (`ce_w` 0.15) overpowers the recency span
  (`RECENCY_MAX_ADDITIVE * (1 - decay floor) = 0.0425`);
- a genuinely separated pair must earn more than 0.5, so the reranker keeps its
  authority where it has a real preference.

| Reranker | evidence window | chosen | tied confidence | separated confidence |
| --- | --- | --- | --- | --- |
| `ms-marco-MiniLM-L-6-v2` | (0.198, 1.424) | 1.0 | 0.045 | 0.712 |
| `bge-reranker-v2-m3` | (1.698, 4.973) | 3.5 | 0.110 | 0.710 |

Both chosen values sit inside their window, deliberately toward the
recency-favouring end. 3.5 realises the `_CE_TARGET_TIED_CONFIDENCE = 0.11`
design target for bge almost exactly (0.38548 / 0.11 = 3.504); 1.0 is a round
value that holds a tied ms-marco pair an order of magnitude below it.

**An unmeasured reranker gets no boost at all.** `_ce_full_scale` returns `None`
for a model not in the table, which zeroes the confidence factor. Guessing a
scale for an unmeasured model is the precise failure this calibration exists to
prevent, and `test_every_supported_model_is_calibrated` fails if a selectable
reranker is ever added without a measured entry.

### Benchmark effect

`RECENCY_MAX_ADDITIVE` held at 0.05; the resolved reranker is
`bge-reranker-v2-m3`, so these values are in bge units. `0` disables the
magnitude boost entirely.

| `full_scale` | MRR | recall@1 | mean rank |
| --- | --- | --- | --- |
| 0 (boost disabled) | 0.8811 | 95/116 (81.9%) | 1.58 |
| 1.7 (window floor) | 0.9085 | 99/116 (85.3%) | 1.41 |
| **3.5 (chosen)** | **0.9085** | **99/116 (85.3%)** | **1.41** |
| 5.0 (just above window ceiling) | 0.9085 | 99/116 (85.3%) | 1.41 |

Two things follow, and the second corrects the first version of this document.

**Disabling the boost measurably degrades retrieval**, by 4 questions of
recall@1 and 0.027 MRR. By this document's own taxonomy that is a *measured
directional* result, not a significant one: it is a single run with no paired
test, and the paired test is one rerun away now that `--retrieval-log` exists.
It is not claimed at any greater strength than that. What it does settle is that
the earlier "benchmark-neutral" framing was wrong — whatever neutrality exists
holds *within* the evidence window, not at zero.

**Inside the window the benchmark is flat.** 1.7, 3.5 and 5.0 are
indistinguishable — identical to four decimal places on every metric. That is
the expected result, because LongMemEval's candidate sets are large and their
cross-encoder scores genuinely spread out, so confidence saturates near 1.0 at
any of these scales and the reranker keeps full authority. The benchmark
therefore confirms that the window is safe but cannot choose within it. The
choice is made by the pair measurements above and by the behavioural invariant,
which is what the degenerate near-tie case — absent from this benchmark — turns
on.

## The original `test_fresh_ranks_above_old` fixture

The first version of this document treated the original fixture as a decay test
that the release had made too strict. It was never a decay test.

Its two records were "…optimizes read performance" (fresh) and "…provides fast
lookup" (60 days old). The two supported rerankers **disagree on which is the
better match for the query**, and both do so decisively within their own scale:

| Reranker | prefers | by |
| --- | --- | --- |
| `ms-marco-MiniLM-L-6-v2` | "provides fast lookup" (the aged record) | 0.712 |
| `bge-reranker-v2-m3` | "optimizes read performance" (the fresh record) | 2.487 |

So the fixture's outcome turned on which reranker happened to be installed.
Under bge it passes with recency switched off entirely — age is irrelevant to
the result. Under ms-marco it fails at 0.17 and passes at 0.18, because that is
where a bounded recency term becomes large enough to overturn a genuine
relevance difference. Making that fixture pass under ms-marco is not a recency
requirement; it is a request to let age outrank relevance.

The fixture was replaced with a pair that differs only by a trailing reference
token, which both models place in their tied regime: 0.007 for ms-marco and
0.267 for bge in raw logit units, 0.228 for bge as production scores it with the
date prepended. An earlier version of this sentence gave bge as 0.073, which is
its *relative*-spread figure, not the raw-logit units every other number in this
document uses. Its
converse — a genuinely better older match must not be buried by age — is now
asserted deliberately in `test_stronger_match_wins_even_when_older`.

## MemoryStress

The specification also names MemoryStress. It is **deferred, not passed**. The
reasons recorded in the first amendment to Quality Gate 4 were factually wrong
and have been corrected in a second amendment; both live with the release
specification, which is not public.

The short form: the generator and the driver both exist and both run, and a
dataset was generated offline and deterministically for this release. Grading is
blocked because every configured LLM provider is out of billing or quota.
Separately, MemoryStress buckets `recall_at_age` by session index rather than
calendar time, and its OMEGA adapter writes the simulated date to
`referenced_date` only — so even fully graded it would measure retrieval under
scale and interference, not time decay.

## Honest limits of this evidence

- The recency sweep separates only its endpoints from its interior, and even
  that separation is not statistically supported. The chosen value rests on a
  benchmark-bounded range plus a behavioural invariant, not on a benchmark
  optimum.
- 116 questions is a small sample. Differences of one or two questions are not
  evidence of anything.
- Retrieval-only evaluation measures ranking, which is the right target here,
  but it does not measure end-to-end answer quality.
- The per-model cross-encoder scales rest on ten pairs per model, built around a
  single query. They are enough to establish that the two regimes are separable
  and that a global constant cannot serve both; they are not a broad
  characterisation of either model.
- The cross-encoder sweep exercises only the resolved reranker. The ms-marco
  scale is supported by pair measurements and by the ranking regression suite
  under that model, not by a LongMemEval sweep.
- MemoryStress is deferred, so the age-bucketed recall evidence the
  specification asks for is missing — though as noted above, that metric would
  not have measured time decay in any case.

## Gate for future ranking changes

Any change to scoring, ranking, or these constants must:

1. run the **complete** Core test suite with the embedding and reranker models
   available, not a targeted subset. The 1.5.13 recency regression was visible
   in a single full-suite run and was missed because only targeted files were
   run on the release branch;
2. run `tests/test_ranking_regression.py` under **every** supported reranker,
   not just the resolved default. Several of the defects corrected here were
   invisible under one model and obvious under the other;
3. confirm before quoting any benchmark number that the harness can observe the
   signal being claimed. The first version of this document failed this check,
   and every conclusion downstream of it was unsound.

CI already runs the full suite and lint for pull requests into `main`; release
branches must meet the same bar before qualification.
