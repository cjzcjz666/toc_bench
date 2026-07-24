# TOC-Bench bootstrap confidence intervals

These files report statistical uncertainty for the full 2,323-item TOC-Bench test set evaluated by 23 models.

## Method

- Unit: QA item.
- Interval: two-sided 95% percentile confidence interval.
- Resamples: 20,000, with fixed seed 20260724.
- Scoring: exact match; invalid or missing extractions remain in the denominator and count as incorrect.
- For a cell containing `n` binary correctness observations with empirical accuracy `p`, resampling the observations with replacement is distribution-equivalent to drawing the number correct from `Binomial(n, p)`.
- Direct model comparisons use paired QA-item resampling of the per-item correctness differences. Marginal confidence-interval overlap is not used as a significance test.

## Files

| File | Coverage |
|---|---|
| `overall_ci.csv` | 23 model-level Overall results |
| `dimension_ci.csv` | all 23×10 model-by-dimension cells |
| `format_ci.csv` | all 23×5 model-by-serialization-label cells |
| `tier_ci.csv` | all 23×3 model-by-tier cells |
| `hallucination_ci.csv` | HDA, HDA-Subject, and HDA-Event for all models |
| `paired_top_models_ci.csv` | paired differences among the highest-overall models |
| `subset_counts.csv` | exact denominators for dimensions, formats, tiers, and hallucination subsets |
| `run_metadata.json` | scope, seed, resampling, and scoring metadata |

TOC-Bench has four conceptual task formats. The five released serialization labels arise because three-event and four-event ordering are stored separately as `ordering_3` and `ordering_4`.
