# Open questions for Yu and the committee

These do not block the initial Grid pilot, but they should be resolved before the related gate.

## Highest priority

1. **SOM2IMG implementation:** Do you already have access to the exact code, manuscript, and
   recommended hyperparameters from W. D. Selke / the committee? Reimplementation risk is the
   largest technical uncertainty in the proposal.
2. **Second dataset:** Should the second dataset test multiclass behavior, mixed feature types,
   or simply replicate the numerical binary setting? This choice changes preprocessing,
   attribution targets, and the strength of the conclusions.
3. **Compute:** Will experiments run on Apple Silicon, an NVIDIA GPU, a university server, or a
   cloud notebook? Determinism, package installation, and timing comparisons depend on this.
4. **Explanation target:** Is predicted-class attribution the committee's intended primary
   analysis, with true-class attribution added for error analysis?
5. **Feature ranking sign:** Should primary rankings use absolute attribution magnitude while
   retaining signed values for secondary analysis, or use positive support for the target only?

## Design decisions

6. What exact feature replacement should faithfulness and occlusion use in standardized space:
   zero/training mean, a conditional value, or more than one sensitivity setting?
7. Should Random Forest and XGBoost comparisons use built-in global importance only, or TreeSHAP
   for like-for-like local explanations as well?

   **Provisional implementation:** The first predictive baseline retains built-in importance only
   as a separately labeled global diagnostic. A local TreeSHAP comparison remains deferred until
   its class target, background/reference data, and committee scope are confirmed.
8. What threshold defines adequate CNN performance beyond beating the majority-class baseline?
9. Should sample-level uncertainty use bootstrap intervals, and does the program expect formal
   hypothesis tests or primarily estimation and effect sizes?
10. For SOM feature collisions, what mapping and aggregation behavior is authoritative?

## Project and writing

11. What is the exact Autumn start date and the committee's preferred progress-meeting cadence?
12. Does the program provide a required thesis Word/LaTeX template, citation style, page limit,
    or repository/data-retention policy?
13. Should the GitHub repository remain private through defense? Private is the safe default.
14. May the final code and derived artifacts be released publicly, and under what license?
15. The LM-IGTD citation appears incomplete on the proposal's reference page (it ends with
    “to leverage the potential of co”). What is the intended full bibliographic entry?
