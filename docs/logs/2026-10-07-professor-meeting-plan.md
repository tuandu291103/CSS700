# Session summary on the professor meeting and revised project plan

Date: 2026-10-07, America/Los_Angeles.
Outcome: source-backed draft plan for Autumn 2026, Winter 2027 and Spring 2027.

## What we did

- Reviewed the professor's questions against the existing design and pilot evidence.
- Identified the supplied IEEE link as SOM2IMG through the author's bibliography. The final
  IEEE full text was inaccessible; retained the separately supplied six-page manuscript as
  a clearly versioned source, not a verified final-paper reproduction.
- Verified UCI dataset counts and proposed Breast Cancer Wisconsin, Spambase and Dry Bean.
  All three appear in the local manuscript. Flagged the 13,611 versus 13,612 Dry Bean discrepancy.
- Reviewed primary sources for DeepInsight, IGTD, PermGrad, versioned Table2Image and saliency
  evaluation. Found a direct prior tracing framework in PermGrad, so no first-framework claim
  is made. Table2Image's latest version changes its title and emphasis; version-specific claims
  are preserved.
- Wrote a [revised plan](../PROJECT_PLAN_2026_2027_REVISED.md) with three primary encoders,
  three explanation methods, random-layout controls, quantitative evaluation, decision gates,
  a resource-counted experiment matrix and a quarter-by-quarter writing schedule.
- Created a [shareable Word version](../deliverables/CSS700_Project_Plan_2026_2027.docx).
  The document skill informed a plain academic layout and page-by-page render verification.
- Updated README and planning entry points while preserving original proposal and historical
  experiment protocols.

## Research conclusion

Tabular-to-image conversion does not automatically improve or destroy interpretability.
Unique value-preserving mappings can retain inputs; collisions can destroy identifiability.
Even an exact inverse coordinate map does not make a regional heatmap an exact individual
feature contribution. The defensible proposed contribution is auditable tracing and controlled
evaluation, with novelty still to be established against prior work.

Existing Spambase replacement-sensitive deletion effects and low Grad-CAM/Occlusion agreement
motivate the research but are not a demonstrated solution. The expanded study remains incomplete.

## Verification and boundaries

- Cross-checked primary source URLs, dataset dimensions, local manuscript table and pilot logs.
- Rendered the Word plan and inspected every final page; source Markdown remains editable.
- Documentation-only change: no new datasets acquired, model training, evaluation, test-set
  access, protocol execution, or external advisor message. No committee approval is implied.
- The previous coding milestone reported 284 tests passing; not rerun for this documentation
  session. Git whitespace and document-structure checks are separate from research tests.
- Code/document push does not back up ignored raw datasets, checkpoints or experiment artifacts.

## Next step

Review the plan with the advisor; obtain final SOM2IMG PDF/code; then build the three-dataset
readiness package beginning with Dry Bean acquisition and audit. Resolve mapping policies and
input-gradient numerical preflights before freezing and launching the new comparative matrix.
