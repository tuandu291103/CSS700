# Full-study readiness: dataset recommendation and implementation decisions

Date: **2026-09-17**. This session is a source/readiness review, not a new model experiment.

## Brief summary

- Compared four public dataset candidates against the existing binary/named-feature/ranking
  contract. Recommend **Spambase for an integrity audit**, not automatic study selection.
- Created `docs/SECOND_DATASET_DECISION.md` with source links, risks, acceptance gates and
  the concrete changes needed in the dataset-specific experiment runners.
- Created `docs/COMMITTEE_CHECKIN_2026-09-17.md`, an unsent draft asking for dataset scope,
  exact SOM2IMG code/version, collision rules and matched comparison settings.
- Added six synthetic checks for larger padded grids and the unchanged CNN architecture.
  No candidate data, new trained model, explanation result or accepted CNN condition was added.

## What the source review established

[UCI Spambase](https://archive.ics.uci.edu/dataset/94/spambase) is the preferred audit
candidate on size, binary numeric scope and feature-count grounds. It is not ready by fiat:
the landing page's missing-value flag conflicts with the
[legacy documentation](https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.DOCUMENTATION).
Raw-release validation and duplicate/group checks must precede any split. The feature dictionary
also motivates training-only tests of integer-count and quartile-sampling semantics. No
candidate was ranked by its published predictive performance and no raw arrays were inspected.

The [author publication list](https://faculty.washington.edu/kimw6/publication.html) and
[institutional BIBM record](https://experts.arizona.edu/en/publications/exploring-tabular-to-image-algorithms-for-applying-cnns-to-tabula/)
provide a concrete publication lead for the committee question. They do not establish an
authoritative SOM2IMG implementation. Searches for SOM2IMG/code/GitHub and author/title
variants did not identify a confirmed repository; this is not proof none exists.

The PDF review workflow could not inspect the public OpenReview document fully because
the direct [PDF request](https://openreview.net/pdf?id=DU5m5X7yu8) and forum request returned
browser verification. Search-index excerpts were discovery leads only; no methods, current
publication status or collision policy were promoted from them. The previously reviewed local
manuscript and earlier collision diagnostic remain the existing evidence, unchanged. The
IEEE link from the author page also failed to open in this session. No access barrier was bypassed.

## Implementation readiness and verification

The metadata-only Grid checks give 8 x 8 with seven padding pixels for 57 features.
Synthetic tests cover 30/34/57/60 features, original-name order, signed/absolute tracing and
exclusion of arbitrarily large padding scores. Two more tests cover 6 x 6 and 8 x 8 CNN
forward/backward compatibility. No actual dataset loader or scientific configuration changed.

Code inspection identified real remaining work: direct Breast Cancer loader calls, pilot-only
source/count guards, and hard-coded medical display class names in LIME. The decision brief
requires dataset-aware entry points without weakening old reproducibility checks.

Validation: **98 tests passed in 22.77 seconds**, including the six new synthetic cases;
lint, formatting and whitespace checks are clean. Existing scientific source/config files
and saved research artifacts were not modified. These tests do not validate Spambase itself.

## On-track assessment and next action

Pilot milestones are complete; full-scope progress now depends on a dataset decision and
SOM2IMG source/mapping clarification. The matrix remains **3/12 CNN conditions**; this review
does not increase it or certify the defense timeline. Earlier timing and artifact-backup
limitations remain open. No external message, upload or Git push was performed.

If Yu authorizes the proposed audit, acquire/pin the official Spambase files and produce an
append-only integrity report without modeling. Review that report before freezing grouping,
split and preprocessing rules. The second-dataset Grid track can then advance independently
of SOM2IMG clarification, but accepted encoder-comparison work still requires the author or
committee decision. No additional LIME tuning is scheduled.
