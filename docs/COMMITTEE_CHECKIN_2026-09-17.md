# Committee check-in draft

**Draft for Yu to review and send; not sent by the assistant.** No committee response or
approval is implied. Supporting brief: `docs/SECOND_DATASET_DECISION.md`.

Subject: CSS700 pilot complete - second dataset and SOM2IMG implementation decisions

Hello,

The Breast Cancer Wisconsin pilot now includes three-seed Grid CNN experiments with
Grad-CAM, Integrated Gradients and Occlusion tracing, deletion checks, three-seed RF/XGBoost
baselines and TreeSHAP, and the seed-42 LIME test pilot with three sampling seeds.

The latest finding is that LIME–SHAP rank agreement can coexist with sizeable surrogate
approximation errors. We retained those errors and all poor-fit cases rather than treating
agreement as fidelity. The report is `docs/logs/2026-09-17-lime-test-pilot.md`.

To begin the two-dataset, two-encoding phase, could we confirm:

1. **Second dataset:** May I audit UCI Spambase as a numeric binary benchmark outside medicine?
   I would first pin the official release and audit missing values, duplicates/groups and
   feature semantics, then freeze preprocessing/splits before modeling. If multiclass or
   mixed categorical data is required instead, please identify that scope preference now.
2. **SOM2IMG source:** Which exact manuscript version and code commit should define the
   thesis implementation? Could you share the repository/archive, environment and settings?
   I found the author-listed BIBM 2025 paper, *Exploring Tabular-to-Image Algorithms for
   Applying CNNs to Tabular Data*, but have not established that its implementation defines
   the SOM2IMG method used in our local manuscript.
3. **Collisions and comparison controls:** How should multiple features selecting one SOM
   pixel be rendered and traced? Our provisional native-BMU check assigned 30 features to
   22 pixels. Should the matched comparison retain our StandardScaler/SmallCNN, or use
   another shared scaler/architecture? Please also confirm grid orientation, SOM/layout seed,
   initialization, training iterations and learning/neighborhood schedule. I will rerun a
   matched Grid control if those choices change. I will not silently overwrite, average or
   relocate colliding features and call it the author method.

Only three of the expected 12 CNN conditions are currently complete. These decisions are
the main dependencies for the full matrix; the completed pilot is not the full thesis.
Could we also confirm the next progress meeting and target defense date?

Thank you,
Yu

## Source lead, not an implementation specification

The [author's publication page](https://faculty.washington.edu/kimw6/publication.html)
lists the BIBM paper. The [institutional publication record](https://experts.arizona.edu/en/publications/exploring-tabular-to-image-algorithms-for-applying-cnns-to-tabula/)
gives DOI `10.1109/BIBM66473.2025.11356899` and pages 5066–5073. Its existence does not verify
a collision policy, code availability or equivalence to either SOM2IMG manuscript version.
