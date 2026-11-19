# Freezing the pre-registration

Run these steps once, when `PREREGISTRATION.md`, `configs/`, dev/test split lists and null-calibrated thresholds are final.
(Tag once the dev results of stages 0 and A have fixed the thresholds; use `prereg-v5` for a
re-freeze instead of moving `prereg-v4`.)

```bash
cd /path/to/sensorfault
git status                      # must be clean: everything below is committed
pytest -q tests/core
git tag -a prereg-v4 -m "Pre-registration v1: hypotheses, thresholds, split, tuning protocol"
git push origin prereg-v4       # optional: public timestamp (also archive e.g. on OSF/Zenodo)
git rev-parse prereg-v4         # record this hash in the paper
```

Rules
* Never move or delete a pushed tag. Changes after tagging = new commit + dated row in `## Deviations` (+ `prereg-v5` if thresholds change).
* Run test-split scenes only after the tag exists: `python -m sensorfault.runner --config configs/stage_a_depth_only_honesty.yaml --split test`
  (the runner checks `git tag -l prereg-v4`; `--allow-untagged` exists for debugging only and the run manifest records
  `prereg_tag_present`).
* Dev-split work (tuning, ablations, `--dry-run`) is allowed before tagging.
* In the paper, cite the tag hash and list every deviation.
