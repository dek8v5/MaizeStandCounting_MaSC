# MaSC results

The archived reference outputs and evaluation material are hosted in the [MaSC Zenodo archive](https://doi.org/10.5281/zenodo.22219350).

## Contents

The results deposit contains outputs from mosaic-mode and raw-frame processing, including row-wise stand-count tables and diagnostic products. Depending on the archived run, these include:

- `prediction_on_each_row.csv`: predicted count for each row.
- `rows_coordinate_and_count.mat`: row boundaries and counts in the MATLAB variable `rowsz`.
- Detection, range, row, and final-count overlay images.
- Evaluation inputs or summaries used to compare predictions with consensus manual counts.

## Download

1. Open the Zenodo link above and download the results archive.
2. From the repository root, extract it here:

   ```bash
   unzip <downloaded-results-archive>.zip -d results/
   find results -type f | sort
   ```

Keep new computations in separate directories such as `results/mosaic_run/` and `results/raw_run/` so the downloaded reference outputs are not overwritten. See the root [README](../README.md#outputs-and-quality-control-checks) for output definitions and validation checks.
