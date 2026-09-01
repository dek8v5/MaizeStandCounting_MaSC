# MaSC data

The imagery and supporting inputs are hosted in the [MaSC Zenodo archive](https://doi.org/10.5281/zenodo.22219350) because they are too large for this Git repository.

## Contents

The data deposit contains the inputs used by the two MaSC workflows:

- `mosaic_mode/`: the RGB mosaic and supporting files used for mosaic-mode counting.
- `raw_mode/raw/`: the deposited 77-frame PNG segment used for raw-frame processing.
- Raw-mode registration support, including the homography matrix CSV and common-coordinate mosaic needed to align frame detections.

The deposited 77-frame segment is not the same inventory as the manuscript's separate 83-frame timing experiment.

## Download

1. Open the Zenodo link above and download the data archive.
2. From the repository root, extract it here while preserving its directory structure:

   ```bash
   unzip <downloaded-data-archive>.zip -d data/
   ```

3. Confirm the extracted inventory:

   ```bash
   find data -type f | sort
   find data/raw_mode/raw -maxdepth 1 -type f -iname '*.png' | wc -l
   ```

Do not rename or reorder the raw frames: their lexical order must remain synchronized with the homography matrices and YOLO label files. See the root [README](../README.md#masc-workflow) for the complete workflows.
