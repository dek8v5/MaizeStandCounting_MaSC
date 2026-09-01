# MaSC model

The trained detector is available from the [MaSC Zenodo archive](https://doi.org/10.5281/zenodo.22219350).

## Contents

The model deposit contains the trained YOLOv9 weights used in the manuscript. The detector has three classes representing one, two, or three maize plants inside a bounding box. MaSC converts these classes to counts of one, two, and three plants, respectively.

## Download

1. Open the Zenodo link above and download the model archive.
2. From the repository root, extract it into this directory:

   ```bash
   unzip <downloaded-model-archive>.zip -d models/
   find models -type f \( -iname '*.pt' -o -iname '*.pth' \) | sort
   ```

3. Place the published `best.pt` at the runtime path used in the root instructions:

   ```bash
   mkdir -p yolov9_model
   cp models/<path-to-best.pt> yolov9_model/best.pt
   test -f yolov9_model/best.pt
   ```

Replace `<path-to-best.pt>` with the path shown by `find`. Keep the original downloaded file in `models/` as an unchanged reference copy. See the root [README](../README.md#one-time-code-and-model-preparation) for YOLOv9 setup and inference parameters.
