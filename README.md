# Complete addressing of convective PCR fluorescence arrays

This repository accompanies our study of temporal aggregation and joint image–geometry registration for fluorescence-array addressing in convective PCR. The method combines a temporally aggregated reference image, background and contrast correction, YOLO spot detection, and bounded lattice registration to infer the full row–column address grid, including weak sites without matched detector candidates.

## Run one 10 × 10 example in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ShuoYang-buct/convective-pcr-array-addressing/blob/main/Run_in_Colab.ipynb)

Click the button, then choose **Runtime → Run all**, or run the cells from top to bottom. The notebook automatically installs dependencies, downloads and checks the paper model, and loads one example sequence (**N008**, 10 × 10).

Each stage is displayed:

1. Original acquired image.
2. Temporal maximum projection of ten selected raw frames.
3. Background subtraction and contrast enhancement.
4. YOLO detection boxes from a new model inference.
5. Filtered and size-normalized candidate boxes.
6. Complete 100-address grid: candidate-supported boxes in green and inferred boxes in orange.
7. Downloadable stage images, final overlay, address table and QC summary.

The default demonstration runs the actual YOLO model. A CPU works; a GPU is optional. A complete grid indicates that all specified addresses are output, while the printed QC status reports whether the grid meets the automatic acceptance checks.

## Example output

![N008 complete grid](docs/example_result.png)

The reference run retains 76 candidate-supported addresses and infers 24 additional addresses, giving the full 100-position grid. Green and orange distinguish these two groups.

## Included data and code

The interactive demo uses only N008 (10 × 10). Its ten selected full-field frames are losslessly encoded PNGs with unchanged acquired grayscale pixels. The source acquisition contains 60 frames; the subset follows the paper's five-earlier/five-final-ten selection rule. Preprocessing runs on the full field before applying the fixed crop. Manual center coordinates are not required or loaded by the demonstration.

Additional data are available as a separate [34-sequence reference-data archive](https://github.com/ShuoYang-buct/convective-pcr-array-addressing/releases/download/v1.0.0/reference_data_34_sequences.zip). The archive contains all 34 enhanced reference images, saved detector candidates, layout dimensions and existing pointwise-confirmed manual centers (6704 addresses), with SHA-256 checksums. Downloading it is optional and does not change the default one-sequence Colab workflow. It does not contain all raw acquisition frames, training data or the private engineering application.

The research code is in `src/`; the Colab notebook exposes the processing steps. Model weights are distributed through the [v1.0.0 Release](https://github.com/ShuoYang-buct/convective-pcr-array-addressing/releases/tag/v1.0.0), with their SHA-256 recorded in `model_manifest.json`. Image materials were provided by TorusDx (Shanghai, China).

## Run locally

```bash
python -m pip install -r requirements.txt
python run_demo.py
```

Results are written to `outputs/demo/`.

After installing dependencies, `python run_demo.py` is the one-command demo.
It verifies all ten input-frame checksums and the paper-model checksum, downloads
the approximately 125 MB model from the existing Release only if it is missing,
then runs actual YOLO inference and lattice registration. For offline use, place
the published model in `weights/best.pt` and run `python run_demo.py --no-download`.
An alternate model path and output directory can be supplied with `--weights`
and `--output`. Input metadata supplies the fixed crop and known layout size;
manual center coordinates are not loaded by this inference.

The outputs include six stage figures, three image arrays saved as PNGs,
`addresses.csv`, `prediction.json`, `summary.json`, and `run_environment.json`.
The last file records dependency versions and input/model verification. Its wall
time includes process startup and figure writing and must not be compared with
the paper's 8.28 s median pipeline timing.

`environment.yml` provides a Python 3.11 Conda setup:
`conda env create -f environment.yml`, then `conda activate convective-pcr-addressing`.
`requirements.txt` gives compatible dependency ranges. `requirements-tested.txt`
records direct dependencies in the existing Windows CPU runtime used for local
verification; it is not a portable, freshly resolved lockfile. A clean Conda
installation and hosted Colab execution have not been repeated for this update.

## What this demonstration reproduces

N008 exercises temporal aggregation, full-field enhancement, fixed cropping,
new model inference, filtering and complete-grid addressing. The reference run
produces 100 addresses, with 76 supported by candidates and 24 inferred; QC is
`accept`. This demonstrates the processing chain for one existing sequence.
It does not establish clinical performance or a new prospective validation.

The separate 34-sequence archive supports retrospective reference-data analysis.
It contains enhanced reference images and existing manual coordinates, rather
than every original acquisition frame. It therefore cannot reproduce raw-frame
processing and timing for the entire paper. The manual references are existing
pointwise confirmations, not a newly collected independent annotation study.

## Licensing

Author-owned research code is available under **PolyForm Noncommercial 1.0.0**.
The unmodified official license text is in `LICENSE`; `LICENSE_SCOPE.md` lists
the covered source files and demo entry points. This is a noncommercial
source-available software license. For commercial permission, contact TorusDx
through the corresponding author.

Model weights, acquisition images, manual coordinates and other data are outside
this code-license grant. Their download availability does not extend this
software license to them. Third-party dependencies, including Ultralytics, retain
their own licenses; see `THIRD_PARTY_NOTICES.md`. This update makes no claim that
the complete dependency stack has a single license or that all dependency
licenses are compatible for every use or distribution.

