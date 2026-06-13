# Film Scan Frame Splitter

A local Streamlit tool for splitting one uncut scanned film strip into individual full-resolution photo files.

The app detects likely film frames, shows editable crop rectangles, and exports the crops as a ZIP. Preview images are downscaled for speed, but every exported frame is cropped from the original uploaded image.

## Features

- Upload JPG, PNG, TIFF, and TIF scans
- Detect horizontal or vertical film strips
- Detect parallel two-column, three-column, or four-column scan layouts with the `Parallel lanes` setting
- Enforce exact 3:2 or 2:3 crops with the `Regular 3:2 grid` method
- Use black film borders and repeated separator gaps with the `Black border pitch` method
- Estimate frame crop boxes with OpenCV thresholding, edge detection, contours, and projection-profile fallback
- Preview detected boxes over a downscaled display copy
- Manually edit, add, delete, and reorder crop boxes in a table
- Export full-resolution crops as TIFF, PNG, or high-quality JPEG
- Large upload support configured up to 2 GB
- Optional border trimming
- Optional color negative inversion
- Optional basic negative color normalization
- Preserves DPI, ICC profile, and JPEG EXIF where Pillow can carry them through

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints, usually:

```text
http://localhost:8501
```

## Workflow

1. Upload one full-resolution film-strip scan.
2. Adjust detection settings if needed.
   - Use `Parallel lanes = 2` for two side-by-side film strips or contact-sheet-style scans.
   - Use `Regular 3:2 grid` when all frames have the same 3:2 shape and consistent spacing.
   - Use `Black border pitch` when black film borders and separator gaps are the clearest signal.
   - Leave `Frames per lane` at `0` when the frame gaps vary; set a number only when you want to force the count.
   - Use `Parallel lanes = Auto` for mixed batches; the app detects two-lane scans and single-lane scans from the white gutter.
3. Review the detected rectangles in the preview.
4. Edit the crop table:
   - Change `x`, `y`, `width`, and `height` to move or resize boxes.
   - Set `enabled` to false to skip a box.
   - Change `order` to reorder exports.
   - Add a row to create a new crop.
5. Choose an export format.
6. Download the ZIP.

## Quality Notes

- Detection and preview use a resized copy only for speed.
- Export always crops from the original uploaded image.
- ZIP export is generated only when you click `Generate ZIP`.
- The generated ZIP is saved to `outputs/film_frames.zip`, which is safer for very large TIFF scans.
- TIFF is the default because it is lossless and best for preserving scan quality.
- JPEG export uses quality 100 and disables chroma subsampling.
- PNG and TIFF are lossless export choices.
- Optional corrections are off by default so the exported crops preserve the scan as closely as possible.

## Limitations

Automatic frame detection is heuristic. Difficult scans with very low contrast, heavy dust, severe skew, unusual masks, or overlapping frame areas may need manual crop edits. Streamlit does not provide native drag handles on images, so this prototype uses a precise editable crop table rather than direct mouse dragging.

For 500 MB TIFF scans, expect high RAM usage because TIFF decoding, preview generation, and full-resolution export all require Pillow to read large image data. Use lossless TIFF or PNG export when quality matters most, and avoid enabling deskew unless needed because deskew creates a second full-resolution working image.
