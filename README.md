## 3D Volume Segmentation Viewer

This project provides a lightweight Python UI for exploring large 3D volumetric datasets and their segmentation maps with synchronized orthogonal views.

### Features
- CLI entrypoint to launch the viewer.
- Load `.npy/.npz`, `.tif/.tiff`, `.zarr`, and `.h5/.hdf5` volumes.
- Basic renderer API with view synchronization scaffolding.
- Annotation tool shortcuts: `Ctrl+B` (brush), `Ctrl+E` (eraser), `Ctrl+F` (flood fill).
- Annotation shortcuts auto-enable `Manual Segmentation` when needed.
- Bounding-box table supports multi-selection (`Ctrl` / `Shift`) for batch label update and delete.
- Selected-bbox segmentation processing from the bbox panel:
  - `Median Filter Selected` (binary 3x3x3 median, threshold 14/27)
  - `Erosion Selected` (binary 3x3x3 erosion, 1 voxel)
  - `Dilation Selected` (binary 3x3x3 dilation, 1 voxel)
- Selected-bbox processing is constrained to the union of selected boxes, runs in the foreground, and is recorded as one undo/redo transaction.

### Setup
Install dependencies:

```
pip install -r requirements.txt
```

### Usage
Run the viewer with an optional volume path:

```
python open_ui_raw_viewer.py /path/to/volume
```

Load optional segmentation maps and bounding boxes at startup:

```
python open_ui_raw_viewer.py /path/to/raw --semantic /path/to/semantic --instance /path/to/instance --bbox /path/to/boxes.bbox.txt
```
