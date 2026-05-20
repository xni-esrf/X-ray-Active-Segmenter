# Pointer/Scaling Baseline Checklist (Step 1)

Goal: reproduce and compare the cursor-size and pointer-offset issue across client machines before any code fix.

## Scope
- Same remote node
- Same container/image
- Same git commit
- Same dataset
- Different local client machine (where UI is displayed)

## Pre-run Capture (per client machine)
Run these commands in the same shell used to launch the app:

```bash
date -Is
hostname
git rev-parse --short HEAD
python -V
echo "DISPLAY=$DISPLAY"
echo "WAYLAND_DISPLAY=$WAYLAND_DISPLAY"
echo "XDG_SESSION_TYPE=$XDG_SESSION_TYPE"
echo "QT_QPA_PLATFORM=$QT_QPA_PLATFORM"
echo "QT_SCALE_FACTOR=$QT_SCALE_FACTOR"
echo "QT_AUTO_SCREEN_SCALE_FACTOR=$QT_AUTO_SCREEN_SCALE_FACTOR"
echo "QT_SCREEN_SCALE_FACTORS=$QT_SCREEN_SCALE_FACTORS"
echo "GDK_SCALE=$GDK_SCALE"
echo "XCURSOR_SIZE=$XCURSOR_SIZE"
```

## Launch Command (per client machine)
Use the exact same command on both machines:

```bash
source .venv/bin/activate && \
python open_ui_raw_viewer.py --log-level DEBUG --load-mode lazy <same_volume_path>
```

If needed, include optional startup files:

```bash
source .venv/bin/activate && \
python open_ui_raw_viewer.py \
  <same_volume_path> \
  --semantic <same_semantic_path> \
  --instance <same_instance_path> \
  --bbox <same_bbox_path> \
  --log-level DEBUG \
  --load-mode lazy
```

## Repro Interaction Sequence (per client machine)
1. Open the same orthogonal view (Axial first).
2. Move mouse to a clearly identifiable pixel/feature.
3. Right-click drag in `pan` mode (no Shift) and confirm whether panning follows cursor.
4. Right-click drag in `center` mode (with Shift) and confirm whether cursor-centering is accurate.
5. Left-click on a known feature and check if crosshair/selection lands exactly under cursor.
6. If bbox mode is enabled: hover box handles and test drag start accuracy.
7. Repeat steps 2-6 in Coronal and Sagittal views.

## Expected vs Observed
Expected:
- Cursor visual size is usable/consistent.
- Click/drag target matches visible cursor location.
- No systematic offset (left/down shift, etc.).

Observed (record):
- Cursor size impression (`normal`, `too small`, `too large`).
- Offset direction (`none`, `left`, `right`, `up`, `down`).
- Approximate offset magnitude (in screen pixels).
- Whether issue is constant or intermittent.

## Results Table (fill manually)
| Field | Machine A (works) | Machine B (fails) |
|---|---|---|
| Date/time |  |  |
| Local OS + DE/WM |  |  |
| Session type (X11/Wayland) |  |  |
| Display scaling setting |  |  |
| Commit hash |  |  |
| Launch command |  |  |
| Cursor size |  |  |
| Offset direction |  |  |
| Offset magnitude (px) |  |  |
| Affected views (A/C/S) |  |  |
| Intermittent or constant |  |  |
| Notes |  |  |

## Log Snippets to Keep
Save startup logs containing:
- OpenGL context line (`vendor`, `renderer`, `version`)
- Any Qt/VisPy warnings around scaling, canvas, or backend

You will use these baseline results to validate Step 2 (source fix) and confirm no regression.
