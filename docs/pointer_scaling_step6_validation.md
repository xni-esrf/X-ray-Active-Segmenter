# Step 6 - Multi-Client Validation Runbook

Goal: validate the pointer/cursor fix on at least two local client machines using the same remote node/container/commit.

Prerequisite:
- Complete the interaction sequence from [pointer_scaling_baseline_checklist.md](/home/laugros/codedir/X-ray-Active-Segmenter/docs/pointer_scaling_baseline_checklist.md)

## 1) Capture machine/session environment (each client)
From repo root:

```bash
source .venv/bin/activate
bash tools/capture_pointer_env.sh logs/clientA_env.txt
```

Repeat on the second client with `clientB_env.txt`.

## 2) Launch app and capture full logs (each client)
Use the same startup command and dataset on both clients.

```bash
source .venv/bin/activate
mkdir -p logs
python open_ui_raw_viewer.py --log-level DEBUG --load-mode lazy <same_volume_path> \
  2>&1 | tee logs/clientA_app.log
```

Repeat on the second client with `clientB_app.log`.

## 3) Execute manual interaction checks
Run the same UI actions on each client:
- right-click pan (no Shift)
- right-click center (Shift)
- left-click pick alignment
- optional bbox handle hover/drag alignment
- repeat on axial/coronal/sagittal

## 4) Extract diagnostics from logs

```bash
bash tools/extract_pointer_diagnostics.sh logs/clientA_app.log
bash tools/extract_pointer_diagnostics.sh logs/clientB_app.log
```

Save outputs, or redirect to files:

```bash
bash tools/extract_pointer_diagnostics.sh logs/clientA_app.log > logs/clientA_diag.txt
bash tools/extract_pointer_diagnostics.sh logs/clientB_app.log > logs/clientB_diag.txt
```

## 5) Acceptance criteria
- Pointer mapping logs are present on both clients.
- No systematic pointer offset is observed in all 3 views.
- Right-click navigation follows cursor on both clients.
- No regression in bbox/annotation pointer interaction.

## 6) If one client still fails
Re-test quickly with legacy compatibility mode:

```bash
source .venv/bin/activate
XRA_USE_LEGACY_POINTER_SCALE=1 python open_ui_raw_viewer.py --log-level DEBUG --load-mode lazy <same_volume_path>
```

Then compare behavior and diagnostics again. This isolates whether failure is tied to client DPI interpretation.

Note:
- `XRA_USE_LEGACY_POINTER_SCALE` is temporary and should be treated as a short-term fallback during the validation cycle.
