# Dice Metric Path Map (Post-Migration)

This document lists the canonical Dice metric paths after migration completion.

## Core Metric Computation

- `src/learning/eval_bbox_dataset.py`
  - `DestVolBuffer.get_dice_pred()` computes per-bbox Dice.
  - `_compute_filtered_mean_dice_score(...)` applies:
    - per-class Dice,
    - ground-truth-only rare-class filtering,
    - inclusive threshold (`count >= max_count / 100`),
    - unweighted mean over kept classes.
  - `compute_volume_weighted_mean_score(...)` computes weighted mean across bboxes.

## Validation Aggregation

- `src/learning/model_training.py`
  - `LearningValidationEvalResult.weighted_mean_dice`
  - `LearningValidationEvalResult.per_box_dice_by_box_id`
  - Validation loop reads `buffer.get_dice_pred()`.
  - Global metric uses bbox-volume weighting.

## Best-Epoch Selection

- `src/learning/model_training.py`
  - `LearningTrainingLoopResult.best_weighted_mean_dice`
  - Best epoch selection compares `weighted_mean_dice` (higher is better).
  - `best_epoch` is 1-based.

## User-Facing Output

- `src/ui/main_window.py`
  - Training completion message: `best weighted dice`.
  - Background log key: `best_weighted_dice`.
- `README.md`
  - Best checkpoint wording reflects validation weighted Dice.

