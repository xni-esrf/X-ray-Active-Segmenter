from __future__ import annotations

from importlib import import_module
from typing import Dict

_EXPORT_MODULE_BY_NAME: Dict[str, str] = {
    "build_learning_dataloader_from_batch": ".dataloader_builder",
    "build_learning_dataloader_from_current_batch": ".dataloader_builder",
    "extract_train_tensor_pairs": ".dataloader_builder",
    "compute_class_weights_from_segmentation_tensors": ".class_weights",
    "compute_and_store_current_learning_class_weights": ".class_weights",
    "encode_target_labels": ".label_utils",
    "build_eval_dataloader_runtimes_from_batch": ".eval_dataloader_builder",
    "build_eval_dataloader_runtimes_from_current_batch": ".eval_dataloader_builder",
    "build_inference_dataloader_runtime_from_entry": ".eval_dataloader_builder",
    "build_inference_dataloader_runtimes_from_batch": ".eval_dataloader_builder",
    "compute_eval_label_values_from_batch": ".eval_dataloader_builder",
    "dispose_inference_runtime": ".eval_dataloader_builder",
    "dispose_inference_runtimes": ".eval_dataloader_builder",
    "LearningInferenceBackgroundResult": ".inference",
    "LearningInferencePrediction": ".inference",
    "LearningInferenceProgress": ".inference",
    "LearningInferenceStopRequested": ".inference",
    "apply_inference_predictions_to_array": ".inference",
    "run_learning_inference": ".inference",
    "derive_label_space_from_semantic_segmentation": ".label_space",
    "LearningLabelSpace": ".label_space",
    "LearningLabelCoverage": ".label_coverage",
    "compute_learning_label_coverage": ".label_coverage",
    "format_learning_label_coverage_warning": ".label_coverage",
    "DEFAULT_FOUNDATION_CHECKPOINT_PATH": ".model_instantiation",
    "DEFAULT_FOUNDATION_MODEL_CONFIG": ".model_instantiation",
    "FoundationCheckpointLoadPreconditions": ".model_instantiation",
    "FoundationCheckpointMetadata": ".model_instantiation",
    "FoundationModelConfig": ".model_instantiation",
    "FoundationInstantiationPreconditions": ".model_instantiation",
    "UnetrPs8MulticlassesSmall": ".model_instantiation",
    "Unetr_ps8_multiclasses_small": ".model_instantiation",
    "inspect_foundation_checkpoint_metadata": ".model_instantiation",
    "instantiate_foundation_model_runtime": ".model_instantiation",
    "save_foundation_model_checkpoint": ".model_instantiation",
    "validate_foundation_checkpoint_load_preconditions": ".model_instantiation",
    "validate_foundation_model_instantiation_preconditions": ".model_instantiation",
    "LearningTrainEpochResult": ".model_training",
    "LearningTrainingEpochProgress": ".model_training",
    "LearningTrainingLoopResult": ".model_training",
    "LearningTrainingPreconditions": ".model_training",
    "LearningValidationEvalResult": ".model_training",
    "evaluate_learning_model_on_validation_dataloaders": ".model_training",
    "train_learning_model_with_validation_loop": ".model_training",
    "train_learning_model_for_one_epoch": ".model_training",
    "validate_learning_model_training_preconditions": ".model_training",
    "DEFAULT_TRAINING_PARAMETERS": ".training_parameters",
    "TrainingParameters": ".training_parameters",
    "validate_training_parameters": ".training_parameters",
    "EvalBBoxDataset": ".eval_bbox_dataset",
    "DestVolBuffer": ".eval_bbox_dataset",
    "InferenceDestVolBuffer": ".eval_bbox_dataset",
    "TrainBBoxDataset": ".train_bbox_dataset",
    "LearningBBoxDataLoaderRuntime": ".session_store",
    "LearningBBoxEvalRuntime": ".session_store",
    "LearningSession": ".session_store",
    "LearningModelRuntime": ".session_store",
    "LearningBBoxTensorBatch": ".session_store",
    "LearningBBoxTensorEntry": ".session_store",
    "set_current_learning_label_space": ".session_store",
    "get_current_learning_label_space": ".session_store",
    "clear_current_learning_label_space": ".session_store",
    "set_current_learning_bbox_batch": ".session_store",
    "set_current_learning_bbox_entries": ".session_store",
    "get_current_learning_bbox_batch": ".session_store",
    "clear_current_learning_bbox_batch": ".session_store",
    "set_current_learning_dataloader_runtime": ".session_store",
    "set_current_learning_dataloader_components": ".session_store",
    "set_current_learning_dataloader_class_weights": ".session_store",
    "get_current_learning_dataloader_runtime": ".session_store",
    "clear_current_learning_dataloader_runtime": ".session_store",
    "set_current_learning_eval_runtimes_by_box_id": ".session_store",
    "set_current_learning_eval_runtime_components_by_box_id": ".session_store",
    "get_current_learning_eval_runtimes_by_box_id": ".session_store",
    "clear_current_learning_eval_runtimes_by_box_id": ".session_store",
    "set_current_learning_model_runtime": ".session_store",
    "set_current_learning_model_components": ".session_store",
    "get_current_learning_model_runtime": ".session_store",
    "clear_current_learning_model_runtime": ".session_store",
    "get_default_learning_session": ".session_store",
    "LearningSourceBundle": ".state_preparation",
    "LearningStatePreparationResult": ".state_preparation",
    "instantiate_model_runtime_from_checkpoint": ".state_preparation",
    "load_learning_sources_from_paths": ".state_preparation",
    "prepare_learning_state_from_sources": ".state_preparation",
    "prepare_learning_state_from_volumes": ".state_preparation",
    "semantic_label_space_source_signature": ".state_preparation",
    "validate_training_preconditions_for_session": ".state_preparation",
}

__all__ = list(_EXPORT_MODULE_BY_NAME)


def __getattr__(name: str) -> object:
    try:
        module_name = _EXPORT_MODULE_BY_NAME[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(tuple(globals()) + tuple(__all__))
