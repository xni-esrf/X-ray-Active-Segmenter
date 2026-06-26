from __future__ import annotations

import unittest

from src.tests.bottom_panel_test_utils import QApplication, BottomPanelTestCase


class BottomPanelLearningTests(BottomPanelTestCase):
    def test_learning_buttons_emit_callbacks(self) -> None:
        load_model_events = []
        save_model_events = []
        segment_inference_events = []
        stop_inference_events = []
        train_events = []
        stop_events = []
        change_training_parameters_events = []
        self.assertEqual(
            self.panel._learning_panel.load_model_button.text(),
            "Load Model",
        )
        self.assertEqual(
            self.panel._learning_panel.save_model_button.text(),
            "Save Model",
        )
        self.assertEqual(
            self.panel._learning_panel.train_model_button.text(),
            "Train Model",
        )
        self.assertEqual(
            self.panel._learning_panel.segment_inference_button.text(),
            "Segment Inference BBox",
        )
        self.assertEqual(
            self.panel._learning_panel.stop_inference_button.text(),
            "Stop Inference",
        )
        self.assertEqual(
            self.panel._learning_panel.stop_training_button.text(),
            "Stop Training",
        )
        self.assertEqual(
            self.panel._learning_panel.change_training_parameters_button.text(),
            "Change default training parameters",
        )
        self.assertFalse(self.panel._learning_panel.stop_training_button.isEnabled())
        self.assertFalse(self.panel._learning_panel.stop_inference_button.isEnabled())
        self.assertEqual(
            self.panel._learning_panel.training_status.text(),
            "Training: Idle",
        )
        self.panel.on_load_model_requested(
            lambda: load_model_events.append("load_model")
        )
        self.panel.on_save_model_requested(
            lambda: save_model_events.append("save_model")
        )
        self.panel.on_segment_inference_requested(
            lambda: segment_inference_events.append("segment_inference")
        )
        self.panel.on_stop_inference_requested(
            lambda: stop_inference_events.append("stop_inference")
        )
        self.panel.on_train_model_requested(lambda: train_events.append("train"))
        self.panel.on_stop_training_requested(lambda: stop_events.append("stop"))
        self.panel.on_change_training_parameters_requested(
            lambda: change_training_parameters_events.append("change")
        )

        self.panel.set_stop_inference_enabled(True)
        self.panel.set_stop_training_enabled(True)
        self.panel._learning_panel.load_model_button.click()
        self.panel._learning_panel.save_model_button.click()
        self.panel._learning_panel.segment_inference_button.click()
        self.panel._learning_panel.stop_inference_button.click()
        self.panel._learning_panel.train_model_button.click()
        self.panel._learning_panel.stop_training_button.click()
        self.panel._learning_panel.change_training_parameters_button.click()
        QApplication.processEvents()

        self.assertEqual(load_model_events, ["load_model"])
        self.assertEqual(save_model_events, ["save_model"])
        self.assertEqual(segment_inference_events, ["segment_inference"])
        self.assertEqual(stop_inference_events, ["stop_inference"])
        self.assertEqual(train_events, ["train"])
        self.assertEqual(stop_events, ["stop"])
        self.assertEqual(change_training_parameters_events, ["change"])

    def test_learning_training_status_display_updates(self) -> None:
        self.assertFalse(self.panel.learning_training_running())
        self.assertEqual(self.panel._learning_panel.training_status.text(), "Training: Idle")
        self.assertFalse(self.panel._learning_panel.stop_training_button.isEnabled())
        self.assertFalse(self.panel._learning_panel.stop_inference_button.isEnabled())

        self.panel.set_learning_training_running(True)
        self.assertTrue(self.panel.learning_training_running())
        self.assertEqual(self.panel._learning_panel.training_status.text(), "Training: Running")

        self.panel.set_learning_training_running(False)
        self.assertFalse(self.panel.learning_training_running())
        self.assertEqual(self.panel._learning_panel.training_status.text(), "Training: Idle")

        self.panel.set_stop_training_enabled(True)
        self.assertTrue(self.panel._learning_panel.stop_training_button.isEnabled())
        self.panel.set_stop_training_enabled(False)
        self.assertFalse(self.panel._learning_panel.stop_training_button.isEnabled())
        self.panel.set_stop_inference_enabled(True)
        self.assertTrue(self.panel._learning_panel.stop_inference_button.isEnabled())
        self.panel.set_stop_inference_enabled(False)
        self.assertFalse(self.panel._learning_panel.stop_inference_button.isEnabled())


if __name__ == "__main__":
    unittest.main()
