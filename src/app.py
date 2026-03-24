from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PySide6.QtWidgets import QApplication

from .annotation import SegmentationEditor
from .bbox import BoundingBoxManager
from .config import AppConfig
from .data import VolumeData
from .events import InputHandlers, SyncManager
from .loading import load_prepared_volume
from .render import Renderer
from .ui import MainWindow
from .ui.dialogs import show_warning
from .utils import get_logger, setup_logging
from .workers import IOWorker


@dataclass
class AppContext:
    renderer: Renderer
    sync_manager: SyncManager
    main_window: MainWindow
    input_handlers: InputHandlers
    bbox_manager: Optional[BoundingBoxManager] = None
    volume: Optional[VolumeData] = None
    semantic_volume: Optional[VolumeData] = None
    instance_volume: Optional[VolumeData] = None
    io_worker: Optional[IOWorker] = None
    semantic_worker: Optional[IOWorker] = None
    instance_worker: Optional[IOWorker] = None
    segmentation_editor: Optional[SegmentationEditor] = None


def run(
    *,
    config: Optional[AppConfig] = None,
    volume_path: Optional[str] = None,
    semantic_path: Optional[str] = None,
    instance_path: Optional[str] = None,
    bbox_path: Optional[str] = None,
    run_event_loop: bool = True,
) -> AppContext:
    config = config or AppConfig()
    setup_logging(config.log_level)
    logger = get_logger(__name__)

    app = QApplication.instance() or QApplication([])

    renderer = Renderer(eager_statistics=(config.load_mode == "ram"))
    sync_manager = SyncManager()
    input_handlers = InputHandlers(sync_manager=sync_manager)
    main_window = MainWindow(
        renderer=renderer,
        sync_manager=sync_manager,
        input_handlers=input_handlers,
        load_mode=config.load_mode,
        cache_max_bytes=config.cache_max_bytes,
    )

    context = AppContext(
        renderer=renderer,
        sync_manager=sync_manager,
        main_window=main_window,
        input_handlers=input_handlers,
    )

    if volume_path:
        logger.info("Loading volume: %s", volume_path)
        try:
            prepared = load_prepared_volume(
                volume_path,
                kind="raw",
                load_mode=config.load_mode,
                cache_max_bytes=config.cache_max_bytes,
                pyramid_levels=4,
            )
            if main_window.set_volume(prepared.volume, levels=prepared.levels):
                context.volume = prepared.volume
                context.io_worker = IOWorker(volume=prepared.volume, cache=prepared.cache)
                main_window.render_all()
        except Exception as exc:
            logger.exception("Failed to load raw volume at startup: %s", volume_path)
            show_warning(str(exc), parent=main_window)

    if semantic_path:
        logger.info("Loading semantic map: %s", semantic_path)
        try:
            prepared = load_prepared_volume(
                semantic_path,
                kind="semantic",
                load_mode=config.load_mode,
                cache_max_bytes=config.cache_max_bytes,
                pyramid_levels=4,
            )
            if main_window.set_semantic_volume(prepared.volume, levels=prepared.levels):
                context.semantic_volume = main_window.semantic_volume()
                context.instance_volume = None
                context.instance_worker = None
                context.segmentation_editor = main_window.segmentation_editor()
                if context.semantic_volume is not None:
                    context.semantic_worker = IOWorker(
                        volume=context.semantic_volume,
                        cache=context.semantic_volume.cache,
                    )
                main_window.render_all()
        except Exception as exc:
            logger.exception("Failed to load semantic map at startup: %s", semantic_path)
            show_warning(str(exc), parent=main_window)

    if instance_path:
        logger.info("Loading instance map: %s", instance_path)
        try:
            prepared = load_prepared_volume(
                instance_path,
                kind="instance",
                load_mode=config.load_mode,
                cache_max_bytes=config.cache_max_bytes,
                pyramid_levels=4,
            )
            if main_window.set_instance_volume(prepared.volume, levels=prepared.levels):
                context.semantic_volume = None
                context.semantic_worker = None
                context.instance_volume = main_window.instance_volume()
                context.segmentation_editor = main_window.segmentation_editor()
                if context.instance_volume is not None:
                    context.instance_worker = IOWorker(
                        volume=context.instance_volume,
                        cache=context.instance_volume.cache,
                    )
                main_window.render_all()
        except Exception as exc:
            logger.exception("Failed to load instance map at startup: %s", instance_path)
            show_warning(str(exc), parent=main_window)

    if bbox_path:
        if context.volume is None:
            logger.warning("Cannot load bounding boxes without a raw volume: %s", bbox_path)
        else:
            logger.info("Loading bounding boxes: %s", bbox_path)
            if not main_window.load_bounding_boxes_path(bbox_path):
                logger.warning("Failed to load bounding boxes from: %s", bbox_path)

    main_window.show()
    if run_event_loop:
        app.exec()

    context.bbox_manager = main_window.bounding_box_manager()
    return context
