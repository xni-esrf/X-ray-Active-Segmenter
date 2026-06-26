from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QCursor, QPainter, QPainterPath, QPen, QPixmap
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


def _resolve_forced_cursor_size(
    forced_cursor_size: Optional[int],
    *,
    logger,
) -> Optional[int]:
    source = "--cursor-size"
    raw_value: Optional[object] = forced_cursor_size
    if raw_value is None:
        source = "XRA_CURSOR_SIZE"
        env_value = os.environ.get("XRA_CURSOR_SIZE")
        if env_value is None or not str(env_value).strip():
            return None
        raw_value = env_value

    try:
        normalized = int(raw_value)
    except (TypeError, ValueError):
        logger.warning("Ignoring invalid %s value: %r", source, raw_value)
        return None

    if normalized <= 0:
        logger.warning("Ignoring non-positive %s value: %d", source, normalized)
        return None

    if normalized < 12:
        logger.info("Clamping %s from %d to minimum cursor size 12 px", source, normalized)
        normalized = 12
    if normalized > 128:
        logger.info("Clamping %s from %d to maximum cursor size 128 px", source, normalized)
        normalized = 128
    return normalized


def _build_forced_arrow_cursor(size_px: int) -> QCursor:
    pixmap = QPixmap(size_px, size_px)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    # Draw a high-contrast arrow with a dark outline for remote-display visibility.
    path = QPainterPath()
    points = (
        QPointF(size_px * 0.10, size_px * 0.05),
        QPointF(size_px * 0.10, size_px * 0.88),
        QPointF(size_px * 0.30, size_px * 0.67),
        QPointF(size_px * 0.42, size_px * 0.97),
        QPointF(size_px * 0.57, size_px * 0.90),
        QPointF(size_px * 0.45, size_px * 0.60),
        QPointF(size_px * 0.74, size_px * 0.60),
    )
    path.moveTo(points[0])
    for point in points[1:]:
        path.lineTo(point)
    path.closeSubpath()

    outline_width = max(1.0, float(size_px) * 0.08)
    painter.setPen(
        QPen(
            QColor(0, 0, 0, 230),
            outline_width,
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin,
        )
    )
    painter.setBrush(QColor(255, 255, 255, 240))
    painter.drawPath(path)
    painter.end()

    hot_x = max(0, min(size_px - 1, int(round(size_px * 0.12))))
    hot_y = max(0, min(size_px - 1, int(round(size_px * 0.06))))
    return QCursor(pixmap, hot_x, hot_y)


def _apply_forced_cursor_size(
    app: QApplication,
    forced_cursor_size: Optional[int],
    *,
    logger,
) -> None:
    size_px = _resolve_forced_cursor_size(forced_cursor_size, logger=logger)
    if size_px is None:
        return
    app.setOverrideCursor(_build_forced_arrow_cursor(size_px))
    logger.info("Applied forced application cursor size: %d px", size_px)


def run(
    *,
    config: Optional[AppConfig] = None,
    volume_path: Optional[str] = None,
    semantic_path: Optional[str] = None,
    instance_path: Optional[str] = None,
    bbox_path: Optional[str] = None,
    forced_cursor_size: Optional[int] = None,
    run_event_loop: bool = True,
) -> AppContext:
    config = config or AppConfig()
    setup_logging(config.log_level)
    logger = get_logger(__name__)

    app = QApplication.instance() or QApplication([])
    _apply_forced_cursor_size(app, forced_cursor_size, logger=logger)

    renderer = Renderer()
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
                pyramid_levels=1,
            )
            if main_window.set_semantic_volume(prepared.volume):
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
                pyramid_levels=1,
            )
            if main_window.set_instance_volume(prepared.volume):
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
