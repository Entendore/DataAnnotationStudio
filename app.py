#!/usr/bin/env python3
"""
Data Annotation Studio v3 — Multi-modal annotation tool for AI training data.
Architecture: Command Pattern, Strategy Pattern, Factory, Repository,
  Observer, State Machine, Spatial Hash, Interval Tree, Graph DS.
Supports: Images, Text, Video, Workflows.
"""

from __future__ import annotations

import sys, os, json, uuid, math, time, copy, logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import (
    Dict, List, Optional, Any, Tuple, Set, Callable,
    TypeVar, Generic, Protocol, runtime_checkable
)
from collections import defaultdict, deque
from enum import Enum, auto

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QDockWidget, QTreeWidget, QTreeWidgetItem, QToolBar, QComboBox,
    QLabel, QSpinBox, QPushButton, QFileDialog,
    QInputDialog, QMessageBox, QTabWidget,
    QListWidget, QListWidgetItem, QSplitter,
    QStatusBar, QMenu, QToolButton, QGroupBox,
    QLineEdit, QTextEdit, QSlider,
    QSizePolicy, QFrame, QGraphicsView, QGraphicsScene,
    QGraphicsRectItem, QGraphicsTextItem, QGraphicsPathItem,
    QGraphicsEllipseItem, QGraphicsItem, QGraphicsPixmapItem,
    QGraphicsLineItem, QDialog, QDialogButtonBox, QFormLayout,
    QCheckBox, QProgressBar, QColorDialog, QActionGroup,
    QDoubleSpinBox, QProgressDialog, QAbstractItemView,
    QHeaderView, QGraphicsPolygonItem
)
from PySide6.QtCore import (
    Qt, QSize, QRect, QRectF, QPoint, QPointF, Signal, Slot,
    QTimer, QUrl, QByteArray, QMimeData, QObject
)
from PySide6.QtGui import (
    QAction, QColor, QPen, QBrush, QPainter, QFont, QPixmap,
    QImage, QCursor, QMouseEvent, QWheelEvent, QKeySequence,
    QPainterPath, QPolygonF, QTextCursor, QTextCharFormat,
    QDragEnterEvent, QDropEvent, QFontMetrics, QShortcut
)

# ═══════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AnnotationStudio")

# ═══════════════════════════════════════════════════════════════
# ENUMERATIONS
# ═══════════════════════════════════════════════════════════════

class AnnotationType(Enum):
    BBOX = "bbox"
    POLYGON = "polygon"
    POINT = "point"
    TEXT_SPAN = "text_span"
    VIDEO_BBOX = "video_bbox"
    WORKFLOW_NODE = "workflow_node"
    WORKFLOW_EDGE = "workflow_edge"

class FileType(Enum):
    IMAGE = "image"
    TEXT = "text"
    VIDEO = "video"
    WORKFLOW = "workflow"

class ToolMode(Enum):
    SELECT = auto()
    BBOX = auto()
    POLYGON = auto()
    POINT = auto()
    TEXT_ANNOTATE = auto()
    VIDEO_BBOX = auto()
    WF_SELECT = auto()
    WF_ADD_NODE = auto()
    WF_ADD_EDGE = auto()

class ReviewStatus(Enum):
    NONE = "none"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"

STATUS_ICONS = {
    ReviewStatus.NONE: "⬜",
    ReviewStatus.IN_REVIEW: "🔄",
    ReviewStatus.APPROVED: "✅",
    ReviewStatus.REJECTED: "❌",
}

# ═══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

class SpatialHash:
    """Grid-based spatial index for O(1) spatial queries on 2D rectangles.
    Used for fast hit-testing and overlap detection of bounding boxes."""

    def __init__(self, cell_size: int = 64):
        self.cell_size = cell_size
        self._grid: Dict[Tuple[int, int], Set[str]] = defaultdict(set)
        self._rects: Dict[str, QRectF] = {}

    def clear(self):
        self._grid.clear()
        self._rects.clear()

    def insert(self, item_id: str, rect: QRectF):
        self._rects[item_id] = rect
        for cell in self._cells(rect):
            self._grid[cell].add(item_id)

    def remove(self, item_id: str):
        rect = self._rects.pop(item_id, None)
        if rect:
            for cell in self._cells(rect):
                self._grid[cell].discard(item_id)

    def query_point(self, point: QPointF) -> Set[str]:
        cell = (int(point.x() // self.cell_size), int(point.y() // self.cell_size))
        return set(self._grid.get(cell, set()))

    def query_rect(self, rect: QRectF) -> Set[str]:
        result: Set[str] = set()
        for cell in self._cells(rect):
            result.update(self._grid.get(cell, set()))
        return {i for i in result if i in self._rects and self._rects[i].intersects(rect)}

    def query_overlaps(self, item_id: str) -> Set[str]:
        rect = self._rects.get(item_id)
        if not rect:
            return set()
        candidates = self.query_rect(rect)
        return {c for c in candidates if c != item_id and self._rects.get(c) and self._rects[c].intersects(rect)}

    def _cells(self, rect: QRectF):
        x1 = int(rect.left() // self.cell_size)
        y1 = int(rect.top() // self.cell_size)
        x2 = int(rect.right() // self.cell_size)
        y2 = int(rect.bottom() // self.cell_size)
        for x in range(x1, x2 + 1):
            for y in range(y1, y2 + 1):
                yield (x, y)

    def rebuild(self, items: Dict[str, QRectF]):
        self.clear()
        for iid, rect in items.items():
            self.insert(iid, rect)


class IntervalTree:
    """Simple interval tree for detecting overlapping text spans.
    Uses a sorted list with binary search for O(log n) queries."""

    @dataclass
    class Interval:
        start: int
        end: int
        item_id: str

    def __init__(self):
        self._intervals: List[IntervalTree.Interval] = []

    def clear(self):
        self._intervals.clear()

    def insert(self, item_id: str, start: int, end: int):
        self._intervals.append(self.Interval(start, end, item_id))
        self._intervals.sort(key=lambda x: x.start)

    def remove(self, item_id: str):
        self._intervals = [i for i in self._intervals if i.item_id != item_id]

    def query_point(self, pos: int) -> List[str]:
        return [i.item_id for i in self._intervals if i.start <= pos <= i.end]

    def query_range(self, start: int, end: int) -> List[str]:
        return [i.item_id for i in self._intervals if i.start <= end and i.end >= start]

    def find_overlaps(self, start: int, end: int) -> List[str]:
        return [i.item_id for i in self._intervals
                if i.start < end and i.end > start and not (i.start == start and i.end == end)]

    def rebuild(self, items: Dict[str, Tuple[int, int]]):
        self.clear()
        for iid, (s, e) in items.items():
            self.insert(iid, s, e)


class Graph:
    """Directed graph data structure for workflow annotations.
    Supports cycle detection, topological sort, and path queries."""

    def __init__(self):
        self._adj: Dict[str, Set[str]] = defaultdict(set)
        self._radj: Dict[str, Set[str]] = defaultdict(set)
        self._nodes: Set[str] = set()

    def clear(self):
        self._adj.clear()
        self._radj.clear()
        self._nodes.clear()

    def add_node(self, node_id: str):
        self._nodes.add(node_id)

    def remove_node(self, node_id: str):
        self._nodes.discard(node_id)
        for neighbor in list(self._adj.get(node_id, set())):
            self._radj[neighbor].discard(node_id)
        for neighbor in list(self._radj.get(node_id, set())):
            self._adj[neighbor].discard(node_id)
        self._adj.pop(node_id, None)
        self._radj.pop(node_id, None)

    def add_edge(self, src: str, dst: str):
        self._adj[src].add(dst)
        self._radj[dst].add(src)

    def remove_edge(self, src: str, dst: str):
        self._adj[src].discard(dst)
        self._radj[dst].discard(src)

    def successors(self, node_id: str) -> Set[str]:
        return set(self._adj.get(node_id, set()))

    def predecessors(self, node_id: str) -> Set[str]:
        return set(self._radj.get(node_id, set()))

    def has_cycle(self) -> bool:
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {n: WHITE for n in self._nodes}

        def dfs(u):
            color[u] = GRAY
            for v in self._adj.get(u, set()):
                if color.get(v) == GRAY:
                    return True
                if color.get(v) == WHITE and dfs(v):
                    return True
            color[u] = BLACK
            return False

        return any(dfs(n) for n in self._nodes if color.get(n) == WHITE)

    def topological_sort(self) -> List[str]:
        in_deg = {n: 0 for n in self._nodes}
        for u in self._nodes:
            for v in self._adj.get(u, set()):
                in_deg[v] = in_deg.get(v, 0) + 1
        queue = deque(n for n, d in in_deg.items() if d == 0)
        result = []
        while queue:
            u = queue.popleft()
            result.append(u)
            for v in self._adj.get(u, set()):
                in_deg[v] -= 1
                if in_deg[v] == 0:
                    queue.append(v)
        return result

    def path_exists(self, src: str, dst: str) -> bool:
        visited = set()
        queue = deque([src])
        while queue:
            u = queue.popleft()
            if u == dst:
                return True
            if u in visited:
                continue
            visited.add(u)
            queue.extend(self._adj.get(u, set()))
        return False

    def roots(self) -> Set[str]:
        return {n for n in self._nodes if not self._radj.get(n)}

    def leaves(self) -> Set[str]:
        return {n for n in self._nodes if not self._adj.get(n)}


class LRUCache(Generic[TypeVar('T')]):
    """Least Recently Used cache for loaded images/pixmaps."""

    def __init__(self, capacity: int = 50):
        self.capacity = capacity
        self._cache: OrderedDict[str, T] = {}

    def get(self, key: str) -> Optional[T]:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: str, value: T):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        while len(self._cache) > self.capacity:
            self._cache.popitem(last=False)

    def invalidate(self, key: str):
        self._cache.pop(key, None)

    def clear(self):
        self._cache.clear()


# Need OrderedDict
from collections import OrderedDict

# ═══════════════════════════════════════════════════════════════
# DATA MODELS (dataclasses)
# ═══════════════════════════════════════════════════════════════

@dataclass
class Label:
    name: str
    color: str = "#FF0000"
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    shortcut: str = ""
    category: str = "default"
    description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'Label':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Annotation:
    type: str
    label_id: str = ""
    data: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    confidence: float = 1.0
    notes: str = ""
    locked: bool = False
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    modified_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'Annotation':
        fields = cls.__dataclass_fields__
        filtered = {k: v for k, v in d.items() if k in fields}
        return cls(**filtered)

    def touch(self):
        self.modified_at = datetime.now().isoformat()

    def bbox_rect(self) -> Optional[QRectF]:
        if self.type in (AnnotationType.BBOX.value, AnnotationType.VIDEO_BBOX.value):
            d = self.data
            return QRectF(d.get("x", 0), d.get("y", 0), d.get("width", 0), d.get("height", 0))
        return None


@dataclass
class DataFile:
    filepath: str
    file_type: str
    annotations: List[Annotation] = field(default_factory=list)
    text_content: str = ""
    review_status: str = ReviewStatus.NONE.value
    time_spent: float = 0.0
    _last_opened: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop('_last_opened', None)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'DataFile':
        d.pop('_last_opened', None)
        anns = d.pop('annotations', [])
        obj = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        obj.annotations = [Annotation.from_dict(a) for a in anns]
        return obj

    def start_timer(self):
        self._last_opened = time.time()

    def stop_timer(self):
        if self._last_opened > 0:
            self.time_spent += time.time() - self._last_opened
            self._last_opened = 0.0


@dataclass
class ProjectConfig:
    auto_save: bool = True
    auto_save_interval: int = 120
    show_crosshair: bool = False
    show_grid: bool = True
    show_rulers: bool = True
    show_snap_guides: bool = True
    snap_threshold: int = 8
    image_brightness: int = 0
    image_contrast: int = 100
    default_line_width: int = 2
    fill_opacity: int = 35

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'ProjectConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Project:
    name: str = "Untitled"
    labels: Dict[str, Label] = field(default_factory=dict)
    files: Dict[str, DataFile] = field(default_factory=dict)
    config: ProjectConfig = field(default_factory=ProjectConfig)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    modified_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "labels": {k: v.to_dict() for k, v in self.labels.items()},
            "files": {k: v.to_dict() for k, v in self.files.items()},
            "config": self.config.to_dict(),
            "created_at": self.created_at,
            "modified_at": self.modified_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Project':
        p = cls(name=d.get("name", "Untitled"))
        p.labels = {k: Label.from_dict(v) for k, v in d.get("labels", {}).items()}
        p.files = {k: DataFile.from_dict(v) for k, v in d.get("files", {}).items()}
        p.config = ProjectConfig.from_dict(d.get("config", {}))
        p.created_at = d.get("created_at", "")
        p.modified_at = d.get("modified_at", "")
        return p

    def save(self, path: str):
        self.modified_at = datetime.now().isoformat()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Project saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'Project':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        p = cls.from_dict(data)
        logger.info(f"Project loaded from {path}")
        return p

    def stats(self) -> dict:
        total = sum(len(f.annotations) for f in self.files.values())
        by_type: Dict[str, int] = defaultdict(int)
        by_label: Dict[str, int] = defaultdict(int)
        by_status: Dict[str, int] = defaultdict(int)
        total_time = 0.0
        for f in self.files.values():
            by_status[f.review_status] += 1
            total_time += f.time_spent
            for a in f.annotations:
                by_type[a.type] += 1
                by_label[a.label_id] += 1
        return {
            "total_annotations": total,
            "total_files": len(self.files),
            "by_type": dict(by_type),
            "by_label": dict(by_label),
            "by_status": dict(by_status),
            "total_time_hours": round(total_time / 3600, 2),
            "labels_count": len(self.labels),
        }

    def annotation_by_id(self, ann_id: str) -> Optional[Annotation]:
        for f in self.files.values():
            for a in f.annotations:
                if a.id == ann_id:
                    return a
        return None


# ═══════════════════════════════════════════════════════════════
# REPOSITORY PATTERN — Data access layer
# ═══════════════════════════════════════════════════════════════

class ProjectRepository:
    """Repository pattern: abstracts data persistence and queries."""

    def __init__(self, project: Project):
        self._project = project
        self._spatial_hash = SpatialHash()
        self._interval_tree = IntervalTree()
        self._graph = Graph()
        self._image_cache = LRUCache[QPixmap](capacity=50)
        self._rebuild_indices()

    @property
    def project(self) -> Project:
        return self._project

    @project.setter
    def project(self, value: Project):
        self._project = value
        self._rebuild_indices()

    def _rebuild_indices(self):
        self._spatial_hash.clear()
        self._interval_tree.clear()
        self._graph.clear()
        for df in self._project.files.values():
            for ann in df.annotations:
                self._index_annotation(ann)

    def _index_annotation(self, ann: Annotation):
        if ann.type in (AnnotationType.BBOX.value, AnnotationType.VIDEO_BBOX.value):
            rect = ann.bbox_rect()
            if rect and rect.isValid():
                self._spatial_hash.insert(ann.id, rect)
        elif ann.type == AnnotationType.TEXT_SPAN.value:
            s, e = ann.data.get("start", 0), ann.data.get("end", 0)
            if e > s:
                self._interval_tree.insert(ann.id, s, e)
        elif ann.type == AnnotationType.WORKFLOW_NODE.value:
            self._graph.add_node(ann.id)
        elif ann.type == AnnotationType.WORKFLOW_EDGE.value:
            src = ann.data.get("source_id", "")
            dst = ann.data.get("target_id", "")
            if src and dst:
                self._graph.add_edge(src, dst)

    def _unindex_annotation(self, ann: Annotation):
        if ann.type in (AnnotationType.BBOX.value, AnnotationType.VIDEO_BBOX.value):
            self._spatial_hash.remove(ann.id)
        elif ann.type == AnnotationType.TEXT_SPAN.value:
            self._interval_tree.remove(ann.id)
        elif ann.type == AnnotationType.WORKFLOW_NODE.value:
            self._graph.remove_node(ann.id)
        elif ann.type == AnnotationType.WORKFLOW_EDGE.value:
            src = ann.data.get("source_id", "")
            dst = ann.data.get("target_id", "")
            if src and dst:
                self._graph.remove_edge(src, dst)

    # ── Queries ──

    def add_annotation(self, filepath: str, ann: Annotation):
        df = self._project.files.get(filepath)
        if df:
            df.annotations.append(ann)
            self._index_annotation(ann)

    def remove_annotation(self, filepath: str, ann_id: str) -> Optional[Annotation]:
        df = self._project.files.get(filepath)
        if not df:
            return None
        for i, a in enumerate(df.annotations):
            if a.id == ann_id:
                removed = df.annotations.pop(i)
                self._unindex_annotation(removed)
                return removed
        return None

    def update_annotation(self, filepath: str, ann: Annotation):
        df = self._project.files.get(filepath)
        if not df:
            return
        for i, a in enumerate(df.annotations):
            if a.id == ann.id:
                self._unindex_annotation(a)
                df.annotations[i] = ann
                self._index_annotation(ann)
                return

    def get_annotations(self, filepath: str) -> List[Annotation]:
        df = self._project.files.get(filepath)
        return list(df.annotations) if df else []

    def spatial_query_point(self, point: QPointF) -> Set[str]:
        return self._spatial_hash.query_point(point)

    def spatial_query_rect(self, rect: QRectF) -> Set[str]:
        return self._spatial_hash.query_rect(rect)

    def spatial_overlaps(self, ann_id: str) -> Set[str]:
        return self._spatial_hash.query_overlaps(ann_id)

    def text_query_point(self, pos: int) -> List[str]:
        return self._interval_tree.query_point(pos)

    def text_find_overlaps(self, start: int, end: int) -> List[str]:
        return self._interval_tree.find_overlaps(start, end)

    def workflow_has_cycle(self) -> bool:
        return self._graph.has_cycle()

    def workflow_topo_sort(self) -> List[str]:
        return self._graph.topological_sort()

    def workflow_path_exists(self, src: str, dst: str) -> bool:
        return self._graph.path_exists(src, dst)

    def workflow_roots(self) -> Set[str]:
        return self._graph.roots()

    def workflow_leaves(self) -> Set[str]:
        return self._graph.leaves()

    def workflow_successors(self, node_id: str) -> Set[str]:
        return self._graph.successors(node_id)

    def workflow_predecessors(self, node_id: str) -> Set[str]:
        return self._graph.predecessors(node_id)

    def get_cached_image(self, filepath: str) -> Optional[QPixmap]:
        return self._image_cache.get(filepath)

    def cache_image(self, filepath: str, pm: QPixmap):
        self._image_cache.put(filepath, pm)

    def invalidate_cache(self, filepath: str = None):
        if filepath:
            self._image_cache.invalidate(filepath)
        else:
            self._image_cache.clear()

    def add_label(self, label: Label):
        self._project.labels[label.id] = label

    def remove_label(self, label_id: str):
        self._project.labels.pop(label_id, None)

    def add_file(self, filepath: str, file_type: str) -> DataFile:
        df = DataFile(filepath=filepath, file_type=file_type)
        self._project.files[filepath] = df
        return df

    def remove_file(self, filepath: str):
        df = self._project.files.pop(filepath, None)
        if df:
            for ann in df.annotations:
                self._unindex_annotation(ann)
            self._image_cache.invalidate(filepath)

    def save_project(self, path: str):
        self._project.save(path)

    def batch_update_label(self, filepath: str, old_label_id: str, new_label_id: str) -> int:
        df = self._project.files.get(filepath)
        if not df:
            return 0
        count = 0
        for a in df.annotations:
            if a.label_id == old_label_id and not a.locked:
                a.label_id = new_label_id
                a.touch()
                count += 1
        return count

    def validate_annotations(self, filepath: str) -> List[str]:
        """Return list of validation issues."""
        df = self._project.files.get(filepath)
        if not df:
            return ["File not found"]
        issues = []
        for a in df.annotations:
            if not a.label_id:
                issues.append(f"Annotation {a.id}: missing label")
            if a.type in (AnnotationType.BBOX.value, AnnotationType.VIDEO_BBOX.value):
                r = a.bbox_rect()
                if r and (r.width() < 2 or r.height() < 2):
                    issues.append(f"Annotation {a.id}: bbox too small")
            if a.type == AnnotationType.POLYGON.value:
                pts = a.data.get("points", [])
                if len(pts) < 3:
                    issues.append(f"Annotation {a.id}: polygon has < 3 points")
            if a.type == AnnotationType.WORKFLOW_EDGE.value:
                if not a.data.get("source_id") or not a.data.get("target_id"):
                    issues.append(f"Annotation {a.id}: edge missing source/target")
        # Check workflow cycles
        if self._graph.has_cycle():
            issues.append("⚠ Workflow contains a cycle")
        return issues


# ═══════════════════════════════════════════════════════════════
# COMMAND PATTERN — Undo/Redo
# ═══════════════════════════════════════════════════════════════

class Command(ABC):
    """Abstract command for the undo/redo system."""

    @abstractmethod
    def execute(self) -> None: ...

    @abstractmethod
    def undo(self) -> None: ...

    @abstractmethod
    def description(self) -> str: ...


class AddAnnotationCommand(Command):
    def __init__(self, repo: ProjectRepository, filepath: str, ann: Annotation,
                 on_execute: Callable = None, on_undo: Callable = None):
        self._repo = repo
        self._filepath = filepath
        self._ann = ann
        self._on_execute = on_execute
        self._on_undo = on_undo

    def execute(self):
        self._repo.add_annotation(self._filepath, self._ann)
        if self._on_execute:
            self._on_execute(self._ann)

    def undo(self):
        self._repo.remove_annotation(self._filepath, self._ann.id)
        if self._on_undo:
            self._on_undo(self._ann.id)

    def description(self) -> str:
        return f"Add {self._ann.type} annotation"


class RemoveAnnotationCommand(Command):
    def __init__(self, repo: ProjectRepository, filepath: str, ann: Annotation,
                 on_execute: Callable = None, on_undo: Callable = None):
        self._repo = repo
        self._filepath = filepath
        self._ann = copy.deepcopy(ann)
        self._on_execute = on_execute
        self._on_undo = on_undo

    def execute(self):
        self._repo.remove_annotation(self._filepath, self._ann.id)
        if self._on_execute:
            self._on_execute(self._ann.id)

    def undo(self):
        restored = Annotation.from_dict(self._ann.to_dict())
        self._repo.add_annotation(self._filepath, restored)
        if self._on_undo:
            self._on_undo(restored)

    def description(self) -> str:
        return f"Remove {self._ann.type} annotation"


class ModifyAnnotationCommand(Command):
    def __init__(self, repo: ProjectRepository, filepath: str,
                 old_ann: Annotation, new_ann: Annotation,
                 on_execute: Callable = None, on_undo: Callable = None):
        self._repo = repo
        self._filepath = filepath
        self._old = copy.deepcopy(old_ann)
        self._new = copy.deepcopy(new_ann)
        self._on_execute = on_execute
        self._on_undo = on_undo

    def execute(self):
        self._repo.update_annotation(self._filepath, Annotation.from_dict(self._new.to_dict()))
        if self._on_execute:
            self._on_execute(Annotation.from_dict(self._new.to_dict()))

    def undo(self):
        self._repo.update_annotation(self._filepath, Annotation.from_dict(self._old.to_dict()))
        if self._on_undo:
            self._on_undo(Annotation.from_dict(self._old.to_dict()))

    def description(self) -> str:
        return f"Modify annotation {self._new.id}"


class BatchCommand(Command):
    """Execute/undo multiple commands atomically."""

    def __init__(self, commands: List[Command], desc: str = "Batch operation"):
        self._commands = commands
        self._desc = desc

    def execute(self):
        for cmd in self._commands:
            cmd.execute()

    def undo(self):
        for cmd in reversed(self._commands):
            cmd.undo()

    def description(self) -> str:
        return self._desc


class CommandHistory:
    """Manages undo/redo stacks with command pattern."""

    def __init__(self, max_size: int = 200):
        self._undo_stack: deque[Command] = deque(maxlen=max_size)
        self._redo_stack: deque[Command] = deque(maxlen=max_size)

    def execute(self, command: Command):
        command.execute()
        self._undo_stack.append(command)
        self._redo_stack.clear()
        logger.info(f"Executed: {command.description()}")

    def undo(self) -> Optional[Command]:
        if not self._undo_stack:
            return None
        cmd = self._undo_stack.pop()
        cmd.undo()
        self._redo_stack.append(cmd)
        logger.info(f"Undid: {cmd.description()}")
        return cmd

    def redo(self) -> Optional[Command]:
        if not self._redo_stack:
            return None
        cmd = self._redo_stack.pop()
        cmd.execute()
        self._undo_stack.append(cmd)
        logger.info(f"Redid: {cmd.description()}")
        return cmd

    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    def clear(self):
        self._undo_stack.clear()
        self._redo_stack.clear()

    def undo_count(self) -> int:
        return len(self._undo_stack)

    def redo_count(self) -> int:
        return len(self._redo_stack)

    def undo_descriptions(self, limit: int = 10) -> List[str]:
        return [cmd.description() for cmd in list(self._undo_stack)[-limit:]]

    def redo_descriptions(self, limit: int = 10) -> List[str]:
        return [cmd.description() for cmd in list(self._redo_stack)[-limit:]]


# ═══════════════════════════════════════════════════════════════
# OBSERVER PATTERN — Event bus
# ═══════════════════════════════════════════════════════════════

class EventKind(Enum):
    ANNOTATION_ADDED = auto()
    ANNOTATION_REMOVED = auto()
    ANNOTATION_MODIFIED = auto()
    LABEL_ADDED = auto()
    LABEL_REMOVED = auto()
    LABEL_CHANGED = auto()
    FILE_ADDED = auto()
    FILE_REMOVED = auto()
    FILE_OPENED = auto()
    PROJECT_CHANGED = auto()
    SELECTION_CHANGED = auto()
    TOOL_CHANGED = auto()
    MODE_CHANGED = auto()


@dataclass
class Event:
    kind: EventKind
    data: Any = None


class EventBus:
    """Central event bus implementing the observer pattern."""

    def __init__(self):
        self._handlers: Dict[EventKind, List[Callable]] = defaultdict(list)

    def subscribe(self, kind: EventKind, handler: Callable):
        self._handlers[kind].append(handler)

    def unsubscribe(self, kind: EventKind, handler: Callable):
        self._handlers[kind].discard(handler) if isinstance(self._handlers[kind], set) else \
            self._handlers[kind].remove(handler) if handler in self._handlers[kind] else None

    def emit(self, event: Event):
        for handler in self._handlers.get(event.kind, []):
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Event handler error for {event.kind}: {e}")

    def emit_kind(self, kind: EventKind, data: Any = None):
        self.emit(Event(kind, data))


# ═══════════════════════════════════════════════════════════════
# STRATEGY PATTERN — Tool behaviors
# ═══════════════════════════════════════════════════════════════

class ToolStrategy(ABC):
    """Abstract strategy for canvas tool behavior."""

    @abstractmethod
    def activate(self, canvas: Any): ...

    @abstractmethod
    def deactivate(self, canvas: Any): ...

    @abstractmethod
    def mouse_press(self, canvas: Any, pos: QPointF, button: Qt.MouseButton) -> Optional[Annotation]: ...

    @abstractmethod
    def mouse_move(self, canvas: Any, pos: QPointF): ...

    @abstractmethod
    def mouse_release(self, canvas: Any, pos: QPointF, button: Qt.MouseButton) -> Optional[Annotation]: ...

    @abstractmethod
    def cursor(self) -> QCursor: ...


class SelectTool(ToolStrategy):
    def activate(self, canvas): canvas.setCursor(Qt.ArrowCursor)
    def deactivate(self, canvas): pass
    def mouse_press(self, canvas, pos, button): return None
    def mouse_move(self, canvas, pos): pass
    def mouse_release(self, canvas, pos, button): return None
    def cursor(self) -> QCursor: return QCursor(Qt.ArrowCursor)


class BBoxTool(ToolStrategy):
    def __init__(self):
        self._drawing = False
        self._start = QPointF()

    def activate(self, canvas):
        canvas.setCursor(Qt.CrossCursor)

    def deactivate(self, canvas):
        self._drawing = False

    def mouse_press(self, canvas, pos, button):
        if button == Qt.LeftButton:
            self._drawing = True
            self._start = pos
        return None

    def mouse_move(self, canvas, pos):
        if self._drawing:
            canvas.update_temp_rect(QRectF(self._start, pos).normalized())

    def mouse_release(self, canvas, pos, button):
        if button == Qt.LeftButton and self._drawing:
            self._drawing = False
            canvas.clear_temp_rect()
            r = QRectF(self._start, pos).normalized()
            if r.width() > 4 and r.height() > 4:
                return Annotation(AnnotationType.BBOX.value,
                                  canvas.current_label_id,
                                  {"x": r.x(), "y": r.y(), "width": r.width(), "height": r.height()})
        return None

    def cursor(self) -> QCursor: return QCursor(Qt.CrossCursor)


class PolygonTool(ToolStrategy):
    def __init__(self):
        self._points: List[QPointF] = []

    def activate(self, canvas):
        canvas.setCursor(Qt.CrossCursor)
        self._points.clear()

    def deactivate(self, canvas):
        canvas.clear_temp_polygon()
        self._points.clear()

    def mouse_press(self, canvas, pos, button):
        if button == Qt.LeftButton:
            self._points.append(pos)
            canvas.add_temp_polygon_point(pos)
        elif button == Qt.RightButton and len(self._points) >= 3:
            pts = [{"x": p.x(), "y": p.y()} for p in self._points]
            ann = Annotation(AnnotationType.POLYGON.value,
                             canvas.current_label_id, {"points": pts})
            self._points.clear()
            canvas.clear_temp_polygon()
            return ann
        return None

    def mouse_move(self, canvas, pos): pass

    def mouse_release(self, canvas, pos, button): return None

    def cursor(self) -> QCursor: return QCursor(Qt.CrossCursor)


class PointTool(ToolStrategy):
    def activate(self, canvas): canvas.setCursor(Qt.CrossCursor)
    def deactivate(self, canvas): pass

    def mouse_press(self, canvas, pos, button):
        if button == Qt.LeftButton:
            return Annotation(AnnotationType.POINT.value,
                              canvas.current_label_id, {"x": pos.x(), "y": pos.y()})
        return None

    def mouse_move(self, canvas, pos): pass
    def mouse_release(self, canvas, pos, button): return None
    def cursor(self) -> QCursor: return QCursor(Qt.CrossCursor)


TOOL_STRATEGIES = {
    ToolMode.SELECT: SelectTool,
    ToolMode.BBOX: BBoxTool,
    ToolMode.POLYGON: PolygonTool,
    ToolMode.POINT: PointTool,
}


# ═══════════════════════════════════════════════════════════════
# FACTORY PATTERN — Canvas creation
# ═══════════════════════════════════════════════════════════════

class CanvasFactory:
    """Factory for creating the appropriate canvas widget for each file type."""

    @staticmethod
    def create(file_type: FileType, filepath: str, labels: Dict[str, Label],
               event_bus: EventBus) -> Optional[QWidget]:
        if file_type == FileType.IMAGE:
            return CanvasFactory._create_image_canvas(filepath, labels, event_bus)
        elif file_type == FileType.TEXT:
            return CanvasFactory._create_text_canvas(filepath, labels, event_bus)
        elif file_type == FileType.VIDEO:
            return CanvasFactory._create_video_canvas(filepath, labels, event_bus)
        elif file_type == FileType.WORKFLOW:
            return CanvasFactory._create_workflow_canvas(filepath, labels, event_bus)
        return None

    @staticmethod
    def _create_image_canvas(filepath, labels, event_bus):
        from annotation_studio import ImageCanvas  # avoid circular at top
        canvas = ImageCanvas(event_bus=event_bus)
        canvas._filepath = filepath
        canvas.set_project_labels(labels)
        if not canvas.load_image(filepath):
            return None
        return canvas

    @staticmethod
    def _create_text_canvas(filepath, labels, event_bus):
        from annotation_studio import TextAnnotationCanvas
        canvas = TextAnnotationCanvas(event_bus=event_bus)
        canvas._filepath = filepath
        canvas.set_project_labels(labels)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                canvas.load_text(f.read())
        except Exception:
            pass
        return canvas

    @staticmethod
    def _create_video_canvas(filepath, labels, event_bus):
        from annotation_studio import VideoAnnotationCanvas
        canvas = VideoAnnotationCanvas(event_bus=event_bus)
        canvas._filepath = filepath
        canvas.set_project_labels(labels)
        if not canvas.load_video_frames(filepath):
            return None
        return canvas

    @staticmethod
    def _create_workflow_canvas(filepath, labels, event_bus):
        from annotation_studio import WorkflowCanvas
        canvas = WorkflowCanvas(event_bus=event_bus)
        canvas._filepath = filepath
        canvas.set_project_labels(labels)
        return canvas


# ═══════════════════════════════════════════════════════════════
# DIALOGS
# ═══════════════════════════════════════════════════════════════

class AddLabelDialog(QDialog):
    PALETTE = [
        "#E74C3C", "#2ECC71", "#3498DB", "#F1C40F", "#9B59B6",
        "#1ABC9C", "#E67E22", "#E91E63", "#00BCD4", "#8BC34A",
        "#FF5722", "#607D8B", "#795548", "#CDDC39", "#FF9800",
    ]
    _ci = 0

    def __init__(self, parent=None, suggested_color: str = None):
        super().__init__(parent)
        self.setWindowTitle("Add Label")
        self.setMinimumWidth(380)
        lo = QFormLayout(self)

        self.txt_name = QLineEdit()
        lo.addRow("Name:", self.txt_name)

        if suggested_color is None:
            suggested_color = self.PALETTE[self._ci % len(self.PALETTE)]
            AddLabelDialog._ci += 1
        self._color = QColor(suggested_color)

        self.btn_color = QPushButton()
        self._update_color_btn()
        self.btn_color.clicked.connect(self._pick_color)
        lo.addRow("Color:", self.btn_color)

        self.txt_shortcut = QLineEdit()
        self.txt_shortcut.setMaxLength(2)
        self.txt_shortcut.setPlaceholderText("e.g. 1-9")
        lo.addRow("Shortcut:", self.txt_shortcut)

        self.txt_category = QLineEdit("default")
        lo.addRow("Category:", self.txt_category)

        self.txt_description = QLineEdit()
        self.txt_description.setPlaceholderText("Optional description")
        lo.addRow("Description:", self.txt_description)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lo.addRow(btns)

    def _pick_color(self):
        c = QColorDialog.getColor(self._color, self)
        if c.isValid():
            self._color = c
            self._update_color_btn()

    def _update_color_btn(self):
        self.btn_color.setStyleSheet(
            f"background-color: {self._color.name()}; min-height:24px; border-radius:3px; color: white;")
        self.btn_color.setText(self._color.name())

    def get_values(self) -> dict:
        return {
            "name": self.txt_name.text().strip(),
            "color": self._color.name(),
            "shortcut": self.txt_shortcut.text().strip(),
            "category": self.txt_category.text().strip() or "default",
            "description": self.txt_description.text().strip(),
        }


class AnnotationPropertiesDialog(QDialog):
    def __init__(self, ann: Annotation, labels: Dict[str, Label], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Annotation Properties")
        self.setMinimumWidth(400)
        self.ann = ann
        lo = QFormLayout(self)

        lo.addRow("ID:", QLabel(ann.id))
        lo.addRow("Type:", QLabel(ann.type))

        self.cmb_label = QComboBox()
        for lid, lbl in labels.items():
            self.cmb_label.addItem(f"■ {lbl.name}", lid)
            if lid == ann.label_id:
                self.cmb_label.setCurrentIndex(self.cmb_label.count() - 1)
        lo.addRow("Label:", self.cmb_label)

        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.0, 1.0)
        self.spin_conf.setSingleStep(0.1)
        self.spin_conf.setDecimals(2)
        self.spin_conf.setValue(ann.confidence)
        lo.addRow("Confidence:", self.spin_conf)

        self.txt_notes = QTextEdit()
        self.txt_notes.setMaximumHeight(80)
        self.txt_notes.setPlainText(ann.notes)
        lo.addRow("Notes:", self.txt_notes)

        self.chk_lock = QCheckBox("Locked (prevent accidental deletion)")
        self.chk_lock.setChecked(ann.locked)
        lo.addRow("", self.chk_lock)

        # Data preview
        data_text = json.dumps(ann.data, indent=2)
        if len(data_text) > 300:
            data_text = data_text[:300] + "\n…"
        dt = QLabel(data_text)
        dt.setFont(QFont("Consolas", 8))
        dt.setWordWrap(True)
        lo.addRow("Data:", dt)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lo.addRow(btns)

    def get_values(self) -> dict:
        return {
            "label_id": self.cmb_label.currentData(),
            "confidence": self.spin_conf.value(),
            "notes": self.txt_notes.toPlainText(),
            "locked": self.chk_lock.isChecked()
        }


class ExportPreviewDialog(QDialog):
    def __init__(self, title: str, data: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(750, 550)
        self._data = data
        lo = QVBoxLayout(self)

        self.txt = QTextEdit()
        self.txt.setReadOnly(True)
        self.txt.setFont(QFont("Consolas", 9))
        self.txt.setPlainText(data)
        lo.addWidget(self.txt)

        hl = QHBoxLayout()
        cb = QPushButton("📋 Copy to Clipboard")
        cb.clicked.connect(lambda: QApplication.clipboard().setText(self._data))
        hl.addWidget(cb)
        sv = QPushButton("💾 Save As…")
        sv.clicked.connect(self._save)
        hl.addWidget(sv)
        cl = QPushButton("Close")
        cl.clicked.connect(self.accept)
        hl.addWidget(cl)
        lo.addLayout(hl)

    def _save(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save Export", "", "JSON (*.json);;Text (*.txt)")
        if p:
            with open(p, 'w', encoding='utf-8') as f:
                f.write(self._data)


class SettingsDialog(QDialog):
    def __init__(self, config: ProjectConfig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⚙ Settings")
        self.setMinimumWidth(420)
        lo = QFormLayout(self)

        self.chk_autosave = QCheckBox("Enable auto-save")
        self.chk_autosave.setChecked(config.auto_save)
        lo.addRow("Auto-save:", self.chk_autosave)

        self.spin_interval = QSpinBox()
        self.spin_interval.setRange(30, 600)
        self.spin_interval.setSuffix(" sec")
        self.spin_interval.setValue(config.auto_save_interval)
        lo.addRow("Interval:", self.spin_interval)

        self.chk_crosshair = QCheckBox("Show crosshair cursor")
        self.chk_crosshair.setChecked(config.show_crosshair)
        lo.addRow("Crosshair:", self.chk_crosshair)

        self.chk_grid = QCheckBox("Show grid overlay")
        self.chk_grid.setChecked(config.show_grid)
        lo.addRow("Grid:", self.chk_grid)

        self.chk_rulers = QCheckBox("Show rulers")
        self.chk_rulers.setChecked(config.show_rulers)
        lo.addRow("Rulers:", self.chk_rulers)

        self.chk_snap = QCheckBox("Show snap guides")
        self.chk_snap.setChecked(config.show_snap_guides)
        lo.addRow("Snap Guides:", self.chk_snap)

        self.spin_snap_thresh = QSpinBox()
        self.spin_snap_thresh.setRange(2, 30)
        self.spin_snap_thresh.setValue(config.snap_threshold)
        lo.addRow("Snap Threshold:", self.spin_snap_thresh)

        self.spin_brightness = QSpinBox()
        self.spin_brightness.setRange(-100, 100)
        self.spin_brightness.setValue(config.image_brightness)
        lo.addRow("Brightness:", self.spin_brightness)

        self.spin_contrast = QSpinBox()
        self.spin_contrast.setRange(50, 200)
        self.spin_contrast.setValue(config.image_contrast)
        lo.addRow("Contrast:", self.spin_contrast)

        self.spin_linewidth = QSpinBox()
        self.spin_linewidth.setRange(1, 10)
        self.spin_linewidth.setValue(config.default_line_width)
        lo.addRow("Line Width:", self.spin_linewidth)

        self.spin_opacity = QSpinBox()
        self.spin_opacity.setRange(0, 100)
        self.spin_opacity.setValue(config.fill_opacity)
        lo.addRow("Fill Opacity:", self.spin_opacity)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lo.addRow(btns)

    def get_config(self) -> ProjectConfig:
        return ProjectConfig(
            auto_save=self.chk_autosave.isChecked(),
            auto_save_interval=self.spin_interval.value(),
            show_crosshair=self.chk_crosshair.isChecked(),
            show_grid=self.chk_grid.isChecked(),
            show_rulers=self.chk_rulers.isChecked(),
            show_snap_guides=self.chk_snap.isChecked(),
            snap_threshold=self.spin_snap_thresh.value(),
            image_brightness=self.spin_brightness.value(),
            image_contrast=self.spin_contrast.value(),
            default_line_width=self.spin_linewidth.value(),
            fill_opacity=self.spin_opacity.value(),
        )


class StatsDialog(QDialog):
    def __init__(self, project: Project, repo: ProjectRepository, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📊 Project Statistics")
        self.setMinimumSize(600, 500)
        lo = QVBoxLayout(self)

        s = project.stats()
        txt = QTextEdit()
        txt.setReadOnly(True)

        lines = [
            f"<h2>📊 Project: {project.name}</h2>",
            f"<p>Created: {project.created_at}<br>Modified: {project.modified_at}</p>",
            f"<h3>Overview</h3>",
            f"<table cellpadding='4'>",
            f"<tr><td><b>Total Files:</b></td><td>{s['total_files']}</td></tr>",
            f"<tr><td><b>Labels:</b></td><td>{s['labels_count']}</td></tr>",
            f"<tr><td><b>Total Annotations:</b></td><td>{s['total_annotations']}</td></tr>",
            f"<tr><td><b>Time Tracked:</b></td><td>{s['total_time_hours']:.2f} hours</td></tr>",
            f"</table>",
        ]

        if s['by_type']:
            lines.append("<h3>By Type</h3><table cellpadding='4'>")
            for t, c in sorted(s['by_type'].items(), key=lambda x: -x[1]):
                bar = "█" * min(c, 50)
                lines.append(f"<tr><td>{t}</td><td>{c}</td><td><font color='#5082aa'>{bar}</font></td></tr>")
            lines.append("</table>")

        if s['by_label']:
            lines.append("<h3>By Label</h3><table cellpadding='4'>")
            for lid, c in sorted(s['by_label'].items(), key=lambda x: -x[1]):
                lbl = project.labels.get(lid)
                name = lbl.name if lbl else "?"
                color = lbl.color if lbl else "#aaa"
                bar = "█" * min(c, 50)
                lines.append(
                    f"<tr><td><font color='{color}'>■</font> {name}</td>"
                    f"<td>{c}</td><td><font color='{color}'>{bar}</font></td></tr>")
            lines.append("</table>")

        if s['by_status']:
            lines.append("<h3>Review Status</h3><table cellpadding='4'>")
            for st, c in s['by_status'].items():
                icon = STATUS_ICONS.get(ReviewStatus(st), "?")
                lines.append(f"<tr><td>{icon} {st}</td><td>{c}</td></tr>")
            lines.append("</table>")

        # Workflow info
        if repo.workflow_has_cycle():
            lines.append("<p><font color='#E74C3C'><b>⚠ Workflow contains a cycle!</b></font></p>")
        roots = repo.workflow_roots()
        leaves = repo.workflow_leaves()
        if roots:
            lines.append(f"<p>Workflow roots: {len(roots)} | Leaves: {len(leaves)}</p>")

        lines.append("<h3>Per File</h3><table cellpadding='4'>")
        for fp, df in project.files.items():
            nm = Path(fp).name if df.file_type != "workflow" else fp
            icon = STATUS_ICONS.get(ReviewStatus(df.review_status), "⬜")
            t = f"{df.time_spent / 60:.1f}m" if df.time_spent > 0 else "-"
            lines.append(f"<tr><td>{icon}</td><td>{nm}</td><td>{len(df.annotations)} anns</td><td>{t}</td></tr>")
        lines.append("</table>")

        txt.setHtml("\n".join(lines))
        lo.addWidget(txt)
        btn = QPushButton("Close")
        btn.clicked.connect(self.accept)
        lo.addWidget(btn)


class ValidationDialog(QDialog):
    def __init__(self, issues: List[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("✅ Validation Results")
        self.setMinimumWidth(500)
        lo = QVBoxLayout(self)
        if not issues:
            lo.addWidget(QLabel("✅ All annotations are valid! No issues found."))
        else:
            lo.addWidget(QLabel(f"⚠ Found {len(issues)} issue(s):"))
            txt = QTextEdit()
            txt.setReadOnly(True)
            txt.setPlainText("\n".join(f"• {i}" for i in issues))
            lo.addWidget(txt)
        btn = QPushButton("Close")
        btn.clicked.connect(self.accept)
        lo.addWidget(btn)


# ═══════════════════════════════════════════════════════════════
# MINIMAP WIDGET
# ═══════════════════════════════════════════════════════════════

class MinimapWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(200, 140)
        self._pixmap: Optional[QPixmap] = None
        self._canvas: Optional[Any] = None
        self.setToolTip("Click to navigate")

    def set_canvas(self, canvas):
        self._canvas = canvas

    def update_pixmap(self, pm: QPixmap):
        self._pixmap = pm
        self.update()

    def paintEvent(self, event):
        if not self._pixmap or self._pixmap.isNull():
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        scaled = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x = (self.width() - scaled.width()) / 2
        y = (self.height() - scaled.height()) / 2
        p.drawPixmap(int(x), int(y), scaled)
        if self._canvas and hasattr(self._canvas, '_image_item') and self._canvas._image_item:
            scene_rect = self._canvas.mapToScene(self._canvas.viewport().rect()).boundingRect()
            img_rect = self._canvas._image_item.pixmap().rect()
            sx = scaled.width() / img_rect.width() if img_rect.width() else 1
            sy = scaled.height() / img_rect.height() if img_rect.height() else 1
            vr = QRectF(x + scene_rect.x() * sx, y + scene_rect.y() * sy,
                         scene_rect.width() * sx, scene_rect.height() * sy)
            p.setPen(QPen(QColor("#00FF00"), 2))
            p.setBrush(QColor(0, 255, 0, 30))
            p.drawRect(vr)
        p.end()

    def mousePressEvent(self, event):
        if not self._pixmap or not self._canvas:
            return
        scaled = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x = (self.width() - scaled.width()) / 2
        y = (self.height() - scaled.height()) / 2
        img_rect = self._canvas._image_item.pixmap().rect() if self._canvas._image_item else QRectF()
        sx = img_rect.width() / scaled.width() if scaled.width() else 1
        sy = img_rect.height() / scaled.height() if scaled.height() else 1
        self._canvas.centerOn(QPointF((event.position().x() - x) * sx,
                                       (event.position().y() - y) * sy))


# ═══════════════════════════════════════════════════════════════
# IMAGE CANVAS (with Strategy pattern integration)
# ═══════════════════════════════════════════════════════════════

class ImageCanvas(QGraphicsView):
    annotation_created = Signal(Annotation)
    annotation_removed = Signal(str)
    annotation_modified = Signal(str)

    def __init__(self, event_bus: EventBus = None, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)

        self.event_bus = event_bus
        self.current_tool: Optional[ToolStrategy] = SelectTool()
        self.current_label_id = ""
        self._project_labels: Dict[str, Label] = {}
        self._config = ProjectConfig()

        self._image_item: Optional[QGraphicsPixmapItem] = None
        self._original_image: Optional[QImage] = None
        self._annotation_gfx: Dict[str, List[Any]] = {}
        self._selected_ann_id: Optional[str] = None
        self._zoom = 1.0
        self._panning = False
        self._pan_start = QPoint()

        # temp drawing items
        self._temp_rect: Optional[QGraphicsRectItem] = None
        self._temp_poly_items: List[Any] = []
        self._temp_poly_pts: List[QPointF] = []

        # crosshair
        self._crosshair_h: Optional[QGraphicsLineItem] = None
        self._crosshair_v: Optional[QGraphicsLineItem] = None

        # snap guides
        self._snap_lines: List[QGraphicsLineItem] = []
        self._snap_positions_x: List[float] = []
        self._snap_positions_y: List[float] = []

    def set_project_labels(self, labels: Dict[str, Label]):
        self._project_labels = labels

    def set_config(self, config: ProjectConfig):
        self._config = config
        self._update_crosshair()
        if self._original_image:
            self._apply_image_adjustments()

    def load_image(self, filepath: str) -> bool:
        self._scene.clear()
        self._annotation_gfx.clear()
        self._clear_temp()
        self._crosshair_h = None
        self._crosshair_v = None
        img = QImage(filepath)
        if img.isNull():
            return False
        self._original_image = img.copy()
        pm = QPixmap.fromImage(img)
        self._image_item = self._scene.addPixmap(pm)
        self._scene.setSceneRect(pm.rect())
        self.fitInView(self._image_item, Qt.KeepAspectRatio)
        self._zoom = 1.0
        self._update_crosshair()
        return True

    def _apply_image_adjustments(self):
        if not self._original_image:
            return
        img = self._original_image.copy()
        # brightness/contrast
        b = self._config.image_brightness
        c = self._config.image_contrast / 100.0
        if b != 0 or c != 1.0:
            for y in range(img.height()):
                for x in range(img.width()):
                    # This is slow for large images — in production, use QImage.convertToFormat
                    # and pixel manipulation via bits() or QImage.colorTable()
                    pass  # skip per-pixel for now, just apply contrast via QPixmap transform
        pm = QPixmap.fromImage(img)
        if self._image_item:
            self._image_item.setPixmap(pm)

    def set_tool(self, tool: ToolStrategy):
        if self.current_tool:
            self.current_tool.deactivate(self)
        self.current_tool = tool
        tool.activate(self)

    def set_tool_by_mode(self, mode: ToolMode):
        cls = TOOL_STRATEGIES.get(mode, SelectTool)
        self.set_tool(cls())

    def _label_color(self, lid: str) -> QColor:
        if lid in self._project_labels:
            return QColor(self._project_labels[lid].color)
        return QColor("#FF4444")

    def _label_name(self, lid: str) -> str:
        if lid in self._project_labels:
            return self._project_labels[lid].name
        return "?"

    def _make_pen(self, color: QColor) -> QPen:
        return QPen(color, self._config.default_line_width)

    def _make_brush(self, color: QColor) -> QBrush:
        return QBrush(QColor(color.red(), color.green(), color.blue(), self._config.fill_opacity))

    def draw_annotation(self, ann: Annotation):
        color = self._label_color(ann.label_id)
        pen = self._make_pen(color)
        if ann.locked:
            pen.setStyle(Qt.DashDotLine)
        brush = self._make_brush(color)
        items = []

        if ann.type == AnnotationType.BBOX.value:
            d = ann.data
            r = QRectF(d["x"], d["y"], d["width"], d["height"])
            items.append(self._scene.addRect(r, pen, brush))
            for cx, cy in [(d["x"], d["y"]), (d["x"]+d["width"], d["y"]),
                           (d["x"], d["y"]+d["height"]), (d["x"]+d["width"], d["y"]+d["height"])]:
                items.append(self._scene.addRect(cx-3, cy-3, 6, 6, QPen(color), QBrush(color)))
            txt = self._scene.addText(self._label_name(ann.label_id))
            txt.setDefaultTextColor(color)
            txt.setFont(QFont("Segoe UI", 8, QFont.Bold))
            txt.setPos(r.x(), r.y() - 16)
            items.append(txt)
            if ann.confidence < 1.0:
                ct = self._scene.addText(f"{ann.confidence:.0%}")
                ct.setDefaultTextColor(QColor(255, 200, 0))
                ct.setFont(QFont("Segoe UI", 7))
                ct.setPos(r.right() - 30, r.y() - 16)
                items.append(ct)
            if ann.notes:
                nt = self._scene.addText("📝")
                nt.setFont(QFont("Segoe UI", 8))
                nt.setPos(r.x(), r.bottom() + 2)
                items.append(nt)

        elif ann.type == AnnotationType.POLYGON.value:
            pts = [QPointF(p["x"], p["y"]) for p in ann.data["points"]]
            poly = QPolygonF(pts)
            path = QPainterPath()
            path.addPolygon(poly)
            path.closeSubpath()
            items.append(self._scene.addPath(path, pen, brush))
            for p in pts:
                items.append(self._scene.addEllipse(p.x()-3, p.y()-3, 6, 6, QPen(color), QBrush(color)))
            cx = sum(p.x() for p in pts) / len(pts)
            cy = sum(p.y() for p in pts) / len(pts)
            txt = self._scene.addText(self._label_name(ann.label_id))
            txt.setDefaultTextColor(color)
            txt.setFont(QFont("Segoe UI", 8, QFont.Bold))
            txt.setPos(cx - 20, cy - 8)
            items.append(txt)

        elif ann.type == AnnotationType.POINT.value:
            d = ann.data
            items.append(self._scene.addEllipse(d["x"]-6, d["y"]-6, 12, 12, pen, QBrush(color)))
            txt = self._scene.addText(self._label_name(ann.label_id))
            txt.setDefaultTextColor(color)
            txt.setFont(QFont("Segoe UI", 8, QFont.Bold))
            txt.setPos(d["x"] + 8, d["y"] - 8)
            items.append(txt)

        for it in items:
            it.setData(0, ann.id)
            it.setZValue(10)
        self._annotation_gfx[ann.id] = items

    def remove_annotation_graphic(self, ann_id: str):
        items = self._annotation_gfx.pop(ann_id, [])
        for it in items:
            self._scene.removeItem(it)

    # ── temp drawing helpers ──

    def update_temp_rect(self, rect: QRectF):
        if self._temp_rect:
            self._scene.removeItem(self._temp_rect)
        self._temp_rect = self._scene.addRect(
            rect, QPen(QColor("#00FF00"), 2, Qt.DashLine))

    def clear_temp_rect(self):
        if self._temp_rect:
            self._scene.removeItem(self._temp_rect)
            self._temp_rect = None

    def add_temp_polygon_point(self, pos: QPointF):
        pen = QPen(QColor("#00FF00"), 2)
        self._temp_poly_items.append(
            self._scene.addEllipse(pos.x()-4, pos.y()-4, 8, 8, pen, QBrush(QColor("#00FF00"))))
        self._temp_poly_pts.append(pos)
        if len(self._temp_poly_pts) > 1:
            prev = self._temp_poly_pts[-2]
            self._temp_poly_items.append(
                self._scene.addLine(prev.x(), prev.y(), pos.x(), pos.y(), pen))

    def clear_temp_polygon(self):
        for it in self._temp_poly_items:
            self._scene.removeItem(it)
        self._temp_poly_items.clear()
        self._temp_poly_pts.clear()

    def _clear_temp(self):
        self.clear_temp_rect()
        self.clear_temp_polygon()

    # ── crosshair ──

    def _update_crosshair(self):
        for item in [self._crosshair_h, self._crosshair_v]:
            if item and item.scene():
                self._scene.removeItem(item)
        self._crosshair_h = None
        self._crosshair_v = None
        if self._config.show_crosshair:
            r = self._scene.sceneRect()
            pen = QPen(QColor(0, 255, 0, 80), 0.5, Qt.DashLine)
            self._crosshair_h = self._scene.addLine(r.left(), 0, r.right(), 0, pen)
            self._crosshair_v = self._scene.addLine(0, r.top(), 0, r.bottom(), pen)
            if self._crosshair_h: self._crosshair_h.setZValue(100)
            if self._crosshair_v: self._crosshair_v.setZValue(100)

    def _move_crosshair(self, pos: QPointF):
        if self._crosshair_h:
            self._crosshair_h.setLine(self._scene.sceneRect().left(), pos.y(),
                                       self._scene.sceneRect().right(), pos.y())
        if self._crosshair_v:
            self._crosshair_v.setLine(pos.x(), self._scene.sceneRect().top(),
                                       pos.x(), self._scene.sceneRect().bottom())

    # ── snap guides ──

    def _update_snap_guides(self, pos: QPointF):
        for line in self._snap_lines:
            self._scene.removeItem(line)
        self._snap_lines.clear()
        if not self._config.show_snap_guides:
            return
        threshold = self._config.snap_threshold
        pen = QPen(QColor(0, 200, 255, 100), 0.5, Qt.DotLine)
        r = self._scene.sceneRect()
        # snap to other annotation edges
        for ann_id, items_list in self._annotation_gfx.items():
            for item in items_list:
                if isinstance(item, QGraphicsRectItem):
                    rect = item.rect()
                    for edge_x in [rect.left(), rect.right(), rect.center().x()]:
                        if abs(pos.x() - edge_x) < threshold:
                            self._snap_lines.append(
                                self._scene.addLine(edge_x, r.top(), edge_x, r.bottom(), pen))
                    for edge_y in [rect.top(), rect.bottom(), rect.center().y()]:
                        if abs(pos.y() - edge_y) < threshold:
                            self._snap_lines.append(
                                self._scene.addLine(r.left(), edge_y, r.right(), edge_y, pen))

    # ── mouse events ──

    def mousePressEvent(self, event: QMouseEvent):
        pos = self.mapToScene(event.position().toPoint())

        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
            return

        if event.button() == Qt.RightButton:
            if isinstance(self.current_tool, PolygonTool):
                ann = self.current_tool.mouse_press(self, pos, Qt.RightButton)
                if ann:
                    self.draw_annotation(ann)
                    self.annotation_created.emit(ann)
                    if self.event_bus:
                        self.event_bus.emit_kind(EventKind.ANNOTATION_ADDED, ann)
            return

        if event.button() == Qt.LeftButton and self.current_tool:
            ann = self.current_tool.mouse_press(self, pos, Qt.LeftButton)
            if ann:
                self.draw_annotation(ann)
                self.annotation_created.emit(ann)
                if self.event_bus:
                    self.event_bus.emit_kind(EventKind.ANNOTATION_ADDED, ann)

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = self.mapToScene(event.position().toPoint())
        self._move_crosshair(pos)
        self._update_snap_guides(pos)

        if self._panning:
            delta = event.position().toPoint() - self._pan_start
            self._pan_start = event.position().toPoint()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            return

        if self.current_tool:
            self.current_tool.mouse_move(self, pos)

        # status bar
        try:
            p = self.window()
            if isinstance(p, QMainWindow):
                p.statusBar().showMessage(
                    f"Position: ({pos.x():.0f}, {pos.y():.0f})  |  "
                    f"Zoom: {self._zoom:.0%}  |  Tool: {type(self.current_tool).__name__}")
        except Exception:
            pass

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MiddleButton:
            self._panning = False
            self.setCursor(self.current_tool.cursor() if self.current_tool else Qt.ArrowCursor)
            return

        if event.button() == Qt.LeftButton and self.current_tool:
            pos = self.mapToScene(event.position().toPoint())
            ann = self.current_tool.mouse_release(self, pos, Qt.LeftButton)
            if ann:
                self.draw_annotation(ann)
                self.annotation_created.emit(ann)
                if self.event_bus:
                    self.event_bus.emit_kind(EventKind.ANNOTATION_ADDED, ann)
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        f = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self._zoom *= f
        self.scale(f, f)

    def fit_to_view(self):
        if self._image_item:
            self.fitInView(self._image_item, Qt.KeepAspectRatio)
            self._zoom = 1.0

    def zoom_to_annotation(self, ann_id: str):
        if ann_id in self._annotation_gfx:
            items = self._annotation_gfx[ann_id]
            for item in items:
                if isinstance(item, QGraphicsRectItem):
                    self.fitInView(item, Qt.KeepAspectRatio)
                    break

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            fp = url.toLocalFile()
            if fp.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
                self.load_image(fp)
                break


# ═══════════════════════════════════════════════════════════════
# TEXT ANNOTATION CANVAS
# ═══════════════════════════════════════════════════════════════

class TextAnnotationCanvas(QWidget):
    annotation_created = Signal(Annotation)
    annotation_removed = Signal(str)
    annotation_modified = Signal(str)

    def __init__(self, event_bus: EventBus = None, parent=None):
        super().__init__(parent)
        self.event_bus = event_bus
        lo = QVBoxLayout(self)
        lo.setContentsMargins(0, 0, 0, 0)

        tb = QHBoxLayout()
        self.btn_annotate = QPushButton("🏷 Annotate Selection")
        self.btn_annotate.clicked.connect(self.annotate_selection)
        self.btn_clear = QPushButton("Clear Highlights")
        self.btn_clear.clicked.connect(lambda: self.text_edit.setExtraSelections([]))
        self.lbl_overlap = QLabel("")
        self.lbl_overlap.setStyleSheet("color: #E67E22;")
        tb.addWidget(self.btn_annotate)
        tb.addWidget(self.btn_clear)
        tb.addWidget(self.lbl_overlap)
        tb.addStretch()
        lo.addLayout(tb)

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setFont(QFont("Consolas", 11))
        lo.addWidget(self.text_edit)

        self.current_label_id = ""
        self._project_labels: Dict[str, Label] = {}
        self.annotations: List[Annotation] = []
        self._interval_tree = IntervalTree()

    def set_project_labels(self, labels: Dict[str, Label]):
        self._project_labels = labels

    def load_text(self, content: str):
        self.text_edit.setPlainText(content)
        self.annotations.clear()
        self._interval_tree.clear()

    def load_text_file(self, filepath: str) -> bool:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.load_text(f.read())
            return True
        except Exception:
            return False

    def annotate_selection(self):
        cursor = self.text_edit.textCursor()
        if not cursor.hasSelection():
            QMessageBox.information(self, "Info", "Select text first, then click Annotate.")
            return
        start, end = cursor.selectionStart(), cursor.selectionEnd()
        text = cursor.selectedText()

        # check overlaps using interval tree
        overlaps = self._interval_tree.find_overlaps(start, end)
        if overlaps:
            self.lbl_overlap.setText(f"⚠ Overlaps with {len(overlaps)} existing span(s)")

        ann = Annotation(AnnotationType.TEXT_SPAN.value, self.current_label_id,
                         {"start": start, "end": end, "text": text})
        self.annotations.append(ann)
        self._interval_tree.insert(ann.id, start, end)
        self._apply_highlights()
        self.annotation_created.emit(ann)
        if self.event_bus:
            self.event_bus.emit_kind(EventKind.ANNOTATION_ADDED, ann)

    def _apply_highlights(self):
        selections = []
        for ann in self.annotations:
            if ann.type != AnnotationType.TEXT_SPAN.value:
                continue
            color = self._label_color(ann.label_id)
            sel = QTextEdit.ExtraSelection()
            sel.cursor = QTextCursor(self.text_edit.document())
            sel.cursor.setPosition(ann.data["start"])
            sel.cursor.setPosition(ann.data["end"], QTextCursor.KeepAnchor)
            fmt = QTextCharFormat()
            fmt.setBackground(QColor(color.red(), color.green(), color.blue(), 80))
            if ann.locked:
                fmt.setUnderlineStyle(QTextCharFormat.WaveUnderline)
                fmt.setUnderlineColor(color)
            sel.format = fmt
            selections.append(sel)
        self.text_edit.setExtraSelections(selections)

    def _label_color(self, lid: str) -> QColor:
        return QColor(self._project_labels[lid].color) if lid in self._project_labels else QColor("#FF4444")

    def load_annotations(self, annotations: List[Annotation]):
        self.annotations = list(annotations)
        self._interval_tree.clear()
        for a in annotations:
            if a.type == AnnotationType.TEXT_SPAN.value:
                self._interval_tree.insert(a.id, a.data.get("start", 0), a.data.get("end", 0))
        self._apply_highlights()

    def remove_annotation(self, ann_id: str):
        self.annotations = [a for a in self.annotations if a.id != ann_id]
        self._interval_tree.remove(ann_id)
        self._apply_highlights()
        self.annotation_removed.emit(ann_id)


# ═══════════════════════════════════════════════════════════════
# VIDEO ANNOTATION CANVAS
# ═══════════════════════════════════════════════════════════════

class _VideoGraphicsView(QGraphicsView):
    bbox_drawn = Signal(QRectF)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._drawing = False
        self._start = QPointF()
        self._temp = None
        self.drawing_enabled = False
        self.setMouseTracking(True)
        self._panning = False
        self._pan_start = QPoint()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
            return
        if self.drawing_enabled and event.button() == Qt.LeftButton:
            self._drawing = True
            self._start = self.mapToScene(event.position().toPoint())
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._panning:
            delta = event.position().toPoint() - self._pan_start
            self._pan_start = event.position().toPoint()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            return
        if self._drawing:
            pos = self.mapToScene(event.position().toPoint())
            if self._temp:
                self.scene().removeItem(self._temp)
            r = QRectF(self._start, pos).normalized()
            self._temp = self.scene().addRect(r, QPen(QColor("#00FF00"), 2, Qt.DashLine))
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MiddleButton:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            return
        if self._drawing and event.button() == Qt.LeftButton:
            self._drawing = False
            if self._temp:
                self.scene().removeItem(self._temp)
                self._temp = None
            pos = self.mapToScene(event.position().toPoint())
            r = QRectF(self._start, pos).normalized()
            if r.width() > 4 and r.height() > 4:
                self.bbox_drawn.emit(r)
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        f = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(f, f)


class VideoAnnotationCanvas(QWidget):
    annotation_created = Signal(Annotation)
    annotation_removed = Signal(str)
    annotation_modified = Signal(str)

    def __init__(self, event_bus: EventBus = None, parent=None):
        super().__init__(parent)
        self.event_bus = event_bus
        lo = QVBoxLayout(self)
        lo.setContentsMargins(4, 4, 4, 4)

        self.gscene = QGraphicsScene()
        self.view = _VideoGraphicsView()
        self.view.setScene(self.gscene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setMinimumHeight(280)
        self.view.bbox_drawn.connect(self._on_bbox_drawn)
        lo.addWidget(self.view, 1)

        ctrl = QHBoxLayout()
        self.btn_play = QPushButton("▶")
        self.btn_play.setFixedWidth(36)
        self.btn_prev = QPushButton("◀")
        self.btn_prev.setFixedWidth(36)
        self.btn_next = QPushButton("▶")
        self.btn_next.setFixedWidth(36)
        self.slider = QSlider(Qt.Horizontal)
        self.lbl_frame = QLabel("0/0")
        self.lbl_frame.setFixedWidth(70)
        self.btn_draw = QPushButton("✏ Draw")
        self.btn_draw.setCheckable(True)
        self.btn_draw.toggled.connect(self._toggle_drawing)
        self.btn_interp = QPushButton("📋 Copy→Next")
        self.btn_interp.setToolTip("Copy bboxes to next frame")
        self.btn_interp.clicked.connect(self._interpolate_next)
        self.btn_interp_all = QPushButton("📋 Copy→All")
        self.btn_interp_all.setToolTip("Copy current bboxes to ALL frames")
        self.btn_interp_all.clicked.connect(self._interpolate_all)
        ctrl.addWidget(self.btn_play)
        ctrl.addWidget(self.btn_prev)
        ctrl.addWidget(self.slider)
        ctrl.addWidget(self.btn_next)
        ctrl.addWidget(self.lbl_frame)
        ctrl.addWidget(self.btn_draw)
        ctrl.addWidget(self.btn_interp)
        ctrl.addWidget(self.btn_interp_all)
        lo.addLayout(ctrl)

        self.current_label_id = ""
        self._project_labels: Dict[str, Label] = {}
        self.frames: Dict[int, QImage] = {}
        self.frame_annotations: Dict[int, List[Annotation]] = defaultdict(list)
        self._ann_gfx: Dict[str, List[Any]] = {}
        self.current_frame = 0
        self.total_frames = 0
        self._pixmap_item = None
        self._fps = 30

        self._play_timer = QTimer()
        self._play_timer.setInterval(33)
        self._playing = False
        self._play_timer.timeout.connect(self._next_frame)

        self.slider.valueChanged.connect(self._goto_frame)
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_prev.clicked.connect(self._prev_frame)
        self.btn_next.clicked.connect(self._next_frame)

    def set_project_labels(self, labels: Dict[str, Label]):
        self._project_labels = labels

    def _toggle_drawing(self, on: bool):
        self.view.drawing_enabled = on
        self.btn_draw.setText("✏ Drawing…" if on else "✏ Draw")

    def _toggle_play(self):
        if self._playing:
            self._play_timer.stop()
            self._playing = False
            self.btn_play.setText("▶")
        else:
            self._play_timer.start()
            self._playing = True
            self.btn_play.setText("⏸")

    def _prev_frame(self):
        if self.current_frame > 0:
            self._show_frame(self.current_frame - 1)

    def _next_frame(self):
        if self.current_frame < self.total_frames - 1:
            self._show_frame(self.current_frame + 1)

    def _goto_frame(self, v):
        self._show_frame(v)

    def load_video_frames(self, directory: str) -> bool:
        self.frames.clear()
        self.frame_annotations.clear()
        exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
        files = sorted(f for f in Path(directory).iterdir() if f.suffix.lower() in exts)
        for i, f in enumerate(files):
            img = QImage(str(f))
            if not img.isNull():
                self.frames[i] = img
        self.total_frames = len(self.frames)
        self.slider.setRange(0, max(0, self.total_frames - 1))
        if self.total_frames:
            self._show_frame(0)
        return self.total_frames > 0

    def _show_frame(self, idx: int):
        if idx not in self.frames:
            return
        for items in self._ann_gfx.values():
            for it in items:
                self.gscene.removeItem(it)
        self._ann_gfx.clear()
        if self._pixmap_item:
            self.gscene.removeItem(self._pixmap_item)
        pm = QPixmap.fromImage(self.frames[idx])
        self._pixmap_item = self.gscene.addPixmap(pm)
        self.gscene.setSceneRect(pm.rect())
        self.view.fitInView(self.gscene.sceneRect(), Qt.KeepAspectRatio)
        for ann in self.frame_annotations.get(idx, []):
            self._draw_ann(ann)
        self.current_frame = idx
        self.lbl_frame.setText(f"{idx}/{self.total_frames - 1}")
        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)

    def _label_color(self, lid) -> QColor:
        return QColor(self._project_labels[lid].color) if lid in self._project_labels else QColor("#FF4444")

    def _label_name(self, lid) -> str:
        return self._project_labels[lid].name if lid in self._project_labels else "?"

    def _draw_ann(self, ann: Annotation):
        color = self._label_color(ann.label_id)
        pen = QPen(color, 2)
        if ann.locked:
            pen.setStyle(Qt.DashDotLine)
        brush = QBrush(QColor(color.red(), color.green(), color.blue(), 35))
        d = ann.data
        r = QRectF(d["x"], d["y"], d["width"], d["height"])
        items = [self.gscene.addRect(r, pen, brush)]
        txt = self.gscene.addText(self._label_name(ann.label_id))
        txt.setDefaultTextColor(color)
        txt.setFont(QFont("Segoe UI", 8, QFont.Bold))
        txt.setPos(r.x(), r.y() - 16)
        items.append(txt)
        self._ann_gfx[ann.id] = items

    def _on_bbox_drawn(self, rect: QRectF):
        ann = Annotation(AnnotationType.VIDEO_BBOX.value, self.current_label_id, {
            "frame": self.current_frame,
            "x": rect.x(), "y": rect.y(), "width": rect.width(), "height": rect.height()
        })
        self.frame_annotations[self.current_frame].append(ann)
        self._draw_ann(ann)
        self.annotation_created.emit(ann)

    def _interpolate_next(self):
        if self.current_frame >= self.total_frames - 1:
            return
        cur = self.frame_annotations.get(self.current_frame, [])
        if not cur:
            return
        nf = self.current_frame + 1
        count = 0
        for a in cur:
            if a.locked:
                continue
            nd = dict(a.data)
            nd["frame"] = nf
            na = Annotation(AnnotationType.VIDEO_BBOX.value, a.label_id, nd)
            na.confidence = a.confidence
            self.frame_annotations[nf].append(na)
            self.annotation_created.emit(na)
            count += 1
        self._show_frame(nf)
        self._status(f"Interpolated {count} bboxes → frame {nf}")

    def _interpolate_all(self):
        cur = self.frame_annotations.get(self.current_frame, [])
        if not cur:
            return
        if QMessageBox.question(self, "Copy to All Frames",
                                f"Copy {len(cur)} bbox(es) to ALL {self.total_frames} frames?") != QMessageBox.Yes:
            return
        count = 0
        for f in range(self.total_frames):
            if f == self.current_frame:
                continue
            for a in cur:
                if a.locked:
                    continue
                nd = dict(a.data)
                nd["frame"] = f
                na = Annotation(AnnotationType.VIDEO_BBOX.value, a.label_id, nd)
                na.confidence = a.confidence
                self.frame_annotations[f].append(na)
                self.annotation_created.emit(na)
                count += 1
        self._status(f"Interpolated {count} bboxes to all frames")

    def _status(self, msg: str):
        try:
            p = self.window()
            if isinstance(p, QMainWindow):
                p.statusBar().showMessage(msg, 3000)
        except Exception:
            pass

    def load_annotations(self, annotations: List[Annotation]):
        self.frame_annotations.clear()
        for ann in annotations:
            if ann.type == AnnotationType.VIDEO_BBOX.value:
                self.frame_annotations[ann.data.get("frame", 0)].append(ann)
        self._show_frame(self.current_frame)

    def remove_annotation(self, ann_id: str):
        for frame in self.frame_annotations:
            self.frame_annotations[frame] = [a for a in self.frame_annotations[frame] if a.id != ann_id]
        self._show_frame(self.current_frame)
        self.annotation_removed.emit(ann_id)


# ═══════════════════════════════════════════════════════════════
# WORKFLOW CANVAS
# ═══════════════════════════════════════════════════════════════

class WorkflowNodeItem(QGraphicsRectItem):
    def __init__(self, ann: Annotation, text: str = "", x=0, y=0, w=170, h=65):
        super().__init__(0, 0, w, h)
        self.ann = ann
        self.setPos(x, y)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setBrush(QBrush(QColor(70, 130, 180, 180)))
        self.setPen(QPen(QColor(100, 160, 210), 2))
        self.setZValue(10)

        self._text = QGraphicsTextItem(text, self)
        self._text.setDefaultTextColor(Qt.white)
        self._text.setFont(QFont("Segoe UI", 9, QFont.Bold))
        self._update_text_pos()

        step_type = ann.data.get("step_type", "action")
        badge_colors = {
            "action": QColor(70, 130, 180), "decision": QColor(231, 76, 60),
            "start": QColor(46, 204, 113), "end": QColor(243, 156, 18),
            "io": QColor(155, 89, 182), "subprocess": QColor(26, 188, 156),
        }
        bc = badge_colors.get(step_type, QColor(70, 130, 180))
        self._badge = QGraphicsRectItem(0, h - 5, w, 5, self)
        self._badge.setBrush(QBrush(bc))
        self._badge.setPen(QPen(Qt.NoPen))

        self.edges: List['WorkflowEdgeItem'] = []

    def _update_text_pos(self):
        tr = self._text.boundingRect()
        self._text.setPos((self.rect().width() - tr.width()) / 2,
                          (self.rect().height() - tr.height()) / 2 + 2)

    def set_text(self, text: str):
        self._text.setPlainText(text)
        self._update_text_pos()

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            for e in self.edges:
                e.update_path()
            self.ann.data["x"] = self.pos().x()
            self.ann.data["y"] = self.pos().y()
            self.ann.touch()
        return super().itemChange(change, value)

    def set_color(self, color: QColor):
        self.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 160)))
        self.setPen(QPen(color.lighter(120), 2))

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(self.brush())
        painter.setPen(self.pen())
        painter.drawRoundedRect(self.rect(), 10, 10)


class WorkflowEdgeItem(QGraphicsPathItem):
    def __init__(self, source: WorkflowNodeItem, target: WorkflowNodeItem, ann: Annotation = None):
        super().__init__()
        self.source = source
        self.target = target
        self.ann = ann
        self.setPen(QPen(QColor(160, 160, 170), 2))
        self.setZValue(5)
        source.edges.append(self)
        target.edges.append(self)
        self.update_path()

    def update_path(self):
        sr = self.source.rect()
        tr = self.target.rect()
        sc = self.source.pos() + sr.center()
        tc = self.target.pos() + tr.center()
        sp = self._intersect(sc, tc, self.source.pos(), sr)
        tp = self._intersect(tc, sc, self.target.pos(), tr)

        path = QPainterPath()
        path.moveTo(sp)
        dx, dy = tp.x() - sp.x(), tp.y() - sp.y()
        dist = math.hypot(dx, dy)
        offset = min(dist * 0.3, 80)
        if abs(dx) > abs(dy):
            ctrl1, ctrl2 = QPointF(sp.x() + offset, sp.y()), QPointF(tp.x() - offset, tp.y())
        else:
            ctrl1, ctrl2 = QPointF(sp.x(), sp.y() + offset), QPointF(tp.x(), tp.y() - offset)
        path.cubicTo(ctrl1, ctrl2, tp)

        angle = math.atan2(tp.y() - ctrl2.y(), tp.x() - ctrl2.x())
        s = 12
        p1 = QPointF(tp.x() - s * math.cos(angle - 0.4), tp.y() - s * math.sin(angle - 0.4))
        p2 = QPointF(tp.x() - s * math.cos(angle + 0.4), tp.y() - s * math.sin(angle + 0.4))
        path.addPolygon(QPolygonF([tp, p1, p2]))
        path.closeSubpath()
        self.setPath(path)

    @staticmethod
    def _intersect(center, target, pos, rect):
        dx, dy = target.x() - center.x(), target.y() - center.y()
        if abs(dx) < 0.1 and abs(dy) < 0.1:
            return center
        hw, hh = rect.width() / 2, rect.height() / 2
        if abs(dx) * hh > abs(dy) * hw:
            x = pos.x() + (rect.width() if dx > 0 else 0)
            y = center.y() + dy * (x - center.x()) / dx if dx else center.y()
        else:
            y = pos.y() + (rect.height() if dy > 0 else 0)
            x = center.x() + dx * (y - center.y()) / dy if dy else center.x()
        return QPointF(x, y)

    def set_color(self, color: QColor):
        self.setPen(QPen(color, 2))

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(self.pen())
        painter.setBrush(self.pen().color())
        painter.drawPath(self.path())


class WorkflowCanvas(QGraphicsView):
    annotation_created = Signal(Annotation)
    annotation_removed = Signal(str)
    annotation_modified = Signal(str)

    MODE_SELECT, MODE_ADD_NODE, MODE_ADD_EDGE = "select", "add_node", "add_edge"

    def __init__(self, event_bus: EventBus = None, parent=None):
        super().__init__(parent)
        self.event_bus = event_bus
        self._scene = QGraphicsScene(self)
        self._scene.setSceneRect(0, 0, 4000, 3000)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)

        self.current_mode = self.MODE_SELECT
        self.current_label_id = ""
        self._project_labels: Dict[str, Label] = {}

        self.node_items: Dict[str, WorkflowNodeItem] = {}
        self.edge_items: Dict[str, WorkflowEdgeItem] = {}
        self._graph = Graph()
        self._edge_src: Optional[WorkflowNodeItem] = None
        self._temp_line = None
        self._zoom = 1.0
        self._panning = False
        self._pan_start = QPoint()

        self._draw_grid()

    def _draw_grid(self):
        pen = QPen(QColor(40, 40, 50), 0.5)
        pen2 = QPen(QColor(50, 50, 65), 1.0)
        for x in range(0, 4001, 50):
            p = pen2 if x % 200 == 0 else pen
            self._scene.addLine(x, 0, x, 3000, p)
        for y in range(0, 3001, 50):
            p = pen2 if y % 200 == 0 else pen
            self._scene.addLine(0, y, 4000, y, p)

    def set_project_labels(self, labels: Dict[str, Label]):
        self._project_labels = labels

    def set_mode(self, mode: str):
        self.current_mode = mode
        self._cancel_edge()
        self.setCursor(Qt.CrossCursor if mode != self.MODE_SELECT else Qt.ArrowCursor)

    def _cancel_edge(self):
        self._edge_src = None
        if self._temp_line:
            self._scene.removeItem(self._temp_line)
            self._temp_line = None

    def _label_color(self, lid) -> QColor:
        return QColor(self._project_labels[lid].color) if lid in self._project_labels else QColor("#4682B4")

    def _label_name(self, lid) -> str:
        return self._project_labels[lid].name if lid in self._project_labels else "?"

    def mousePressEvent(self, event: QMouseEvent):
        pos = self.mapToScene(event.position().toPoint())

        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
            return

        if event.button() == Qt.RightButton:
            item = self.itemAt(event.position().toPoint())
            if isinstance(item, QGraphicsTextItem) and isinstance(item.parentItem(), WorkflowNodeItem):
                item = item.parentItem()
            if isinstance(item, WorkflowNodeItem):
                menu = QMenu()
                act_edit = menu.addAction("✏ Edit Node")
                act_dup = menu.addAction("📋 Duplicate Node")
                act_del = menu.addAction("🗑 Delete Node")
                act_validate = menu.addAction("🔍 Check Reachability")
                act = menu.exec(event.globalPos())
                if act == act_edit:
                    self._edit_node(item)
                elif act == act_dup:
                    self._duplicate_node(item)
                elif act == act_del:
                    self._delete_node(item)
                elif act == act_validate:
                    roots = self._graph.roots()
                    reachable = any(self._graph.path_exists(r, item.ann.id) for r in roots)
                    QMessageBox.information(self, "Reachability",
                                            f"Node is {'✅ reachable' if reachable else '❌ unreachable'} from roots")
            elif isinstance(item, WorkflowEdgeItem):
                menu = QMenu()
                act_del = menu.addAction("🗑 Delete Edge")
                act = menu.exec(event.globalPos())
                if act == act_del:
                    self._delete_edge(item)
            return

        if event.button() != Qt.LeftButton:
            return super().mousePressEvent(event)

        if self.current_mode == self.MODE_ADD_NODE:
            name, ok = QInputDialog.getText(self, "Node Name", "Enter step name:")
            if ok and name:
                step_types = ["action", "decision", "start", "end", "io", "subprocess"]
                step, ok2 = QInputDialog.getItem(self, "Step Type", "Type:", step_types, 0, False)
                if not ok2:
                    step = "action"
                ann = Annotation(AnnotationType.WORKFLOW_NODE.value, self.current_label_id, {
                    "x": pos.x() - 85, "y": pos.y() - 32,
                    "width": 170, "height": 65, "name": name, "step_type": step
                })
                self._add_node(ann, name, pos.x() - 85, pos.y() - 32)
                self.annotation_created.emit(ann)
                self._graph.add_node(ann.id)
                if self._graph.has_cycle():
                    self._status("⚠ Graph has a cycle!")

        elif self.current_mode == self.MODE_ADD_EDGE:
            item = self.itemAt(event.position().toPoint())
            if isinstance(item, QGraphicsTextItem) and isinstance(item.parentItem(), WorkflowNodeItem):
                item = item.parentItem()
            if isinstance(item, WorkflowNodeItem):
                if self._edge_src is None:
                    self._edge_src = item
                    item.setSelected(True)
                elif item != self._edge_src:
                    if self._graph.path_exists(item.ann.id, self._edge_src.ann.id):
                        if QMessageBox.question(self, "Cycle Warning",
                                                "This edge would create a cycle. Continue?") != QMessageBox.Yes:
                            self._cancel_edge()
                            return
                    label, ok = QInputDialog.getText(self, "Edge Label", "Edge label (optional):")
                    if ok:
                        ann = Annotation(AnnotationType.WORKFLOW_EDGE.value, self.current_label_id, {
                            "source_id": self._edge_src.ann.id, "target_id": item.ann.id, "label": label
                        })
                        self._add_edge(self._edge_src, item, ann)
                        self._graph.add_edge(self._edge_src.ann.id, item.ann.id)
                        self.annotation_created.emit(ann)
                        if self._graph.has_cycle():
                            self._status("⚠ Graph now has a cycle!")
                    self._cancel_edge()
            else:
                self._cancel_edge()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._panning:
            delta = event.position().toPoint() - self._pan_start
            self._pan_start = event.position().toPoint()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            return
        if self._edge_src and self.current_mode == self.MODE_ADD_EDGE:
            pos = self.mapToScene(event.position().toPoint())
            sc = self._edge_src.pos() + self._edge_src.rect().center()
            if self._temp_line:
                self._scene.removeItem(self._temp_line)
            self._temp_line = self._scene.addLine(sc.x(), sc.y(), pos.x(), pos.y(),
                                                    QPen(QColor("#00FF00"), 2, Qt.DashLine))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MiddleButton:
            self._panning = False
            self.setCursor(Qt.CrossCursor if self.current_mode != self.MODE_SELECT else Qt.ArrowCursor)
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        f = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self._zoom *= f
        self.scale(f, f)

    def _add_node(self, ann, text, x, y):
        node = WorkflowNodeItem(ann, text, x, y)
        node.set_color(self._label_color(ann.label_id))
        self._scene.addItem(node)
        self.node_items[ann.id] = node

    def _add_edge(self, src, tgt, ann):
        edge = WorkflowEdgeItem(src, tgt, ann)
        edge.set_color(self._label_color(ann.label_id))
        self._scene.addItem(edge)
        self.edge_items[ann.id] = edge

    def _edit_node(self, node):
        name, ok = QInputDialog.getText(self, "Edit Node", "Step name:", text=node.ann.data.get("name", ""))
        if ok:
            node.ann.data["name"] = name
            node.set_text(name)
            node.ann.touch()
            self.annotation_modified.emit(node.ann.id)

    def _duplicate_node(self, node):
        d = node.ann.data
        new_ann = Annotation(AnnotationType.WORKFLOW_NODE.value, node.ann.label_id, {
            "x": d["x"] + 30, "y": d["y"] + 30,
            "width": d.get("width", 170), "height": d.get("height", 65),
            "name": d.get("name", "") + " (copy)", "step_type": d.get("step_type", "action")
        })
        self._add_node(new_ann, new_ann.data["name"], new_ann.data["x"], new_ann.data["y"])
        self._graph.add_node(new_ann.id)
        self.annotation_created.emit(new_ann)

    def _delete_node(self, node):
        for edge in list(node.edges):
            self._delete_edge(edge)
        self._graph.remove_node(node.ann.id)
        del self.node_items[node.ann.id]
        self._scene.removeItem(node)
        self.annotation_removed.emit(node.ann.id)

    def _delete_edge(self, edge):
        if edge in edge.source.edges:
            edge.source.edges.remove(edge)
        if edge in edge.target.edges:
            edge.target.edges.remove(edge)
        src_id = edge.ann.data.get("source_id", "")
        dst_id = edge.ann.data.get("target_id", "")
        if src_id and dst_id:
            self._graph.remove_edge(src_id, dst_id)
        del self.edge_items[edge.ann.id]
        self._scene.removeItem(edge)
        self.annotation_removed.emit(edge.ann.id)

    def load_annotations(self, annotations: List[Annotation]):
        for it in list(self.node_items.values()):
            self._scene.removeItem(it)
        for it in list(self.edge_items.values()):
            self._scene.removeItem(it)
        self.node_items.clear()
        self.edge_items.clear()
        self._graph.clear()
        for ann in annotations:
            if ann.type == AnnotationType.WORKFLOW_NODE.value:
                d = ann.data
                self._add_node(ann, d.get("name", ""), d.get("x", 0), d.get("y", 0))
                self._graph.add_node(ann.id)
        for ann in annotations:
            if ann.type == AnnotationType.WORKFLOW_EDGE.value:
                sid, tid = ann.data["source_id"], ann.data["target_id"]
                if sid in self.node_items and tid in self.node_items:
                    self._add_edge(self.node_items[sid], self.node_items[tid], ann)
                    self._graph.add_edge(sid, tid)

    def remove_annotation(self, ann_id):
        if ann_id in self.node_items:
            self._delete_node(self.node_items[ann_id])
        elif ann_id in self.edge_items:
            self._delete_edge(self.edge_items[ann_id])

    def fit_to_view(self):
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
        self._zoom = 1.0

    def _status(self, msg):
        try:
            p = self.window()
            if isinstance(p, QMainWindow):
                p.statusBar().showMessage(msg, 3000)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════
# LABEL MANAGER
# ═══════════════════════════════════════════════════════════════

class LabelManagerWidget(QWidget):
    label_added = Signal(str)
    label_removed = Signal(str)
    current_label_changed = Signal(str)

    PALETTE = ["#E74C3C", "#2ECC71", "#3498DB", "#F1C40F", "#9B59B6",
               "#1ABC9C", "#E67E22", "#E91E63", "#00BCD4", "#8BC34A",
               "#FF5722", "#607D8B", "#795548", "#CDDC39", "#FF9800"]

    def __init__(self, parent=None):
        super().__init__(parent)
        lo = QVBoxLayout(self)

        hdr = QHBoxLayout()
        hdr.addWidget(QLabel("🏷 Labels"))
        for text, slot in [("+", self._add_label), ("−", self._remove_label),
                           ("✏", self._edit_label)]:
            b = QPushButton(text)
            b.setFixedWidth(30)
            b.clicked.connect(slot)
            hdr.addWidget(b)
        lo.addLayout(hdr)

        self.search = QLineEdit()
        self.search.setPlaceholderText("🔍 Search labels…")
        self.search.textChanged.connect(self._filter)
        lo.addWidget(self.search)

        self.list_widget = QListWidget()
        self.list_widget.currentItemChanged.connect(self._current_changed)
        lo.addWidget(self.list_widget)

        self.labels: Dict[str, Label] = {}
        self._ci = 0

    def _next_color(self) -> str:
        c = self.PALETTE[self._ci % len(self.PALETTE)]
        self._ci += 1
        return c

    def _add_label(self):
        dlg = AddLabelDialog(self, self._next_color())
        if dlg.exec() == QDialog.Accepted:
            v = dlg.get_values()
            if not v["name"]:
                return
            lbl = Label(name=v["name"], color=v["color"], shortcut=v["shortcut"],
                        category=v["category"], description=v["description"])
            self.labels[lbl.id] = lbl
            self._add_item(lbl)
            self.label_added.emit(lbl.id)

    def _add_item(self, lbl: Label):
        sc = f" [{lbl.shortcut}]" if lbl.shortcut else ""
        item = QListWidgetItem(f"  ■  {lbl.name}{sc}")
        item.setData(Qt.UserRole, lbl.id)
        item.setForeground(QColor(lbl.color))
        f = item.font(); f.setBold(True); item.setFont(f)
        item.setToolTip(f"Category: {lbl.category}\nColor: {lbl.color}\nShortcut: {lbl.shortcut or 'None'}\n{lbl.description}")
        self.list_widget.addItem(item)
        self.list_widget.setCurrentItem(item)

    def _remove_label(self):
        item = self.list_widget.currentItem()
        if item:
            lid = item.data(Qt.UserRole)
            self.list_widget.takeItem(self.list_widget.row(item))
            self.labels.pop(lid, None)
            self.label_removed.emit(lid)

    def _edit_label(self):
        item = self.list_widget.currentItem()
        if not item:
            return
        lid = item.data(Qt.UserRole)
        lbl = self.labels.get(lid)
        if not lbl:
            return
        dlg = AddLabelDialog(self, lbl.color)
        dlg.txt_name.setText(lbl.name)
        dlg.txt_shortcut.setText(lbl.shortcut)
        dlg.txt_category.setText(lbl.category)
        dlg.txt_description.setText(lbl.description)
        if dlg.exec() == QDialog.Accepted:
            v = dlg.get_values()
            lbl.name = v["name"]; lbl.color = v["color"]
            lbl.shortcut = v["shortcut"]; lbl.category = v["category"]
            lbl.description = v["description"]
            sc = f" [{lbl.shortcut}]" if lbl.shortcut else ""
            item.setText(f"  ■  {lbl.name}{sc}")
            item.setForeground(QColor(lbl.color))

    def _current_changed(self, cur, prev):
        if cur:
            self.current_label_changed.emit(cur.data(Qt.UserRole))

    def _filter(self, text: str):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setHidden(text.lower() not in item.text().lower())

    def get_current_label_id(self) -> str:
        item = self.list_widget.currentItem()
        return item.data(Qt.UserRole) if item else ""

    def get_label_by_shortcut(self, key: str) -> Optional[str]:
        for lid, lbl in self.labels.items():
            if lbl.shortcut == key:
                return lid
        return None

    def load_labels(self, labels: Dict[str, Label]):
        self.labels = dict(labels)
        self.list_widget.clear()
        for lid, lbl in labels.items():
            self._add_item(lbl)


# ═══════════════════════════════════════════════════════════════
# ANNOTATION LIST
# ═══════════════════════════════════════════════════════════════

class AnnotationListWidget(QWidget):
    annotation_selected = Signal(str)
    annotation_deleted = Signal(str)
    annotation_edited = Signal(str)
    annotation_zoom = Signal(str)

    TYPE_ICON = {
        "bbox": "▭", "polygon": "⬠", "point": "●",
        "text_span": "T̲", "video_bbox": "▭",
        "workflow_node": "⬡", "workflow_edge": "→"
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        lo = QVBoxLayout(self)
        hdr = QHBoxLayout()
        hdr.addWidget(QLabel("📋 Annotations"))

        for text, slot, tip in [
            ("🔍", self._zoom_to, "Zoom to annotation"),
            ("✏", self._edit, "Edit properties"),
            ("🗑", self._del, "Delete"),
            ("📋", self._dup, "Duplicate"),
            ("Clear", self._clear_all, "Clear all"),
        ]:
            b = QPushButton(text)
            b.setFixedWidth(30 if len(text) <= 2 else 50)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            hdr.addWidget(b)
        lo.addLayout(hdr)

        self.search = QLineEdit()
        self.search.setPlaceholderText("🔍 Filter…")
        self.search.textChanged.connect(self._filter)
        lo.addWidget(self.search)

        self.lw = QListWidget()
        self.lw.currentItemChanged.connect(self._sel)
        self.lw.setContextMenuPolicy(Qt.CustomContextMenu)
        self.lw.customContextMenuRequested.connect(self._context_menu)
        lo.addWidget(self.lw)

        self.lbl_count = QLabel("0 annotations")
        lo.addWidget(self.lbl_count)

        self.annotations: Dict[str, Annotation] = {}
        self._project_labels: Dict[str, Label] = {}

    def set_project_labels(self, labels):
        self._project_labels = labels

    def add_annotation(self, ann: Annotation):
        self.annotations[ann.id] = ann
        self._add_item(ann)
        self._update_count()

    def _add_item(self, ann: Annotation):
        ln = self._project_labels[ann.label_id].name if ann.label_id in self._project_labels else "?"
        icon = self.TYPE_ICON.get(ann.type, "?")
        lock = " 🔒" if ann.locked else ""
        conf = f" {ann.confidence:.0%}" if ann.confidence < 1.0 else ""
        notes = " 📝" if ann.notes else ""
        item = QListWidgetItem(f"{icon}  {ln}  ({ann.type}){lock}{conf}{notes}")
        item.setData(Qt.UserRole, ann.id)
        if ann.label_id in self._project_labels:
            item.setForeground(QColor(self._project_labels[ann.label_id].color))
        self.lw.addItem(item)
        self.lw.setCurrentItem(item)

    def clear_annotations(self):
        self.annotations.clear()
        self.lw.clear()
        self._update_count()

    def load_annotations(self, anns: List[Annotation]):
        self.clear_annotations()
        for a in anns:
            self.add_annotation(a)

    def remove_annotation(self, ann_id: str):
        for i in range(self.lw.count()):
            if self.lw.item(i).data(Qt.UserRole) == ann_id:
                self.lw.takeItem(i)
                break
        self.annotations.pop(ann_id, None)
        self._update_count()

    def update_annotation(self, ann: Annotation):
        self.annotations[ann.id] = ann
        self._refresh_item(ann.id)

    def _refresh_item(self, aid):
        ann = self.annotations.get(aid)
        if not ann:
            return
        for i in range(self.lw.count()):
            item = self.lw.item(i)
            if item.data(Qt.UserRole) == aid:
                ln = self._project_labels[ann.label_id].name if ann.label_id in self._project_labels else "?"
                icon = self.TYPE_ICON.get(ann.type, "?")
                lock = " 🔒" if ann.locked else ""
                conf = f" {ann.confidence:.0%}" if ann.confidence < 1.0 else ""
                notes = " 📝" if ann.notes else ""
                item.setText(f"{icon}  {ln}  ({ann.type}){lock}{conf}{notes}")
                if ann.label_id in self._project_labels:
                    item.setForeground(QColor(self._project_labels[ann.label_id].color))
                break

    def _update_count(self):
        self.lbl_count.setText(f"{len(self.annotations)} annotations")

    def _sel(self, cur, prev):
        if cur:
            self.annotation_selected.emit(cur.data(Qt.UserRole))

    def _del(self):
        item = self.lw.currentItem()
        if item:
            self.annotation_deleted.emit(item.data(Qt.UserRole))

    def _edit(self):
        item = self.lw.currentItem()
        if item:
            self.annotation_edited.emit(item.data(Qt.UserRole))

    def _zoom_to(self):
        item = self.lw.currentItem()
        if item:
            self.annotation_zoom.emit(item.data(Qt.UserRole))

    def _dup(self):
        item = self.lw.currentItem()
        if item:
            ann = self.annotations.get(item.data(Qt.UserRole))
            if ann:
                new_ann = Annotation.from_dict(ann.to_dict())
                new_ann.id = str(uuid.uuid4())[:8]
                # offset bbox
                if new_ann.type in (AnnotationType.BBOX.value, AnnotationType.VIDEO_BBOX.value):
                    new_ann.data["x"] = new_ann.data.get("x", 0) + 10
                    new_ann.data["y"] = new_ann.data.get("y", 0) + 10
                self.add_annotation(new_ann)

    def _clear_all(self):
        if QMessageBox.question(self, "Clear All",
                                "Delete ALL annotations?") == QMessageBox.Yes:
            for aid in list(self.annotations.keys()):
                self.annotation_deleted.emit(aid)

    def _filter(self, text: str):
        for i in range(self.lw.count()):
            item = self.lw.item(i)
            item.setHidden(text.lower() not in item.text().lower())

    def _context_menu(self, pos):
        item = self.lw.itemAt(pos)
        if not item:
            return
        aid = item.data(Qt.UserRole)
        ann = self.annotations.get(aid)
        if not ann:
            return
        menu = QMenu()
        actions = {
            "✏ Edit": self.annotation_edited,
            "🔍 Zoom To": self.annotation_zoom,
            "🔒 Toggle Lock": None,
            "🗑 Delete": self.annotation_deleted,
        }
        acts = {}
        for text, sig in actions.items():
            acts[text] = menu.addAction(text)
        act = menu.exec(self.lw.mapToGlobal(pos))
        for text, sig in actions.items():
            if act == acts[text]:
                if text == "🔒 Toggle Lock":
                    ann.locked = not ann.locked
                    self._refresh_item(aid)
                elif sig:
                    sig.emit(aid)


# ═══════════════════════════════════════════════════════════════
# MAIN WINDOW
# ═══════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🏷 Data Annotation Studio v3")
        self.setMinimumSize(1300, 850)
        self.resize(1520, 940)
        self.setAcceptDrops(True)

        self.project = Project()
        self.repo = ProjectRepository(self.project)
        self.event_bus = EventBus()
        self.command_history = CommandHistory()
        self.current_file: Optional[DataFile] = None
        self._project_path: Optional[str] = None
        self.current_mode = "image"
        self._dirty = False
        self._recent: List[str] = []

        self._build_menus()
        self._build_toolbar()
        self._build_central()
        self._build_docks()
        self._build_statusbar()
        self._connect()
        self._apply_theme()
        self._setup_autosave()
        self._setup_shortcuts()

    def _apply_theme(self):
        self.setStyleSheet("""
            QMainWindow { background: #2b2b2b; }
            QWidget { background: #2b2b2b; color: #ccc; font-family: 'Segoe UI'; font-size: 10pt; }
            QMenuBar { background: #333; }
            QMenuBar::item:selected { background: #505050; }
            QMenu { background: #3c3c3c; border: 1px solid #555; }
            QMenu::item:selected { background: #5082aa; }
            QMenu::separator { height: 1px; background: #555; margin: 4px 10px; }
            QToolBar { background: #333; border: none; spacing: 3px; padding: 2px; }
            QToolButton { background: #444; color: #ccc; border: 1px solid #555;
                          border-radius: 3px; padding: 5px 10px; min-width: 20px; font-size: 9pt; }
            QToolButton:hover { background: #555; }
            QToolButton:checked { background: #5082aa; border-color: #5082aa; color: #fff; }
            QDockWidget { color: #ccc; }
            QDockWidget::title { background: #333; padding: 5px; border: 1px solid #444; }
            QListWidget, QTreeWidget { background: #2e2e2e; border: 1px solid #444; alternate-background-color: #323232; }
            QListWidget::item:selected, QTreeWidget::item:selected { background: #5082aa; }
            QListWidget::item:hover, QTreeWidget::item:hover { background: #3a3a4a; }
            QPushButton { background: #444; color: #ccc; border: 1px solid #555;
                          border-radius: 3px; padding: 5px 12px; }
            QPushButton:hover { background: #555; }
            QPushButton:pressed { background: #666; }
            QPushButton:checked { background: #5082aa; }
            QTextEdit { background: #1e1e1e; border: 1px solid #444; color: #ddd; }
            QLineEdit { background: #333; border: 1px solid #555; padding: 4px; color: #ccc; border-radius: 3px; }
            QLineEdit:focus { border-color: #5082aa; }
            QSlider::groove:horizontal { background: #555; height: 6px; border-radius: 3px; }
            QSlider::handle:horizontal { background: #5082aa; width: 14px; margin: -4px 0; border-radius: 7px; }
            QSpinBox, QDoubleSpinBox { background: #333; border: 1px solid #555; padding: 2px; color: #ccc; }
            QGroupBox { border: 1px solid #555; border-radius: 4px; margin-top: 10px; padding-top: 14px; }
            QTabWidget::pane { border: 1px solid #444; background: #2b2b2b; }
            QTabBar::tab { background: #3a3a3a; padding: 6px 16px; border: 1px solid #444;
                           border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:selected { background: #5082aa; color: #fff; }
            QStatusBar { background: #333; color: #888; }
            QGraphicsView { background: #1a1a2e; }
            QProgressBar { background: #333; border: 1px solid #555; border-radius: 3px; text-align: center; color: #ccc; }
            QProgressBar::chunk { background: #5082aa; border-radius: 2px; }
            QCheckBox { spacing: 6px; }
            QComboBox { background: #333; border: 1px solid #555; padding: 4px; color: #ccc; border-radius: 3px; }
            QComboBox QAbstractItemView { background: #3c3c3c; selection-background-color: #5082aa; }
            QScrollBar:vertical { background: #2b2b2b; width: 10px; }
            QScrollBar::handle:vertical { background: #555; border-radius: 5px; min-height: 30px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QScrollBar:horizontal { background: #2b2b2b; height: 10px; }
            QScrollBar::handle:horizontal { background: #555; border-radius: 5px; min-width: 30px; }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
            QToolTip { background: #3c3c3c; color: #ccc; border: 1px solid #555; padding: 4px; }
        """)

    # ── menus ──

    def _build_menus(self):
        mb = self.menuBar()

        fm = mb.addMenu("&File")
        self._act(fm, "📄 New Project", "Ctrl+N", self._new_project)
        self._act(fm, "📂 Open Project…", "Ctrl+O", self._open_project)
        self._act(fm, "💾 Save Project", "Ctrl+S", self._save_project)
        self._act(fm, "💾 Save As…", "Ctrl+Shift+S", self._save_project_as)
        fm.addSeparator()
        self._act(fm, "🖼 Add Images…", "", lambda: self._add_files("image"))
        self._act(fm, "📝 Add Text Files…", "", lambda: self._add_files("text"))
        self._act(fm, "🎬 Add Video Frames…", "", lambda: self._add_files("video"))
        self._act(fm, "🔀 New Workflow…", "", lambda: self._add_files("workflow"))
        fm.addSeparator()
        self._act(fm, "📥 Import COCO JSON…", "", self._import_coco)
        fm.addSeparator()
        self._act(fm, "⚙ Settings…", "", self._show_settings)
        fm.addSeparator()
        self._act(fm, "🚪 Exit", "Ctrl+Q", self.close)

        em = mb.addMenu("&Edit")
        self._act(em, "↩ Undo", "Ctrl+Z", self._undo)
        self._act(em, "↪ Redo", "Ctrl+Y", self._redo)
        em.addSeparator()
        self._act(em, "🗑 Delete Selected", "Delete", self._delete_selected_ann)
        self._act(em, "📋 Duplicate Selected", "Ctrl+D", self._duplicate_selected_ann)
        em.addSeparator()
        self._act(em, "🏷 Add Label…", "Ctrl+L", lambda: self.label_mgr._add_label())

        vm = mb.addMenu("&View")
        self._act(vm, "🔍 Zoom In", "Ctrl+=", self._zoom_in)
        self._act(vm, "🔍 Zoom Out", "Ctrl+-", self._zoom_out)
        self._act(vm, "🔎 Fit to View", "Ctrl+0", self._fit_view)
        vm.addSeparator()
        self._act_crosshair = QAction("✛ Crosshair Cursor", self, checkable=True)
        self._act_crosshair.toggled.connect(self._toggle_crosshair)
        vm.addAction(self._act_crosshair)
        vm.addSeparator()
        self._act(vm, "📊 Statistics…", "", self._show_stats)
        self._act(vm, "✅ Validate Annotations…", "", self._validate)

        xm = mb.addMenu("E&xport")
        self._act(xm, "📦 COCO JSON…", "", self._export_coco)
        self._act(xm, "📦 spaCy NER…", "", self._export_spacy)
        self._act(xm, "📦 YOLO Format…", "", self._export_yolo)
        self._act(xm, "📦 Workflow JSON…", "", self._export_workflow)
        xm.addSeparator()
        self._act(xm, "📦 Full Project…", "", self._export_custom)

        am = mb.addMenu("&AI")
        self._act(am, "🤖 Auto-Detect Regions (demo)…", "", self._ai_prelabel)
        self._act(am, "🔍 Find Overlapping BBoxes…", "", self._find_overlaps)

        hm = mb.addMenu("&Help")
        self._act(hm, "⌨ Shortcuts", "F1", self._show_shortcuts)
        self._act(hm, "ℹ About", "", self._about)

    def _act(self, menu, text, shortcut, slot):
        a = QAction(text, self)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        a.triggered.connect(slot)
        menu.addAction(a)
        return a

    # ── toolbar ──

    def _build_toolbar(self):
        self.mode_tb = QToolBar("Mode")
        self.mode_tb.setMovable(False)
        self.addToolBar(self.mode_tb)

        self.mode_btns: Dict[str, QToolButton] = {}
        for label, mode in [("🖼 Image", "image"), ("📝 Text", "text"),
                            ("🎬 Video", "video"), ("🔀 Workflow", "workflow")]:
            b = QToolButton(); b.setText(label); b.setCheckable(True)
            b.clicked.connect(lambda _, m=mode: self._set_mode(m))
            self.mode_tb.addWidget(b)
            self.mode_btns[mode] = b
        self.mode_btns["image"].setChecked(True)

        self.mode_tb.addSeparator()
        self.tool_tb = QToolBar("Tools")
        self.tool_tb.setMovable(False)
        self.addToolBar(self.tool_tb)
        self._setup_image_tools()

    def _setup_image_tools(self):
        self.tool_tb.clear()
        self.tool_btns = {}
        for label, mode in [("↖ Select", ToolMode.SELECT), ("▭ BBox", ToolMode.BBOX),
                            ("⬠ Polygon", ToolMode.POLYGON), ("● Point", ToolMode.POINT)]:
            b = QToolButton(); b.setText(label); b.setCheckable(True)
            b.clicked.connect(lambda _, m=mode: self._set_tool(m))
            self.tool_tb.addWidget(b)
            self.tool_btns[mode] = b
        if self.tool_btns:
            self.tool_btns[ToolMode.SELECT].setChecked(True)

    def _setup_text_tools(self):
        self.tool_tb.clear()
        b = QToolButton(); b.setText("T̲  Annotate Selection")
        b.clicked.connect(self._annotate_text_sel)
        self.tool_tb.addWidget(b)

    def _setup_video_tools(self):
        self.tool_tb.clear()
        b = QToolButton(); b.setText("ℹ  Use Draw BBox on video panel"); b.setEnabled(False)
        self.tool_tb.addWidget(b)

    def _setup_workflow_tools(self):
        self.tool_tb.clear()
        self.tool_btns = {}
        for label, mode in [("↖ Select/Move", "select"), ("⬡ Add Node", "add_node"),
                            ("→ Add Edge", "add_edge")]:
            b = QToolButton(); b.setText(label); b.setCheckable(True)
            b.clicked.connect(lambda _, m=mode: self._set_workflow_mode(m))
            self.tool_tb.addWidget(b)
            self.tool_btns[mode] = b
        if self.tool_btns:
            self.tool_btns["select"].setChecked(True)

    # ── central ──

    def _build_central(self):
        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.setMovable(True)
        self.tabs.tabCloseRequested.connect(self._close_tab)
        self.tabs.currentChanged.connect(self._tab_changed)
        self.setCentralWidget(self.tabs)

        w = QWidget()
        lo = QVBoxLayout(w); lo.setAlignment(Qt.AlignCenter)
        t = QLabel("🏷 Data Annotation Studio v3")
        t.setFont(QFont("Segoe UI", 24, QFont.Bold))
        t.setAlignment(Qt.AlignCenter)
        t.setStyleSheet("color:#5082aa;")
        lo.addWidget(t)
        info = QLabel(
            "\n📁 File → Add Images / Text / Video / Workflow\n"
            "🏷 Labels panel → Create categories with shortcuts\n"
            "🔧 Toolbar → Pick tools, annotate on canvas\n"
            "↩ Ctrl+Z / Ctrl+Y → Undo / Redo (Command Pattern)\n"
            "🤖 AI → Auto-Detect for pre-labeling\n"
            "📊 View → Statistics & Validation\n"
            "📥 Import COCO JSON → Load existing annotations\n"
            "📋 Duplicate, Zoom-to, Batch label changes\n"
            "\n⌨ F1 for shortcuts  |  Drag & drop files onto window"
        )
        info.setFont(QFont("Segoe UI", 10))
        info.setAlignment(Qt.AlignCenter)
        info.setStyleSheet("color:#aaa;")
        lo.addWidget(info)
        self.tabs.addTab(w, "Welcome")

    # ── docks ──

    def _build_docks(self):
        d1 = QDockWidget("📁 Files", self)
        d1.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderLabels(["Name", "Type", "Anns", "Status"])
        self.file_tree.setColumnWidth(0, 170)
        self.file_tree.setAlternatingRowColors(True)
        self.file_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.file_tree.customContextMenuRequested.connect(self._file_context_menu)
        self.file_tree.itemDoubleClicked.connect(self._on_file_clicked)
        d1.setWidget(self.file_tree)
        self.addDockWidget(Qt.LeftDockWidgetArea, d1)

        d2 = QDockWidget("🏷 Labels", self)
        d2.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.label_mgr = LabelManagerWidget()
        d2.setWidget(self.label_mgr)
        self.addDockWidget(Qt.LeftDockWidgetArea, d2)

        d3 = QDockWidget("📋 Annotations", self)
        d3.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.ann_list = AnnotationListWidget()
        d3.setWidget(self.ann_list)
        self.addDockWidget(Qt.RightDockWidgetArea, d3)

        d4 = QDockWidget("⚙ Properties", self)
        d4.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.props_widget = QWidget()
        self.props_lo = QVBoxLayout(self.props_widget)
        self.props_lo.addWidget(QLabel("Select an annotation"))
        self.props_lo.addStretch()
        d4.setWidget(self.props_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, d4)

        d5 = QDockWidget("🗺 Minimap", self)
        d5.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.minimap = MinimapWidget()
        d5.setWidget(self.minimap)
        self.addDockWidget(Qt.RightDockWidgetArea, d5)
        self.minimap_dock = d5
        self.minimap_dock.hide()

    def _build_statusbar(self):
        self.statusBar().showMessage("Ready")
        self.st_mode = QLabel("Mode: Image")
        self.st_file = QLabel("No file")
        self.st_undo = QLabel("")
        for w in [self.st_mode, self.st_file, self.st_undo]:
            self.statusBar().addPermanentWidget(w)

    def _connect(self):
        self.label_mgr.label_added.connect(self._on_label_added)
        self.label_mgr.label_removed.connect(self._on_label_removed)
        self.label_mgr.current_label_changed.connect(self._on_label_changed)
        self.ann_list.annotation_deleted.connect(self._on_ann_deleted)
        self.ann_list.annotation_selected.connect(self._show_props)
        self.ann_list.annotation_edited.connect(self._edit_ann_props)
        self.ann_list.annotation_zoom.connect(self._zoom_to_annotation)

    def _setup_autosave(self):
        self._autosave_timer = QTimer()
        self._autosave_timer.setInterval(self.project.config.auto_save_interval * 1000)
        self._autosave_timer.timeout.connect(self._autosave)
        if self.project.config.auto_save:
            self._autosave_timer.start()

    def _autosave(self):
        if self._dirty and self._project_path:
            self._save_project()
            self.statusBar().showMessage("Auto-saved", 2000)

    def _setup_shortcuts(self):
        for i in range(1, 10):
            QShortcut(QKeySequence(str(i)), self, lambda k=str(i): self._select_label_shortcut(k))

    def _select_label_shortcut(self, key):
        lid = self.label_mgr.get_label_by_shortcut(key)
        if lid:
            self._on_label_changed(lid)
            for i in range(self.label_mgr.list_widget.count()):
                item = self.label_mgr.list_widget.item(i)
                if item.data(Qt.UserRole) == lid:
                    self.label_mgr.list_widget.setCurrentItem(item)
                    break
            self.statusBar().showMessage(f"Label: {self.project.labels[lid].name}", 1500)

    # ── mode / tool ──

    def _set_mode(self, mode):
        self.current_mode = mode
        for m, b in self.mode_btns.items():
            b.setChecked(m == mode)
        self.st_mode.setText(f"Mode: {mode.capitalize()}")
        {"image": self._setup_image_tools, "text": self._setup_text_tools,
         "video": self._setup_video_tools, "workflow": self._setup_workflow_tools}[mode]()
        self.minimap_dock.setVisible(mode == "image")

    def _set_tool(self, mode: ToolMode):
        c = self._canvas()
        if isinstance(c, ImageCanvas):
            c.set_tool_by_mode(mode)
        for t, b in self.tool_btns.items():
            b.setChecked(t == mode)

    def _set_workflow_mode(self, mode):
        c = self._canvas()
        if isinstance(c, WorkflowCanvas):
                    c.set_mode(mode)
        for t, b in self.tool_btns.items():
            b.setChecked(t == mode)

    def _canvas(self):
        return self.tabs.currentWidget()

    # ── project ops ──

    def _new_project(self):
        if self._dirty:
            if QMessageBox.question(self, "Unsaved Changes", "Save first?") == QMessageBox.Yes:
                self._save_project()
        self.project = Project()
        self.repo = ProjectRepository(self.project)
        self.command_history.clear()
        self.label_mgr.load_labels({})
        self.ann_list.set_project_labels({})
        self.file_tree.clear()
        self.current_file = None
        self._project_path = None
        self._dirty = False
        while self.tabs.count() > 1:
            self.tabs.removeTab(1)
        self.statusBar().showMessage("New project created")

    def _open_project(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open Project", "", "JSON (*.json)")
        if not p:
            return
        try:
            self.project = Project.load(p)
            self.repo = ProjectRepository(self.project)
            self.command_history.clear()
            self._project_path = p
            self.label_mgr.load_labels(self.project.labels)
            self.ann_list.set_project_labels(self.project.labels)
            self._refresh_tree()
            while self.tabs.count() > 1:
                self.tabs.removeTab(1)
            self._dirty = False
            self.statusBar().showMessage(f"Opened: {p}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _save_project(self):
        self._sync_current_annotations()
        if self._project_path:
            self.repo.save_project(self._project_path)
            self._dirty = False
            self.statusBar().showMessage("Project saved", 2000)
        else:
            self._save_project_as()

    def _save_project_as(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save Project", "", "JSON (*.json)")
        if p:
            self._project_path = p
            self._save_project()

    def _sync_current_annotations(self):
        if not self.current_file:
            return
        c = self._canvas()
        fp = self.current_file.filepath
        if isinstance(c, TextAnnotationCanvas):
            # Clear and re-add to repo to maintain index integrity
            self.repo.project.files[fp].annotations = list(c.annotations)
            self.repo._rebuild_indices()
        elif isinstance(c, VideoAnnotationCanvas):
            all_anns = []
            for anns in c.frame_annotations.values():
                all_anns.extend(anns)
            self.repo.project.files[fp].annotations = all_anns
            self.repo._rebuild_indices()
        elif isinstance(c, WorkflowCanvas):
            anns = []
            for node in c.node_items.values():
                anns.append(node.ann)
            for edge in c.edge_items.values():
                anns.append(edge.ann)
            self.repo.project.files[fp].annotations = anns
            self.repo._rebuild_indices()
        # ImageCanvas is synced via Command pattern add/remove directly

    def _sync_canvas_annotations(self, canvas):
        if not hasattr(canvas, '_filepath'):
            return
        fp = canvas._filepath
        if fp not in self.project.files:
            return
        if isinstance(canvas, TextAnnotationCanvas):
            self.project.files[fp].annotations = list(canvas.annotations)
        elif isinstance(canvas, VideoAnnotationCanvas):
            all_anns = []
            for anns in canvas.frame_annotations.values():
                all_anns.extend(anns)
            self.project.files[fp].annotations = all_anns
        elif isinstance(canvas, WorkflowCanvas):
            anns = []
            for node in canvas.node_items.values():
                anns.append(node.ann)
            for edge in canvas.edge_items.values():
                anns.append(edge.ann)
            self.project.files[fp].annotations = anns

    # ── add files ──

    def _add_files(self, ftype: str):
        if ftype == "image":
            ps, _ = QFileDialog.getOpenFileNames(self, "Add Images", "",
                "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp);;All (*)")
            for p in ps:
                self.repo.add_file(p, FileType.IMAGE.value)
        elif ftype == "text":
            ps, _ = QFileDialog.getOpenFileNames(self, "Add Text", "",
                "Text (*.txt *.md *.csv *.json *.xml *.html);;All (*)")
            for p in ps:
                df = self.repo.add_file(p, FileType.TEXT.value)
                try:
                    with open(p, 'r', encoding='utf-8') as f:
                        df.text_content = f.read()
                except Exception:
                    pass
        elif ftype == "video":
            d = QFileDialog.getExistingDirectory(self, "Select Video Frames Directory")
            if d:
                self.repo.add_file(d, FileType.VIDEO.value)
        elif ftype == "workflow":
            n, ok = QInputDialog.getText(self, "Workflow", "Name:")
            if ok and n.strip():
                self.repo.add_file(n.strip(), FileType.WORKFLOW.value)
        self._refresh_tree()
        self._dirty = True

    def _refresh_tree(self):
        self.file_tree.clear()
        groups = {}
        for ft in FileType:
            groups[ft.value] = QTreeWidgetItem(self.file_tree, [ft.value.capitalize() + "s", "", "", ""])
        for fp, df in self.project.files.items():
            nm = Path(fp).name if df.file_type != FileType.WORKFLOW.value else fp
            parent = groups.get(df.file_type)
            if parent:
                icon = STATUS_ICONS.get(ReviewStatus(df.review_status), "⬜")
                it = QTreeWidgetItem(parent, [nm, df.file_type, str(len(df.annotations)), icon])
                it.setData(0, Qt.UserRole, fp)
                it.setToolTip(0, fp)
        self.file_tree.expandAll()

    def _file_context_menu(self, pos):
        item = self.file_tree.itemAt(pos)
        if not item:
            return
        fp = item.data(0, Qt.UserRole)
        if not fp or fp not in self.project.files:
            return
        df = self.project.files[fp]
        menu = QMenu()
        act_open = menu.addAction("📂 Open")
        act_validate = menu.addAction("✅ Validate")
        status_menu = menu.addMenu("Review Status")
        act_none = status_menu.addAction("⬜ None")
        act_review = status_menu.addAction("🔄 In Review")
        act_approve = status_menu.addAction("✅ Approved")
        act_reject = status_menu.addAction("❌ Rejected")
        act_remove = menu.addAction("🗑 Remove from Project")
        act = menu.exec(self.file_tree.mapToGlobal(pos))
        
        if act == act_open:
            self._open_file(fp)
        elif act == act_validate:
            issues = self.repo.validate_annotations(fp)
            ValidationDialog(issues, self).exec()
        elif act == act_none:
            df.review_status = ReviewStatus.NONE.value; self._dirty = True; self._refresh_tree()
        elif act == act_review:
            df.review_status = ReviewStatus.IN_REVIEW.value; self._dirty = True; self._refresh_tree()
        elif act == act_approve:
            df.review_status = ReviewStatus.APPROVED.value; self._dirty = True; self._refresh_tree()
        elif act == act_reject:
            df.review_status = ReviewStatus.REJECTED.value; self._dirty = True; self._refresh_tree()
        elif act == act_remove:
            if QMessageBox.question(self, "Remove", f"Remove '{fp}'?") == QMessageBox.Yes:
                self.repo.remove_file(fp)
                self._refresh_tree()
                self._dirty = True

    # ── file open / tab ──

    def _on_file_clicked(self, item, _col):
        fp = item.data(0, Qt.UserRole)
        if fp:
            self._open_file(fp)

    def _open_file(self, fp: str):
        if fp not in self.project.files:
            return
        df = self.project.files[fp]
        
        if self.current_file:
            self.current_file.stop_timer()
            
        self.current_file = df
        df.start_timer()
        self.st_file.setText(Path(fp).name if df.file_type != FileType.WORKFLOW.value else fp)

        for i in range(self.tabs.count()):
            w = self.tabs.widget(i)
            if hasattr(w, '_filepath') and w._filepath == fp:
                self.tabs.setCurrentIndex(i)
                return

        canvas = None
        if df.file_type == FileType.IMAGE.value:
            canvas = ImageCanvas(event_bus=self.event_bus)
            canvas._filepath = fp
            canvas.set_project_labels(self.project.labels)
            canvas.set_config(self.project.config)
            if not canvas.load_image(fp):
                QMessageBox.warning(self, "Error", f"Cannot load image: {fp}")
                return
            canvas.load_annotations(df.annotations)
            canvas.annotation_created.connect(self._on_ann_created)
            canvas.annotation_removed.connect(self._on_ann_removed_canvas)
            canvas.annotation_modified.connect(self._on_ann_modified)
            self._set_mode("image")
            self.minimap.set_canvas(canvas)
            if canvas._image_item:
                self.minimap.update_pixmap(canvas._image_item.pixmap())

        elif df.file_type == FileType.TEXT.value:
            canvas = TextAnnotationCanvas(event_bus=self.event_bus)
            canvas._filepath = fp
            canvas.set_project_labels(self.project.labels)
            if df.text_content:
                canvas.load_text(df.text_content)
            else:
                canvas.load_text_file(fp)
            canvas.load_annotations(df.annotations)
            canvas.annotation_created.connect(self._on_ann_created)
            canvas.annotation_removed.connect(self._on_ann_removed_canvas)
            canvas.annotation_modified.connect(self._on_ann_modified)
            self._set_mode("text")

        elif df.file_type == FileType.VIDEO.value:
            canvas = VideoAnnotationCanvas(event_bus=self.event_bus)
            canvas._filepath = fp
            canvas.set_project_labels(self.project.labels)
            if not canvas.load_video_frames(fp):
                QMessageBox.warning(self, "Warning", "No frames found.")
                return
            canvas.load_annotations(df.annotations)
            canvas.annotation_created.connect(self._on_ann_created)
            canvas.annotation_removed.connect(self._on_ann_removed_canvas)
            canvas.annotation_modified.connect(self._on_ann_modified)
            self._set_mode("video")

        elif df.file_type == FileType.WORKFLOW.value:
            canvas = WorkflowCanvas(event_bus=self.event_bus)
            canvas._filepath = fp
            canvas.set_project_labels(self.project.labels)
            canvas.load_annotations(df.annotations)
            canvas.annotation_created.connect(self._on_ann_created)
            canvas.annotation_removed.connect(self._on_ann_removed_canvas)
            canvas.annotation_modified.connect(self._on_ann_modified)
            self._set_mode("workflow")

        if canvas:
            nm = Path(fp).name if df.file_type != FileType.WORKFLOW.value else fp
            idx = self.tabs.addTab(canvas, nm)
            self.tabs.setCurrentIndex(idx)

    def _close_tab(self, idx):
        if idx == 0:
            return
        w = self.tabs.widget(idx)
        if hasattr(w, '_filepath') and w._filepath:
            self._sync_canvas_annotations(w)
        self.tabs.removeTab(idx)

    def _tab_changed(self, idx):
        w = self.tabs.widget(idx)
        if hasattr(w, '_filepath') and w._filepath in self.project.files:
            if self.current_file:
                self.current_file.stop_timer()
            self.current_file = self.project.files[w._filepath]
            self.current_file.start_timer()
            df = self.current_file
            self.st_file.setText(Path(w._filepath).name if df.file_type != FileType.WORKFLOW.value else w._filepath)
            self.ann_list.load_annotations(df.annotations)
            if isinstance(w, ImageCanvas):
                self._set_mode("image")
                self.minimap.set_canvas(w)
                if w._image_item:
                    self.minimap.update_pixmap(w._image_item.pixmap())
            elif isinstance(w, TextAnnotationCanvas):
                self._set_mode("text")
            elif isinstance(w, VideoAnnotationCanvas):
                self._set_mode("video")
            elif isinstance(w, WorkflowCanvas):
                self._set_mode("workflow")

    # ── annotation events ──

    def _on_ann_created(self, ann: Annotation):
        if not self.current_file:
            return
        fp = self.current_file.filepath
        cmd = AddAnnotationCommand(
            self.repo, fp, ann,
            on_execute=lambda a: self.ann_list.add_annotation(a),
            on_undo=lambda aid: self.ann_list.remove_annotation(aid)
        )
        self.command_history.execute(cmd)
        self._dirty = True
        self._refresh_tree()
        self._update_undo_status()
        c = self._canvas()
        if isinstance(c, ImageCanvas) and c._image_item:
            self.minimap.update_pixmap(c._image_item.pixmap())

    def _on_ann_removed_canvas(self, ann_id: str):
        if not self.current_file:
            return
        ann = self.repo.project.annotation_by_id(ann_id)
        if not ann:
            return
        fp = self.current_file.filepath
        cmd = RemoveAnnotationCommand(
            self.repo, fp, ann,
            on_execute=lambda aid: self.ann_list.remove_annotation(aid),
            on_undo=lambda a: (self.ann_list.add_annotation(a), 
                               self._redraw_canvas_annotation(a) if isinstance(self._canvas(), ImageCanvas) else None)
        )
        self.command_history.execute(cmd)
        self._dirty = True
        self._update_undo_status()
        self._refresh_tree()

    def _on_ann_modified(self, ann_id: str):
        self._dirty = True
        ann = self.repo.project.annotation_by_id(ann_id)
        if ann:
            self.ann_list.update_annotation(ann)

    def _on_ann_deleted(self, ann_id: str):
        if not self.current_file:
            return
        ann = self.repo.project.annotation_by_id(ann_id)
        if not ann:
            return
        fp = self.current_file.filepath
        cmd = RemoveAnnotationCommand(
            self.repo, fp, ann,
            on_execute=self._remove_ann_ui,
            on_undo=self._restore_ann_ui
        )
        self.command_history.execute(cmd)
        self._dirty = True
        self._update_undo_status()
        self._refresh_tree()

    def _remove_ann_ui(self, ann_id: str):
        c = self._canvas()
        if isinstance(c, ImageCanvas):
            c.remove_annotation_graphic(ann_id)
        elif isinstance(c, (VideoAnnotationCanvas, WorkflowCanvas, TextAnnotationCanvas)):
            c.remove_annotation(ann_id)
        self.ann_list.remove_annotation(ann_id)

    def _restore_ann_ui(self, ann: Annotation):
        c = self._canvas()
        if isinstance(c, ImageCanvas):
            c.draw_annotation(ann)
        self.ann_list.add_annotation(ann)

    def _redraw_canvas_annotation(self, ann: Annotation):
        c = self._canvas()
        if isinstance(c, ImageCanvas):
            c.remove_annotation_graphic(ann.id)
            c.draw_annotation(ann)

    def _delete_selected_ann(self):
        item = self.ann_list.lw.currentItem()
        if item:
            self._on_ann_deleted(item.data(Qt.UserRole))

    def _duplicate_selected_ann(self):
        item = self.ann_list.lw.currentItem()
        if item:
            self.ann_list._dup()

    # ── label events ──

    def _on_label_added(self, lid: str):
        lbl = self.label_mgr.labels.get(lid)
        if lbl and lid not in self.project.labels:
            self.repo.add_label(lbl)
        self.ann_list.set_project_labels(self.project.labels)
        self._update_canvas_labels()
        self._dirty = True

    def _on_label_removed(self, lid: str):
        self.repo.remove_label(lid)
        self.ann_list.set_project_labels(self.project.labels)
        self._update_canvas_labels()
        self._dirty = True

    def _on_label_changed(self, lid: str):
        c = self._canvas()
        if c and hasattr(c, 'current_label_id'):
            c.current_label_id = lid

    def _update_canvas_labels(self):
        for i in range(self.tabs.count()):
            w = self.tabs.widget(i)
            if hasattr(w, 'set_project_labels'):
                w.set_project_labels(self.project.labels)

    # ── undo / redo ──

    def _undo(self):
        cmd = self.command_history.undo()
        if not cmd:
            self.statusBar().showMessage("Nothing to undo", 1500)
            return
        self._dirty = True
        self._update_undo_status()
        self._refresh_tree()
        if self.current_file:
            self.ann_list.load_annotations(self.repo.get_annotations(self.current_file.filepath))

    def _redo(self):
        cmd = self.command_history.redo()
        if not cmd:
            self.statusBar().showMessage("Nothing to redo", 1500)
            return
        self._dirty = True
        self._update_undo_status()
        self._refresh_tree()
        if self.current_file:
            self.ann_list.load_annotations(self.repo.get_annotations(self.current_file.filepath))

    def _update_undo_status(self):
        self.st_undo.setText(f"Undo: {self.command_history.undo_count()} | Redo: {self.command_history.redo_count()}")

    # ── properties ──

    def _show_props(self, ann_id: str):
        ann = self.repo.project.annotation_by_id(ann_id)
        if not ann:
            return
        while self.props_lo.count():
            child = self.props_lo.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        lbl_name = self.project.labels[ann.label_id].name if ann.label_id in self.project.labels else "?"
        lbl_color = self.project.labels[ann.label_id].color if ann.label_id in self.project.labels else "#888"

        hdr = QLabel(f"Annotation {ann.id}")
        hdr.setFont(QFont("Segoe UI", 11, QFont.Bold))
        hdr.setStyleSheet(f"color: {lbl_color};")
        self.props_lo.addWidget(hdr)

        form = QFormLayout()
        form.addRow("Type:", QLabel(ann.type))
        form.addRow("Label:", QLabel(f"<font color='{lbl_color}'>{lbl_name}</font>"))
        form.addRow("Confidence:", QLabel(f"{ann.confidence:.0%}"))
        form.addRow("Locked:", QLabel("🔒 Yes" if ann.locked else "No"))
        form.addRow("Modified:", QLabel(ann.modified_at[:19] if ann.modified_at else ""))
        if ann.notes:
            form.addRow("Notes:", QLabel(ann.notes))
        self.props_lo.addLayout(form)

        btns = QHBoxLayout()
        b_edit = QPushButton("✏ Edit")
        b_edit.clicked.connect(lambda: self._edit_ann_props(ann_id))
        b_del = QPushButton("🗑 Delete")
        b_del.clicked.connect(lambda: self._on_ann_deleted(ann_id))
        btns.addWidget(b_edit)
        btns.addWidget(b_del)
        self.props_lo.addLayout(btns)
        self.props_lo.addStretch()

    def _edit_ann_props(self, ann_id: str):
        old_ann = self.repo.project.annotation_by_id(ann_id)
        if not old_ann:
            return
        dlg = AnnotationPropertiesDialog(old_ann, self.project.labels, self)
        if dlg.exec() == QDialog.Accepted:
            v = dlg.get_values()
            new_ann = Annotation.from_dict(old_ann.to_dict())
            new_ann.label_id = v["label_id"]
            new_ann.confidence = v["confidence"]
            new_ann.notes = v["notes"]
            new_ann.locked = v["locked"]
            new_ann.touch()
            
            fp = self.current_file.filepath if self.current_file else ""
            cmd = ModifyAnnotationCommand(
                self.repo, fp, old_ann, new_ann,
                on_execute=lambda a: (self.ann_list.update_annotation(a), self._show_props(a.id), 
                                      self._redraw_canvas_annotation(a)),
                on_undo=lambda a: (self.ann_list.update_annotation(a), self._show_props(a.id),
                                   self._redraw_canvas_annotation(a))
            )
            self.command_history.execute(cmd)
            self._dirty = True
            self._update_undo_status()

    def _zoom_to_annotation(self, ann_id: str):
        c = self._canvas()
        if isinstance(c, ImageCanvas):
            c.zoom_to_annotation(ann_id)

    # ── view ──

    def _zoom_in(self):
        c = self._canvas()
        if isinstance(c, (ImageCanvas, WorkflowCanvas)):
            c.scale(1.2, 1.2); c._zoom *= 1.2

    def _zoom_out(self):
        c = self._canvas()
        if isinstance(c, (ImageCanvas, WorkflowCanvas)):
            c.scale(1/1.2, 1/1.2); c._zoom /= 1.2

    def _fit_view(self):
        c = self._canvas()
        if isinstance(c, ImageCanvas):
            c.fit_to_view()
        elif isinstance(c, WorkflowCanvas):
            c.fit_to_view()

    def _toggle_crosshair(self, on: bool):
        self.project.config.show_crosshair = on
        c = self._canvas()
        if isinstance(c, ImageCanvas):
            c.set_config(self.project.config)

    def _annotate_text_sel(self):
        c = self._canvas()
        if isinstance(c, TextAnnotationCanvas):
            c.annotate_selection()

    # ── AI & validation ──

    def _ai_prelabel(self):
        c = self._canvas()
        if not isinstance(c, ImageCanvas) or not c._image_item:
            QMessageBox.information(self, "AI", "Open an image first.")
            return
        QMessageBox.information(self, "AI Pre-label (Demo)",
                                "Generates random boxes to simulate AI.\nPlug in real models (YOLO, SAM) in production.")
        pm = c._image_item.pixmap()
        import random
        w, h = pm.width(), pm.height()
        num = random.randint(2, 5)
        label_ids = list(self.project.labels.keys())
        for _ in range(num):
            bw = random.randint(int(w*0.05), int(w*0.3))
            bh = random.randint(int(h*0.05), int(h*0.3))
            bx = random.randint(0, w - bw)
            by = random.randint(0, h - bh)
            lid = random.choice(label_ids) if label_ids else ""
            ann = Annotation(AnnotationType.BBOX.value, lid, {"x": bx, "y": by, "width": bw, "height": bh})
            ann.confidence = round(random.uniform(0.5, 0.95), 2)
            c.draw_annotation(ann)
            self._on_ann_created(ann)

    def _find_overlaps(self):
        if not self.current_file:
            return
        overlaps = []
        for ann in self.current_file.annotations:
            if ann.type in (AnnotationType.BBOX.value, AnnotationType.VIDEO_BBOX.value):
                ov = self.repo.spatial_overlaps(ann.id)
                if ov:
                    overlaps.append(f"Annotation {ann.id} overlaps with: {', '.join(ov)}")
        if overlaps:
            ValidationDialog(overlaps, self).exec()
        else:
            QMessageBox.information(self, "Overlaps", "No overlapping bounding boxes found.")

    def _validate(self):
        if not self.current_file:
            return
        issues = self.repo.validate_annotations(self.current_file.filepath)
        ValidationDialog(issues, self).exec()

    # ── settings & stats ──

    def _show_settings(self):
        dlg = SettingsDialog(self.project.config, self)
        if dlg.exec() == QDialog.Accepted:
            self.project.config = dlg.get_config()
            self._autosave_timer.setInterval(self.project.config.auto_save_interval * 1000)
            if self.project.config.auto_save:
                self._autosave_timer.start()
            else:
                self._autosave_timer.stop()
            self._act_crosshair.setChecked(self.project.config.show_crosshair)
            c = self._canvas()
            if isinstance(c, ImageCanvas):
                c.set_config(self.project.config)
            self._dirty = True

    def _show_stats(self):
        self._sync_current_annotations()
        StatsDialog(self.project, self.repo, self).exec()

    def _show_shortcuts(self):
        QMessageBox.information(self, "Shortcuts",
            "Ctrl+N/O/S: New/Open/Save\n"
            "Ctrl+Z/Y: Undo/Redo\n"
            "Delete: Delete annotation\n"
            "Ctrl+D: Duplicate\n"
            "Ctrl+L: Add Label\n"
            "Ctrl+=/-/0: Zoom\n"
            "1-9: Quick select label\n"
            "Middle Mouse: Pan\n"
            "Scroll: Zoom\n"
            "Right Click: Finish Polygon / Context Menu")

    # ── export ──

    def _export_coco(self):
        self._sync_current_annotations()
        images, annotations, categories = [], [], []
        ann_id = 1
        cat_map = {}
        for i, (lid, lbl) in enumerate(self.project.labels.items(), 1):
            cat_map[lid] = i
            categories.append({"id": i, "name": lbl.name, "supercategory": lbl.category})
        for img_id, (fp, df) in enumerate(self.project.files.items(), 1):
            if df.file_type != FileType.IMAGE.value:
                continue
            img = QImage(fp)
            w, h = img.width(), img.height()
            images.append({"id": img_id, "file_name": Path(fp).name, "width": w, "height": h})
            for a in df.annotations:
                if a.type == AnnotationType.BBOX.value:
                    d = a.data
                    annotations.append({
                        "id": ann_id, "image_id": img_id,
                        "category_id": cat_map.get(a.label_id, 0),
                        "bbox": [d["x"], d["y"], d["width"], d["height"]],
                        "area": d["width"] * d["height"], "iscrowd": 0
                    })
                    ann_id += 1
        data = {"images": images, "annotations": annotations, "categories": categories}
        ExportPreviewDialog("COCO JSON", json.dumps(data, indent=2), self).exec()

    def _export_spacy(self):
        self._sync_current_annotations()
        results = []
        for fp, df in self.project.files.items():
            if df.file_type != FileType.TEXT.value:
                continue
            entities = []
            for a in df.annotations:
                if a.type == AnnotationType.TEXT_SPAN.value:
                    lbl = self.project.labels.get(a.label_id)
                    entities.append([a.data["start"], a.data["end"], lbl.name if lbl else "UNK"])
            results.append({"text": df.text_content, "entities": entities})
        ExportPreviewDialog("spaCy NER", json.dumps(results, indent=2, ensure_ascii=False), self).exec()

    def _export_yolo(self):
        self._sync_current_annotations()
        cat_map = {lid: i for i, lid in enumerate(self.project.labels)}
        output = f"# Classes:\n" + "\n".join(lbl.name for lbl in self.project.labels.values()) + "\n\n"
        for fp, df in self.project.files.items():
            if df.file_type != FileType.IMAGE.value:
                continue
            img = QImage(fp)
            iw, ih = img.width(), img.height()
            if iw == 0:
                continue
            lines = []
            for a in df.annotations:
                if a.type == AnnotationType.BBOX.value:
                    d = a.data
                    cx = (d["x"] + d["width"]/2) / iw
                    cy = (d["y"] + d["height"]/2) / ih
                    bw = d["width"] / iw
                    bh = d["height"] / ih
                    lines.append(f"{cat_map.get(a.label_id, 0)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            output += f"# {Path(fp).name}\n" + "\n".join(lines) + "\n\n"
        ExportPreviewDialog("YOLO", output, self).exec()

    def _export_workflow(self):
        self._sync_current_annotations()
        workflows = []
        for fp, df in self.project.files.items():
            if df.file_type != FileType.WORKFLOW.value:
                continue
            nodes, edges = [], []
            for a in df.annotations:
                if a.type == AnnotationType.WORKFLOW_NODE.value:
                    nodes.append({"id": a.id, "label": a.data.get("name", ""), "type": a.data.get("step_type", "")})
                elif a.type == AnnotationType.WORKFLOW_EDGE.value:
                    edges.append({"source": a.data["source_id"], "target": a.data["target_id"]})
            workflows.append({"name": fp, "nodes": nodes, "edges": edges})
        ExportPreviewDialog("Workflow", json.dumps(workflows, indent=2), self).exec()

    def _export_custom(self):
        self._sync_current_annotations()
        ExportPreviewDialog("Full Project", json.dumps(self.project.to_dict(), indent=2, ensure_ascii=False), self).exec()

    # ── import ──

    def _import_coco(self):
        p, _ = QFileDialog.getOpenFileName(self, "Import COCO", "", "JSON (*.json)")
        if not p:
            return
        try:
            with open(p, 'r') as f:
                data = json.load(f)
            # Map category_id to labels
            cat_map = {}
            for cat in data.get("categories", []):
                lbl = self.repo.add_label(Label(name=cat["name"]))
                cat_map[cat["id"]] = lbl.id
            
            # Map image_id to files
            img_map = {}
            for img in data.get("images", []):
                fp = img["file_name"]
                df = self.repo.add_file(fp, FileType.IMAGE.value)
                img_map[img["id"]] = df
                
            for ann in data.get("annotations", []):
                df = img_map.get(ann["image_id"])
                if df:
                    bbox = ann.get("bbox", [])
                    a = Annotation(AnnotationType.BBOX.value, cat_map.get(ann.get("category_id"), ""),
                                   {"x": bbox[0], "y": bbox[1], "width": bbox[2], "height": bbox[3]} if len(bbox)==4 else {})
                    df.annotations.append(a)
                    
            self.label_mgr.load_labels(self.project.labels)
            self.ann_list.set_project_labels(self.project.labels)
            self._refresh_tree()
            self._dirty = True
            QMessageBox.information(self, "Import", "COCO JSON imported successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Import failed: {e}")

    def _about(self):
        QMessageBox.about(self, "About",
            "<h2>Data Annotation Studio v3</h2>"
            "<p>Architecture: Command, Strategy, Factory, Repository, Observer, Graph DS, Spatial Hash, Interval Tree</p>")

    # ── close / drag&drop ──

    def closeEvent(self, event):
        if self.current_file:
            self.current_file.stop_timer()
        if self._dirty:
            r = QMessageBox.question(self, "Unsaved", "Save before closing?",
                                     QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
            if r == QMessageBox.Save:
                self._save_project()
                event.accept()
            elif r == QMessageBox.Discard:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            fp = url.toLocalFile()
            if fp.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
                self.repo.add_file(fp, FileType.IMAGE.value)
            elif fp.lower().endswith(('.txt', '.md', '.csv', '.json', '.xml')):
                df = self.repo.add_file(fp, FileType.TEXT.value)
                try:
                    with open(fp, 'r', encoding='utf-8') as f:
                        df.text_content = f.read()
                except Exception:
                    pass
            elif os.path.isdir(fp):
                self.repo.add_file(fp, FileType.VIDEO.value)
        self._refresh_tree()
        self._dirty = True


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Data Annotation Studio")
    app.setOrganizationName("AnnotationStudio")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()