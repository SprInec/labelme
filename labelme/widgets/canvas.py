import collections
from typing import Optional

import imgviz
from loguru import logger
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets

import osam
import numpy as np
from labelme._automation import polygon_from_mask
import labelme.utils
from labelme.shape import Shape

# TODO(unknown):
# - [maybe] Find optimal epsilon value.


CURSOR_DEFAULT = QtCore.Qt.ArrowCursor  # type: ignore[attr-defined]
CURSOR_POINT = QtCore.Qt.PointingHandCursor  # type: ignore[attr-defined]
CURSOR_DRAW = QtCore.Qt.CrossCursor  # type: ignore[attr-defined]
CURSOR_MOVE = QtCore.Qt.ClosedHandCursor  # type: ignore[attr-defined]
CURSOR_GRAB = QtCore.Qt.OpenHandCursor  # type: ignore[attr-defined]

MOVE_SPEED = 5.0


class Canvas(QtWidgets.QWidget):
    zoomRequest = QtCore.pyqtSignal(int, QtCore.QPoint)
    scrollRequest = QtCore.pyqtSignal(int, int)
    newShape = QtCore.pyqtSignal()
    selectionChanged = QtCore.pyqtSignal(list)
    shapeMoved = QtCore.pyqtSignal()
    drawingPolygon = QtCore.pyqtSignal(bool)
    vertexSelected = QtCore.pyqtSignal(bool)
    mouseMoved = QtCore.pyqtSignal(QtCore.QPointF)
    modeChanged = QtCore.pyqtSignal(str)  # 添加模式改变的信号，用于通知状态栏

    CREATE, EDIT = 0, 1

    # polygon, rectangle, line, or point
    _createMode = "polygon"

    _fill_drawing = False

    def __init__(self, *args, **kwargs):
        self.epsilon = kwargs.pop("epsilon", 10.0)
        self.double_click = kwargs.pop("double_click", "close")
        if self.double_click not in [None, "close"]:
            raise ValueError(
                "Unexpected value for double_click event: {}".format(
                    self.double_click)
            )
        self.num_backups = kwargs.pop("num_backups", 10)
        self.crosshair = kwargs.pop(
            "crosshair",
            {
                "polygon": False,
                "rectangle": True,
                "circle": False,
                "line": False,
                "point": False,
                "linestrip": False,
                "ai_polygon": False,
                "ai_mask": False,
            },
        )
        super(Canvas, self).__init__(*args, **kwargs)
        # Initialise local state.
        self.mode = self.EDIT
        self.shapes = []
        self.shapesBackups = []
        self.current = None
        self.selectedShapes = []  # save the selected shapes here
        self.selectedShapesCopy = []
        # self.line represents:
        #   - createMode == 'polygon': edge from last point to current
        #   - createMode == 'rectangle': diagonal line of the rectangle
        #   - createMode == 'line': the line
        #   - createMode == 'point': the point
        self.line = Shape(line_color=QtGui.QColor(0, 0, 255))
        self.prevPoint = QtCore.QPointF()
        self.prevMovePoint = QtCore.QPointF()
        self.offsets = QtCore.QPointF(), QtCore.QPointF()
        self.scale = 1.0
        self.pixmap = QtGui.QPixmap()
        self.visible = {}
        self._hideBackround = False
        self.hideBackround = False
        self.hShape = None
        self.prevhShape = None
        self.hVertex = None
        self.prevhVertex = None
        self.hEdge = None
        self.prevhEdge = None
        self.movingShape = False
        self.snapping = True
        self.hShapeIsSelected = False
        self._painter = QtGui.QPainter()
        self._cursor = CURSOR_DEFAULT
        # Menus:
        # 0: right-click without selection and dragging of shapes
        # 1: right-click with selection and dragging of shapes
        self.menus = (QtWidgets.QMenu(), QtWidgets.QMenu())
        # Set widget options.
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.WheelFocus)
        self.image_embedding = None
        self.ai_model = None

        # 添加框选相关变量
        self.selectionBox = None  # 用于存储框选的矩形区域
        self.isSelecting = False  # 是否正在框选
        self.selectionStartPoint = None  # 框选的起始点

        self._sam: Optional[osam.types.Model] = None
        self._sam_embedding: collections.OrderedDict[
            bytes, osam.types.ImageEmbedding
        ] = collections.OrderedDict()

        # 初始化时发送一次模式信号
        QtCore.QTimer.singleShot(0, lambda: self.modeChanged.emit("编辑模式"))

    def fillDrawing(self):
        return self._fill_drawing

    def setFillDrawing(self, value):
        self._fill_drawing = value

    @property
    def createMode(self):
        return self._createMode

    @createMode.setter
    def createMode(self, value):
        if value not in [
            "polygon",
            "rectangle",
            "circle",
            "line",
            "point",
            "linestrip",
            "ai_polygon",
            "ai_mask",
        ]:
            raise ValueError("Unsupported createMode: %s" % value)

        # 如果当前已处于AI模式，再次点击同一AI模式按钮时，应该退出该模式
        if self._createMode in ["ai_polygon", "ai_mask"] and value == self._createMode:
            # 退出AI模式，回到编辑模式
            self.mode = self.EDIT
            # 重置为普通多边形模式
            self._createMode = "polygon"
            # 发送模式改变信号
            self.modeChanged.emit("编辑模式")
            return

        # 如果从AI模式切换到其他模式，或者从其他模式切换到AI模式，需要重置状态
        old_mode = self._createMode
        self._createMode = value

        # 如果当前正在绘制，且模式发生了变化，需要重置当前绘制状态
        if self.drawing() and old_mode != value:
            # 重置当前绘制的形状
            self.current = None
            self.line.points = []
            self.line.point_labels = []
            # 如果之前有发送绘制中的信号，需要取消
            self.drawingPolygon.emit(False)
            self.update()

        # 发送模式改变信号
        if self.mode == self.CREATE:
            mode_name = {
                "polygon": "多边形",
                "rectangle": "矩形",
                "circle": "圆形",
                "line": "线段",
                "point": "点",
                "linestrip": "折线",
                "ai_polygon": "AI多边形",
                "ai_mask": "AI蒙版",
            }.get(value, value)
            self.modeChanged.emit(f"创建{mode_name}模式")
        else:
            self.modeChanged.emit("编辑模式")

    def _compute_and_cache_image_embedding(self) -> None:
        if self.pixmap is None:
            logger.warning("Pixmap is not set yet")
            return

        if self._sam is None:
            logger.warning("AI model is not initialized yet")
            return

        sam: osam.types.Model = self._sam

        try:
            image: np.ndarray = labelme.utils.img_qt_to_arr(
                self.pixmap.toImage())
            if image is None or image.size == 0:
                logger.warning("Invalid image")
                return

            if image.tobytes() in self._sam_embedding:
                return

            logger.debug("Computing image embeddings for model {!r}", sam.name)
            self._sam_embedding[image.tobytes()] = sam.encode_image(
                image=imgviz.asrgb(image)
            )
        except Exception as e:
            logger.error(f"计算图像嵌入失败: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def initializeAiModel(self, model_name):
        if self.pixmap is None:
            logger.warning("Pixmap is not set yet")
            return

        try:
            # 如果model_name是UI显示名称而不是标识符，需要转换
            # 从错误日志中看到传入的名称是"Sam2 (balanced)"等UI显示名称
            # 而非"sam2:latest"等标识符
            api_model_name = model_name

            # 检查是否是UI显示名称，如果是，则转换为对应的模型标识符
            ui_name_to_id = {
                "SegmentAnything (accuracy)": "sam:latest",
                "SegmentAnything (balanced)": "sam:300m",
                "SegmentAnything (speed)": "sam:100m",
                "EfficientSam (accuracy)": "efficientsam:latest",
                "EfficientSam (speed)": "efficientsam:10m",
                "Sam2 (accuracy)": "sam2:large",
                "Sam2 (balanced)": "sam2:latest",  # 对应Sam2BasePlus
                "Sam2 (speed)": "sam2:small",
                "Sam2 (tiny)": "sam2:tiny"
            }

            if model_name in ui_name_to_id:
                api_model_name = ui_name_to_id[model_name]
                logger.info(
                    f"将UI显示名称 '{model_name}' 转换为模型标识符 '{api_model_name}'")

            if self._sam is None or self._sam.name != api_model_name:
                logger.debug("Initializing AI model {!r}", api_model_name)
                self._sam = osam.apis.get_model_type_by_name(api_model_name)()
                self._sam_embedding.clear()

            self._compute_and_cache_image_embedding()
        except Exception as e:
            logger.error(f"初始化AI模型失败: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def storeShapes(self):
        shapesBackup = []
        for shape in self.shapes:
            shapesBackup.append(shape.copy())
        if len(self.shapesBackups) > self.num_backups:
            self.shapesBackups = self.shapesBackups[-self.num_backups - 1:]
        self.shapesBackups.append(shapesBackup)

    @property
    def isShapeRestorable(self):
        # We save the state AFTER each edit (not before) so for an
        # edit to be undoable, we expect the CURRENT and the PREVIOUS state
        # to be in the undo stack.
        if len(self.shapesBackups) < 2:
            return False
        return True

    def restoreShape(self):
        # This does _part_ of the job of restoring shapes.
        # The complete process is also done in app.py::undoShapeEdit
        # and app.py::loadShapes and our own Canvas::loadShapes function.
        if not self.isShapeRestorable:
            return
        self.shapesBackups.pop()  # latest

        # The application will eventually call Canvas.loadShapes which will
        # push this right back onto the stack.
        shapesBackup = self.shapesBackups.pop()
        self.shapes = shapesBackup
        self.selectedShapes = []
        for shape in self.shapes:
            shape.selected = False
        self.update()

    def enterEvent(self, ev):
        self.overrideCursor(self._cursor)

    def leaveEvent(self, ev):
        self.unHighlight()
        self.restoreCursor()

    def focusOutEvent(self, ev):
        self.restoreCursor()

    def isVisible(self, shape):
        """检查形状是否可见"""
        # 先检查Canvas的可见性设置
        canvas_visible = self.visible.get(shape, True)
        # 再检查形状自己的可见性设置
        shape_visible = shape.isVisible() if hasattr(shape, 'isVisible') else True
        # 两者都为True时形状才可见
        return canvas_visible and shape_visible

    def drawing(self):
        return self.mode == self.CREATE

    def editing(self):
        return self.mode == self.EDIT

    def setEditing(self, value=True):
        self.mode = self.EDIT if value else self.CREATE
        if self.mode == self.EDIT:
            # CREATE -> EDIT
            self.repaint()  # clear crosshair
            # 重置AI工具的状态
            if self.createMode in ["ai_polygon", "ai_mask"]:
                # 如果是从AI工具转到编辑模式，重置createMode为polygon
                self._createMode = "polygon"
            # 发送模式改变信号
            self.modeChanged.emit("编辑模式")
        else:
            # EDIT -> CREATE
            self.unHighlight()
            self.deSelectShape()
            # 发送模式改变信号
            mode_name = {
                "polygon": "多边形",
                "rectangle": "矩形",
                "circle": "圆形",
                "line": "线段",
                "point": "点",
                "linestrip": "折线",
                "ai_polygon": "AI多边形",
                "ai_mask": "AI蒙版",
            }.get(self.createMode, self.createMode)
            self.modeChanged.emit(f"创建{mode_name}模式")

    def unHighlight(self):
        if self.hShape:
            # 清除悬停状态
            if hasattr(self.hShape, 'setHoverState'):
                self.hShape.setHoverState(False)
            self.hShape.highlightClear()
            self.update()
        self.prevhShape = self.hShape
        self.prevhVertex = self.hVertex
        self.prevhEdge = self.hEdge
        self.hShape = self.hVertex = self.hEdge = None

    def selectedVertex(self):
        return self.hVertex is not None

    def selectedEdge(self):
        return self.hEdge is not None

    def mouseMoveEvent(self, ev):
        """Update line with last point and current coordinates."""
        try:
            pos = self.transformPos(ev.localPos())
        except AttributeError:
            return

        self.mouseMoved.emit(pos)

        self.prevMovePoint = pos
        self.restoreCursor()

        # 处理框选
        if self.isSelecting:
            self.selectionBox = QtCore.QRectF(
                self.selectionStartPoint, pos
            ).normalized()
            self.update()
            return

        # 更新现有的悬停状态
        need_update = False
        for shape in self.shapes:
            if hasattr(shape, 'hovered') and shape.hovered:
                shape.setHoverState(False)
                need_update = True

        # Polygon drawing.
        if self.drawing():
            if self.createMode in ["ai_polygon", "ai_mask"]:
                self.line.shape_type = "points"
            else:
                self.line.shape_type = self.createMode

            self.overrideCursor(CURSOR_DRAW)
            if not self.current:
                self.repaint()  # draw crosshair
                return

            if self.outOfPixmap(pos):
                # Don't allow the user to draw outside the pixmap.
                # Project the point to the pixmap's edges.
                pos = self.intersectionPoint(self.current[-1], pos)
            elif (
                self.snapping
                and len(self.current) > 1
                and self.createMode == "polygon"
                and self.closeEnough(pos, self.current[0])
            ):
                # Attract line to starting point and
                # colorise to alert the user.
                pos = self.current[0]
                self.overrideCursor(CURSOR_POINT)
                self.current.highlightVertex(0, Shape.NEAR_VERTEX)
            if self.createMode in ["polygon", "linestrip"]:
                self.line.points = [self.current[-1], pos]
                self.line.point_labels = [1, 1]
            elif self.createMode in ["ai_polygon", "ai_mask"]:
                self.line.points = [self.current.points[-1], pos]
                self.line.point_labels = [
                    self.current.point_labels[-1],
                    0 if ev.modifiers() & QtCore.Qt.ShiftModifier else 1,
                ]
            elif self.createMode == "rectangle":
                self.line.points = [self.current[0], pos]
                self.line.point_labels = [1, 1]
                self.line.close()
            elif self.createMode == "circle":
                self.line.points = [self.current[0], pos]
                self.line.point_labels = [1, 1]
                self.line.shape_type = "circle"
            elif self.createMode == "line":
                self.line.points = [self.current[0], pos]
                self.line.point_labels = [1, 1]
                self.line.close()
            elif self.createMode == "point":
                self.line.points = [self.current[0]]
                self.line.point_labels = [1]
                self.line.close()
            assert len(self.line.points) == len(self.line.point_labels)
            self.repaint()
            self.current.highlightClear()
            return

        # Polygon copy moving.
        if QtCore.Qt.RightButton & ev.buttons():
            if self.selectedShapesCopy and self.prevPoint:
                self.overrideCursor(CURSOR_MOVE)
                self.boundedMoveShapes(self.selectedShapesCopy, pos)
                self.repaint()
            elif self.selectedShapes:
                self.selectedShapesCopy = [s.copy()
                                           for s in self.selectedShapes]
                self.repaint()
            return

        # Polygon/Vertex moving.
        if QtCore.Qt.LeftButton & ev.buttons():
            if self.selectedVertex():
                self.boundedMoveVertex(pos)
                self.repaint()
                self.movingShape = True
            elif self.selectedShapes and self.prevPoint:
                self.overrideCursor(CURSOR_MOVE)
                self.boundedMoveShapes(self.selectedShapes, pos)
                self.repaint()
                self.movingShape = True
            return

        # Just hovering over the canvas, 2 possibilities:
        # - Highlight shapes
        # - Highlight vertex
        # Update shape/vertex fill and tooltip value accordingly.
        self.setToolTip(self.tr("Image"))

        # 先重置所有形状的悬停状态
        for shape in self.shapes:
            if hasattr(shape, 'hovered') and shape.hovered:
                shape.setHoverState(False)
                need_update = True

        for shape in reversed([s for s in self.shapes if self.isVisible(s)]):
            # 处理所有类型标签的悬停效果和提示
            if shape.containsPoint(pos):
                # 设置悬停状态
                if hasattr(shape, 'setHoverState'):
                    shape.setHoverState(True)
                    # 显示当前标签名称
                    self.setToolTip(shape.label)
                    self.setStatusTip(self.toolTip())
                    if shape.shape_type == "point":
                        self.overrideCursor(CURSOR_POINT)
                    else:
                        self.overrideCursor(CURSOR_GRAB)
                    self.update()

                    # 如果不需要处理顶点高亮，则直接处理下一个
                    if shape.shape_type == "point":
                        break

            # Look for a nearby vertex to highlight. If that fails,
            # check if we happen to be inside a shape.
            index = shape.nearestVertex(pos, self.epsilon)
            index_edge = shape.nearestEdge(pos, self.epsilon)
            if index is not None:
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.prevhVertex = self.hVertex = index
                self.prevhShape = self.hShape = shape
                self.prevhEdge = self.hEdge
                self.hEdge = None
                shape.highlightVertex(index, shape.MOVE_VERTEX)
                self.overrideCursor(CURSOR_POINT)
                self.setToolTip(
                    self.tr(
                        "Click & Drag to move point\n"
                        "ALT + SHIFT + Click to delete point"
                    )
                )
                self.setStatusTip(self.toolTip())
                self.update()
                break
            elif index_edge is not None and shape.canAddPoint():
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.prevhVertex = self.hVertex
                self.hVertex = None
                self.prevhShape = self.hShape = shape
                self.prevhEdge = self.hEdge = index_edge
                self.overrideCursor(CURSOR_POINT)
                self.setToolTip(self.tr("ALT + Click to create point"))
                self.setStatusTip(self.toolTip())
                self.update()
                break
            elif shape.containsPoint(pos):
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.prevhVertex = self.hVertex
                self.hVertex = None
                self.prevhShape = self.hShape = shape
                self.prevhEdge = self.hEdge
                self.hEdge = None
                self.setToolTip(
                    self.tr("Click & drag to move shape '%s'") % shape.label
                )
                self.setStatusTip(self.toolTip())
                self.overrideCursor(CURSOR_GRAB)
                self.update()
                break
        else:  # Nothing found, clear highlights, reset state.
            self.unHighlight()
        self.vertexSelected.emit(self.hVertex is not None)

    def addPointToEdge(self):
        shape = self.prevhShape
        index = self.prevhEdge
        point = self.prevMovePoint
        if shape is None or index is None or point is None:
            return
        shape.insertPoint(index, point)
        shape.highlightVertex(index, shape.MOVE_VERTEX)
        self.hShape = shape
        self.hVertex = index
        self.hEdge = None
        self.movingShape = True

    def removeSelectedPoint(self):
        shape = self.prevhShape
        index = self.prevhVertex
        if shape is None or index is None:
            return
        shape.removePoint(index)
        shape.highlightClear()
        self.hShape = shape
        self.prevhVertex = None
        self.movingShape = True  # Save changes

    def mousePressEvent(self, ev):
        pos = self.transformPos(ev.localPos())

        is_shift_pressed = ev.modifiers() & QtCore.Qt.ShiftModifier

        if ev.button() == QtCore.Qt.LeftButton:
            if self.drawing():
                if self.current:
                    # Add point to existing shape.
                    if self.createMode == "polygon":
                        self.current.addPoint(self.line[1])
                        self.line[0] = self.current[-1]
                        if self.current.isClosed():
                            self.finalise()
                    elif self.createMode in ["rectangle", "circle", "line"]:
                        assert len(self.current.points) == 1
                        self.current.points = self.line.points
                        self.finalise()
                    elif self.createMode == "linestrip":
                        self.current.addPoint(self.line[1])
                        self.line[0] = self.current[-1]
                        if int(ev.modifiers()) == QtCore.Qt.ControlModifier:
                            self.finalise()
                    elif self.createMode in ["ai_polygon", "ai_mask"]:
                        self.current.addPoint(
                            self.line.points[1],
                            label=self.line.point_labels[1],
                        )
                        self.line.points[0] = self.current.points[-1]
                        self.line.point_labels[0] = self.current.point_labels[-1]
                        if ev.modifiers() & QtCore.Qt.ControlModifier:
                            self.finalise()
                elif not self.outOfPixmap(pos):
                    # Create new shape.
                    self.current = Shape(
                        shape_type="points"
                        if self.createMode in ["ai_polygon", "ai_mask"]
                        else self.createMode
                    )
                    self.current.addPoint(
                        pos, label=0 if is_shift_pressed else 1)
                    if self.createMode == "point":
                        self.finalise()
                    elif (
                        self.createMode in ["ai_polygon", "ai_mask"]
                        and ev.modifiers() & QtCore.Qt.ControlModifier
                    ):
                        self.finalise()
                    else:
                        if self.createMode == "circle":
                            self.current.shape_type = "circle"
                        self.line.points = [pos, pos]
                        if (
                            self.createMode in ["ai_polygon", "ai_mask"]
                            and is_shift_pressed
                        ):
                            self.line.point_labels = [0, 0]
                        else:
                            self.line.point_labels = [1, 1]
                        self.setHiding()
                        self.drawingPolygon.emit(True)
                        self.update()
            elif self.editing():
                # 检查是否点击了顶点或边缘
                is_vertex_selected = self.selectedVertex()
                is_edge_selected = self.selectedEdge()

                # 处理特殊修改键的情况
                if is_edge_selected and ev.modifiers() == QtCore.Qt.AltModifier:
                    self.addPointToEdge()
                elif is_vertex_selected and ev.modifiers() == (
                    QtCore.Qt.AltModifier | QtCore.Qt.ShiftModifier
                ):
                    self.removeSelectedPoint()
                # 如果没有选中顶点或边缘，并且不是在形状上点击，则开始框选
                elif not is_vertex_selected and not is_edge_selected:
                    # 检查点击位置是否在任何形状内
                    shape_clicked = False
                    for shape in reversed(self.shapes):
                        if self.isVisible(shape) and shape.containsPoint(pos):
                            shape_clicked = True
                            break

                    # 如果没有点击到任何形状，则开始框选
                    if not shape_clicked:
                        # 如果没有按住Ctrl键，则清除之前的选择
                        if not (ev.modifiers() & QtCore.Qt.ControlModifier):
                            self.deSelectShape()
                        # 开始框选
                        self.isSelecting = True
                        self.selectionStartPoint = pos
                        self.selectionBox = QtCore.QRectF(pos, pos)
                        self.update()
                        return  # 开始框选后直接返回，不执行后续代码

                # 如果点击了形状，则选择该形状
                group_mode = int(ev.modifiers()) == QtCore.Qt.ControlModifier
                self.selectShapePoint(pos, multiple_selection_mode=group_mode)
                self.prevPoint = pos
                self.repaint()
        elif ev.button() == QtCore.Qt.RightButton and self.editing():
            group_mode = int(ev.modifiers()) == QtCore.Qt.ControlModifier
            if not self.selectedShapes or (
                self.hShape is not None and self.hShape not in self.selectedShapes
            ):
                self.selectShapePoint(pos, multiple_selection_mode=group_mode)
                self.repaint()
            self.prevPoint = pos

    def mouseReleaseEvent(self, ev):
        if ev.button() == QtCore.Qt.RightButton:
            menu = self.menus[len(self.selectedShapesCopy) > 0]
            self.restoreCursor()
            if not menu.exec_(self.mapToGlobal(ev.pos())) and self.selectedShapesCopy:
                # Cancel the move by deleting the shadow copy.
                self.selectedShapesCopy = []
                self.repaint()
        elif ev.button() == QtCore.Qt.LeftButton:
            # 处理框选结束
            if self.isSelecting:
                self.isSelecting = False
                if self.selectionBox and self.selectionBox.width() > 5 and self.selectionBox.height() > 5:
                    # 选择框选区域内的所有形状
                    selected_shapes = []
                    for shape in self.shapes:
                        if self.isVisible(shape) and self.shapeIsInSelectionBox(shape, self.selectionBox):
                            selected_shapes.append(shape)

                    # 如果按住Ctrl键，则添加到已选择的形状中
                    if ev.modifiers() & QtCore.Qt.ControlModifier:
                        for shape in selected_shapes:
                            if shape not in self.selectedShapes:
                                self.selectedShapes.append(shape)
                    else:
                        self.selectedShapes = selected_shapes

                    self.selectionChanged.emit(self.selectedShapes)

                self.selectionBox = None
                self.update()
                return

            if self.editing():
                if (
                    self.hShape is not None
                    and self.hShapeIsSelected
                    and not self.movingShape
                ):
                    self.selectionChanged.emit(
                        [x for x in self.selectedShapes if x != self.hShape]
                    )

        if self.movingShape and self.hShape:
            index = self.shapes.index(self.hShape)
            if self.shapesBackups[-1][index].points != self.shapes[index].points:
                self.storeShapes()
                self.shapeMoved.emit()

            self.movingShape = False

    def endMove(self, copy):
        assert self.selectedShapes and self.selectedShapesCopy
        assert len(self.selectedShapesCopy) == len(self.selectedShapes)
        if copy:
            for i, shape in enumerate(self.selectedShapesCopy):
                self.shapes.append(shape)
                self.selectedShapes[i].selected = False
                self.selectedShapes[i] = shape
        else:
            for i, shape in enumerate(self.selectedShapesCopy):
                self.selectedShapes[i].points = shape.points
        self.selectedShapesCopy = []
        self.repaint()
        self.storeShapes()
        return True

    def hideBackroundShapes(self, value):
        self.hideBackround = value
        if self.selectedShapes:
            # Only hide other shapes if there is a current selection.
            # Otherwise the user will not be able to select a shape.
            self.setHiding(True)
            self.update()

    def setHiding(self, enable=True):
        self._hideBackround = self.hideBackround if enable else False

    def canCloseShape(self):
        return self.drawing() and (
            (self.current and len(self.current) > 2)
            or self.createMode in ["ai_polygon", "ai_mask"]
        )

    def mouseDoubleClickEvent(self, ev):
        if self.double_click != "close":
            return

        if (
            self.createMode == "polygon" and self.canCloseShape()
        ) or self.createMode in ["ai_polygon", "ai_mask"]:
            self.finalise()

    def selectShapes(self, shapes):
        self.setHiding()
        self.selectionChanged.emit(shapes)
        self.update()

    def selectShapePoint(self, point, multiple_selection_mode):
        """Select the first shape created which contains this point."""
        if self.selectedVertex():  # A vertex is marked for selection.
            index, shape = self.hVertex, self.hShape
            shape.highlightVertex(index, shape.MOVE_VERTEX)
        else:
            for shape in reversed(self.shapes):
                if self.isVisible(shape) and shape.containsPoint(point):
                    self.setHiding()
                    if shape not in self.selectedShapes:
                        if multiple_selection_mode:
                            self.selectionChanged.emit(
                                self.selectedShapes + [shape])
                        else:
                            self.selectionChanged.emit([shape])
                        self.hShapeIsSelected = False
                    else:
                        self.hShapeIsSelected = True
                    self.calculateOffsets(point)
                    return
        self.deSelectShape()

    def calculateOffsets(self, point):
        left = self.pixmap.width() - 1
        right = 0
        top = self.pixmap.height() - 1
        bottom = 0
        for s in self.selectedShapes:
            rect = s.boundingRect()
            if rect.left() < left:
                left = rect.left()
            if rect.right() > right:
                right = rect.right()
            if rect.top() < top:
                top = rect.top()
            if rect.bottom() > bottom:
                bottom = rect.bottom()

        x1 = left - point.x()
        y1 = top - point.y()
        x2 = right - point.x()
        y2 = bottom - point.y()
        self.offsets = QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)

    def boundedMoveVertex(self, pos):
        index, shape = self.hVertex, self.hShape
        point = shape[index]
        if self.outOfPixmap(pos):
            pos = self.intersectionPoint(point, pos)
        shape.moveVertexBy(index, pos - point)

    def boundedMoveShapes(self, shapes, pos):
        if self.outOfPixmap(pos):
            return False  # No need to move
        o1 = pos + self.offsets[0]
        if self.outOfPixmap(o1):
            pos -= QtCore.QPointF(min(0, o1.x()), min(0, o1.y()))
        o2 = pos + self.offsets[1]
        if self.outOfPixmap(o2):
            pos += QtCore.QPointF(
                min(0, self.pixmap.width() - o2.x()),
                min(0, self.pixmap.height() - o2.y()),
            )
        # XXX: The next line tracks the new position of the cursor
        # relative to the shape, but also results in making it
        # a bit "shaky" when nearing the border and allows it to
        # go outside of the shape's area for some reason.
        # self.calculateOffsets(self.selectedShapes, pos)
        dp = pos - self.prevPoint
        if dp:
            for shape in shapes:
                shape.moveBy(dp)
            self.prevPoint = pos
            return True
        return False

    def deSelectShape(self):
        if self.selectedShapes:
            self.setHiding(False)
            self.selectionChanged.emit([])
            self.hShapeIsSelected = False
            self.update()

    def deleteSelected(self):
        deleted_shapes = []
        if self.selectedShapes:
            for shape in self.selectedShapes:
                self.shapes.remove(shape)
                deleted_shapes.append(shape)
            self.storeShapes()
            self.selectedShapes = []
            self.update()
        return deleted_shapes

    def deleteShape(self, shape):
        if shape in self.selectedShapes:
            self.selectedShapes.remove(shape)
        if shape in self.shapes:
            self.shapes.remove(shape)
        self.storeShapes()
        self.update()

    def paintEvent(self, event: Optional[QtGui.QPaintEvent]) -> None:
        if not self.pixmap:
            return super(Canvas, self).paintEvent(event)

        p = self._painter
        p.begin(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setRenderHint(QtGui.QPainter.HighQualityAntialiasing)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

        p.scale(self.scale, self.scale)
        p.translate(self.offsetToCenter())

        p.drawPixmap(0, 0, self.pixmap)

        # 绘制选择框
        if self.isSelecting and self.selectionBox:
            p.save()
            p.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0), 1, QtCore.Qt.DashLine))
            p.setBrush(QtGui.QColor(0, 255, 0, 32))
            p.drawRect(self.selectionBox)
            p.restore()

        p.scale(1 / self.scale, 1 / self.scale)

        # draw crosshair
        if (
            self.crosshair[self._createMode]
            and self.drawing()
            and self.prevMovePoint
            and not self.outOfPixmap(self.prevMovePoint)
        ):
            p.setPen(QtGui.QColor(0, 255, 0))
            p.drawLine(
                0,
                int(self.prevMovePoint.y() * self.scale),
                self.width() - 1,
                int(self.prevMovePoint.y() * self.scale),
            )
            p.drawLine(
                int(self.prevMovePoint.x() * self.scale),
                0,
                int(self.prevMovePoint.x() * self.scale),
                self.height() - 1,
            )

        Shape.scale = self.scale
        for shape in self.shapes:
            if (shape.selected or not self._hideBackround) and self.isVisible(shape):
                shape.fill = shape.selected or shape == self.hShape
                shape.paint(p)
        if self.current:
            self.current.paint(p)
            assert len(self.line.points) == len(self.line.point_labels)
            self.line.paint(p)
        if self.selectedShapesCopy:
            for s in self.selectedShapesCopy:
                s.paint(p)

        if not self.current:
            p.end()
            return

        if (
            self.createMode == "polygon"
            and self.fillDrawing()
            and len(self.current.points) >= 2
        ):
            drawing_shape = self.current.copy()
            if drawing_shape.fill_color.getRgb()[3] == 0:
                logger.warning(
                    "fill_drawing=true, but fill_color is transparent,"
                    " so forcing to be opaque."
                )
                drawing_shape.fill_color.setAlpha(64)
            drawing_shape.addPoint(self.line[1])

        if self.createMode not in ["ai_polygon", "ai_mask"]:
            p.end()
            return

        drawing_shape = self.current.copy()
        drawing_shape.addPoint(
            point=self.line.points[1],
            label=self.line.point_labels[1],
        )
        if self.createMode in ["ai_polygon", "ai_mask"]:
            if self._sam is None:
                logger.warning("SAM model is not set yet")
                p.end()
                return
            _update_shape_with_sam(
                shape=drawing_shape,
                createMode=self.createMode,
                model_name=self._sam.name,
                image_embedding=self._sam_embedding[
                    labelme.utils.img_qt_to_arr(
                        self.pixmap.toImage()).tobytes()
                ],
            )
        drawing_shape.fill = self.fillDrawing()
        drawing_shape.selected = True
        drawing_shape.paint(p)
        p.end()

    def transformPos(self, point):
        """Convert from widget-logical coordinates to painter-logical ones."""
        return point / self.scale - self.offsetToCenter()

    def offsetToCenter(self):
        s = self.scale
        area = super(Canvas, self).size()
        w, h = self.pixmap.width() * s, self.pixmap.height() * s
        aw, ah = area.width(), area.height()
        x = (aw - w) / (2 * s) if aw > w else 0
        y = (ah - h) / (2 * s) if ah > h else 0
        return QtCore.QPointF(x, y)

    def outOfPixmap(self, p):
        w, h = self.pixmap.width(), self.pixmap.height()
        return not (0 <= p.x() <= w - 1 and 0 <= p.y() <= h - 1)

    def finalise(self):
        assert self.current
        # 只有在self._sam不为None且当前是AI模式时才调用_update_shape_with_sam
        if (hasattr(self, '_sam') and self._sam is not None and 
            hasattr(self, '_sam_embedding') and self.createMode in ["ai_polygon", "ai_mask"]):
            _update_shape_with_sam(
                shape=self.current,
                createMode=self.createMode,
                model_name=self._sam.name,
                image_embedding=self._sam_embedding[
                    labelme.utils.img_qt_to_arr(
                        self.pixmap.toImage()).tobytes()
                ],
            )
        self.current.close()

        self.shapes.append(self.current)
        self.storeShapes()
        self.current = None
        self.setHiding(False)
        self.drawingPolygon.emit(False)
        self.newShape.emit()  # 确保这行代码存在，发出新形状创建完成的信号

        # 如果是AI工具，完成后重置状态
        if self.createMode in ["ai_polygon", "ai_mask"]:
            self.line.points = []
            self.line.point_labels = []
            # 完成AI工具标注后自动切换回编辑模式
            self.mode = self.EDIT
            old_mode = self.createMode
            self._createMode = "polygon"
            # 发送模式改变信号
            self.modeChanged.emit("编辑模式")

        self.update()

    def closeEnough(self, p1, p2):
        # d = distance(p1 - p2)
        # m = (p1-p2).manhattanLength()
        # print "d %.2f, m %d, %.2f" % (d, m, d - m)
        # divide by scale to allow more precision when zoomed in
        return labelme.utils.distance(p1 - p2) < (self.epsilon / self.scale)

    def intersectionPoint(self, p1, p2):
        # Cycle through each image edge in clockwise fashion,
        # and find the one intersecting the current line segment.
        # http://paulbourke.net/geometry/lineline2d/
        size = self.pixmap.size()
        points = [
            (0, 0),
            (size.width() - 1, 0),
            (size.width() - 1, size.height() - 1),
            (0, size.height() - 1),
        ]
        # x1, y1 should be in the pixmap, x2, y2 should be out of the pixmap
        x1 = min(max(p1.x(), 0), size.width() - 1)
        y1 = min(max(p1.y(), 0), size.height() - 1)
        x2, y2 = p2.x(), p2.y()
        d, i, (x, y) = min(self.intersectingEdges((x1, y1), (x2, y2), points))
        x3, y3 = points[i]
        x4, y4 = points[(i + 1) % 4]
        if (x, y) == (x1, y1):
            # Handle cases where previous point is on one of the edges.
            if x3 == x4:
                return QtCore.QPointF(x3, min(max(0, y2), max(y3, y4)))
            else:  # y3 == y4
                return QtCore.QPointF(min(max(0, x2), max(x3, x4)), y3)
        return QtCore.QPointF(x, y)

    def intersectingEdges(self, point1, point2, points):
        """Find intersecting edges.

        For each edge formed by `points', yield the intersection
        with the line segment `(x1,y1) - (x2,y2)`, if it exists.
        Also return the distance of `(x2,y2)' to the middle of the
        edge along with its index, so that the one closest can be chosen.
        """
        (x1, y1) = point1
        (x2, y2) = point2
        for i in range(4):
            x3, y3 = points[i]
            x4, y4 = points[(i + 1) % 4]
            denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
            nua = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
            nub = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)
            if denom == 0:
                # This covers two cases:
                #   nua == nub == 0: Coincident
                #   otherwise: Parallel
                continue
            ua, ub = nua / denom, nub / denom
            if 0 <= ua <= 1 and 0 <= ub <= 1:
                x = x1 + ua * (x2 - x1)
                y = y1 + ua * (y2 - y1)
                m = QtCore.QPointF((x3 + x4) / 2, (y3 + y4) / 2)
                d = labelme.utils.distance(m - QtCore.QPointF(x2, y2))
                yield d, i, (x, y)

    # These two, along with a call to adjustSize are required for the
    # scroll area.
    def sizeHint(self):
        return self.minimumSizeHint()

    def minimumSizeHint(self):
        if self.pixmap:
            return self.scale * self.pixmap.size()
        return super(Canvas, self).minimumSizeHint()

    def wheelEvent(self, ev):
        mods = ev.modifiers()
        delta = ev.angleDelta()
        if QtCore.Qt.ControlModifier == int(mods):
            # with Ctrl/Command key
            # zoom
            self.zoomRequest.emit(delta.y(), ev.pos())
        else:
            # scroll
            self.scrollRequest.emit(delta.x(), QtCore.Qt.Horizontal)
            self.scrollRequest.emit(delta.y(), QtCore.Qt.Vertical)
        ev.accept()

    def moveByKeyboard(self, offset):
        if self.selectedShapes:
            self.boundedMoveShapes(self.selectedShapes,
                                   self.prevPoint + offset)
            self.repaint()
            self.movingShape = True

    def keyPressEvent(self, ev):
        modifiers = ev.modifiers()
        key = ev.key()
        if self.drawing():
            if key == QtCore.Qt.Key_Escape and self.current:
                self.current = None
                self.drawingPolygon.emit(False)
                self.update()
            elif key == QtCore.Qt.Key_Return and self.canCloseShape():
                self.finalise()
            elif modifiers == QtCore.Qt.AltModifier:
                self.snapping = False
        elif self.editing():
            if key == QtCore.Qt.Key_Delete or key == QtCore.Qt.Key_Backspace:
                # 删除所有选中的形状
                if self.selectedShapes:
                    # 这里不直接调用deleteSelected，而是发出信号让MainWindow处理
                    # 因为MainWindow需要更新标签列表和其他UI元素
                    # 我们可以通过selectionChanged信号触发MainWindow的deleteSelectedShape方法
                    # 先保存当前选中的形状
                    selected = self.selectedShapes.copy()
                    # 发出信号，这会触发MainWindow的shapeSelectionChanged方法
                    self.selectionChanged.emit(selected)
                    # 然后可以通过其他方式触发删除操作
                    # 这里我们直接删除，MainWindow会在下一次调用时处理UI更新
                    self.deleteSelected()
            elif key == QtCore.Qt.Key_Up:
                self.moveByKeyboard(QtCore.QPointF(0.0, -MOVE_SPEED))
            elif key == QtCore.Qt.Key_Down:
                self.moveByKeyboard(QtCore.QPointF(0.0, MOVE_SPEED))
            elif key == QtCore.Qt.Key_Left:
                self.moveByKeyboard(QtCore.QPointF(-MOVE_SPEED, 0.0))
            elif key == QtCore.Qt.Key_Right:
                self.moveByKeyboard(QtCore.QPointF(MOVE_SPEED, 0.0))

    def keyReleaseEvent(self, ev):
        modifiers = ev.modifiers()
        if self.drawing():
            if int(modifiers) == 0:
                self.snapping = True
        elif self.editing():
            if self.movingShape and self.selectedShapes:
                index = self.shapes.index(self.selectedShapes[0])
                if self.shapesBackups[-1][index].points != self.shapes[index].points:
                    self.storeShapes()
                    self.shapeMoved.emit()

                self.movingShape = False

    def setLastLabel(self, text, flags, group_id=None, description=None):
        assert text
        self.shapes[-1].label = text
        self.shapes[-1].flags = flags
        self.shapes[-1].group_id = group_id
        self.shapes[-1].description = description
        self.shapesBackups.pop()
        self.storeShapes()
        return self.shapes[-1]

    def undoLastLine(self):
        assert self.shapes
        self.current = self.shapes.pop()
        self.current.setOpen()
        self.current.restoreShapeRaw()
        if self.createMode in ["polygon", "linestrip"]:
            self.line.points = [self.current[-1], self.current[0]]
        elif self.createMode in ["rectangle", "line", "circle"]:
            self.current.points = self.current.points[0:1]
        elif self.createMode == "point":
            self.current = None
        self.drawingPolygon.emit(True)

    def undoLastPoint(self):
        if not self.current or self.current.isClosed():
            return
        self.current.popPoint()
        if len(self.current) > 0:
            self.line[0] = self.current[-1]
        else:
            self.current = None
            self.drawingPolygon.emit(False)
        self.update()

    def loadPixmap(self, pixmap, clear_shapes=True):
        self.pixmap = pixmap
        if self._sam:
            self._compute_and_cache_image_embedding()
        if clear_shapes:
            self.shapes = []
        self.update()

    def loadShapes(self, shapes, replace=True):
        if replace:
            self.shapes = list(shapes)
        else:
            self.shapes.extend(shapes)
        self.storeShapes()
        self.current = None
        self.hShape = None
        self.hVertex = None
        self.hEdge = None
        self.update()

    def setShapeVisible(self, shape, value):
        self.visible[shape] = value
        self.update()

    def overrideCursor(self, cursor):
        self.restoreCursor()
        self._cursor = cursor
        QtWidgets.QApplication.setOverrideCursor(cursor)

    def restoreCursor(self):
        QtWidgets.QApplication.restoreOverrideCursor()

    def resetState(self):
        self.restoreCursor()
        self.pixmap = None
        self.shapesBackups = []
        self.update()

    def shapeIsInSelectionBox(self, shape, selection_box):
        """检查形状是否在选择框内"""
        # 检查形状的任何点是否在选择框内
        for point in shape.points:
            if selection_box.contains(point):
                return True
        return False


def _update_shape_with_sam(
    shape: Shape,
    createMode: str,
    model_name: str,
    image_embedding: osam.types.ImageEmbedding,
) -> None:
    if createMode not in ["ai_polygon", "ai_mask"]:
        raise ValueError(
            f"createMode must be 'ai_polygon' or 'ai_mask', not {createMode}"
        )

    try:
        logger.info(f"使用模型 {model_name} 进行分割")

        if not shape.points:
            logger.warning("形状没有点")
            return

        points = [[point.x(), point.y()] for point in shape.points]
        point_labels = shape.point_labels

        if len(points) != len(point_labels):
            logger.warning(
                f"点数量 ({len(points)}) 与标签数量 ({len(point_labels)}) 不匹配")
            # 确保长度一致
            point_labels = [1] * len(points)

        logger.info(f"点: {points}")
        logger.info(f"标签: {point_labels}")

        response: osam.types.GenerateResponse = osam.apis.generate(
            osam.types.GenerateRequest(
                model=model_name,
                image_embedding=image_embedding,
                prompt=osam.types.Prompt(
                    points=points,
                    point_labels=point_labels,
                ),
            )
        )

        logger.info(
            f"获取到注释数量: {len(response.annotations) if response.annotations else 0}")

        if not response.annotations:
            logger.warning("模型 {!r} 未返回任何注释", model_name)
            return

        if createMode == "ai_mask":
            y1: int
            x1: int
            y2: int
            x2: int
            if response.annotations[0].bounding_box is None:
                mask = response.annotations[0].mask
                if mask is None or mask.size == 0:
                    logger.warning("返回的掩码为空")
                    return

                logger.info(f"掩码形状: {mask.shape}")
                bbox = imgviz.instances.mask_to_bbox([mask])
                if len(bbox) == 0:
                    logger.warning("无法从掩码计算边界框")
                    return

                y1, x1, y2, x2 = bbox[0].astype(int)
                logger.info(f"从掩码计算的边界框: [{x1}, {y1}, {x2}, {y2}]")
            else:
                y1 = response.annotations[0].bounding_box.ymin
                x1 = response.annotations[0].bounding_box.xmin
                y2 = response.annotations[0].bounding_box.ymax
                x2 = response.annotations[0].bounding_box.xmax
                logger.info(f"模型提供的边界框: [{x1}, {y1}, {x2}, {y2}]")

            mask_roi = response.annotations[0].mask[y1: y2 + 1, x1: x2 + 1]
            if mask_roi is None or mask_roi.size == 0:
                logger.warning("裁剪后的掩码为空")
                return

            logger.info(f"裁剪后的掩码形状: {mask_roi.shape}")

            shape.setShapeRefined(
                shape_type="mask",
                points=[QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)],
                point_labels=[1, 1],
                mask=mask_roi,
            )
        elif createMode == "ai_polygon":
            mask = response.annotations[0].mask
            if mask is None or mask.size == 0:
                logger.warning("返回的掩码为空")
                return

            logger.info(f"多边形掩码形状: {mask.shape}")

            points = polygon_from_mask.compute_polygon_from_mask(mask=mask)

            if len(points) < 2:
                logger.warning("计算的多边形点数过少")
                return

            logger.info(f"计算的多边形点数: {len(points)}")

            shape.setShapeRefined(
                shape_type="polygon",
                points=[QtCore.QPointF(point[0], point[1])
                        for point in points],
                point_labels=[1] * len(points),
            )
    except Exception as e:
        logger.error(f"使用AI模型更新形状时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
