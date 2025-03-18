import copy

import numpy as np
import skimage.measure
from loguru import logger
from PyQt5 import QtCore
from PyQt5 import QtGui

import labelme.utils

# TODO(unknown):
# - [opt] Store paths instead of creating new ones at each paint.


class Shape(object):
    # Render handles as squares
    P_SQUARE = 0

    # Render handles as circles
    P_ROUND = 1

    # Flag for the handles we would move if dragging
    MOVE_VERTEX = 0

    # Flag for all other handles on the current shape
    NEAR_VERTEX = 1

    PEN_WIDTH = 2
    # 添加选中形状的线宽
    SELECT_PEN_WIDTH = 4

    # The following class variables influence the drawing of all shape objects.
    line_color = None
    fill_color = None
    select_line_color = None
    select_fill_color = None
    vertex_fill_color = None
    hvertex_fill_color = None
    point_type = P_ROUND
    point_size = 8
    # 增加选中时的顶点大小，尤其对点标签有效
    select_point_size = 12
    scale = 1.0

    # 显示标签名称的标志
    show_label_names = False

    def __init__(
        self,
        label=None,
        line_color=None,
        shape_type=None,
        flags=None,
        group_id=None,
        description=None,
        mask=None,
    ):
        self.label = label
        self.group_id = group_id
        self.points = []
        self.point_labels = []
        self.shape_type = shape_type
        self._shape_raw = None
        self._points_raw = []
        self._shape_type_raw = None
        self.fill = False
        self.selected = False
        self.visible = True
        self.flags = flags
        self.description = description
        self.other_data = {}
        self.mask = mask

        self._highlightIndex = None
        self._highlightMode = self.NEAR_VERTEX
        self._highlightSettings = {
            self.NEAR_VERTEX: (4, self.P_ROUND),
            self.MOVE_VERTEX: (1.5, self.P_SQUARE),
        }

        self._closed = False
        # 添加悬停标志
        self.hovered = False

        if line_color is not None:
            # Override the class line_color attribute
            # with an object attribute. Currently this
            # is used for drawing the pending line a different color.
            self.line_color = line_color

    def _scale_point(self, point: QtCore.QPointF) -> QtCore.QPointF:
        return QtCore.QPointF(point.x() * self.scale, point.y() * self.scale)

    def setShapeRefined(self, shape_type, points, point_labels, mask=None):
        self._shape_raw = (self.shape_type, self.points, self.point_labels)
        self.shape_type = shape_type
        self.points = points
        self.point_labels = point_labels
        self.mask = mask

    def restoreShapeRaw(self):
        if self._shape_raw is None:
            return
        self.shape_type, self.points, self.point_labels = self._shape_raw
        self._shape_raw = None

    @property
    def shape_type(self):
        return self._shape_type

    @shape_type.setter
    def shape_type(self, value):
        if value is None:
            value = "polygon"
        if value not in [
            "polygon",
            "rectangle",
            "point",
            "line",
            "circle",
            "linestrip",
            "points",
            "mask",
        ]:
            raise ValueError("Unexpected shape_type: {}".format(value))
        self._shape_type = value

    def close(self):
        self._closed = True

    def addPoint(self, point, label=1):
        if self.points and point == self.points[0]:
            self.close()
        else:
            self.points.append(point)
            self.point_labels.append(label)

    def canAddPoint(self):
        return self.shape_type in ["polygon", "linestrip"]

    def popPoint(self):
        if self.points:
            if self.point_labels:
                self.point_labels.pop()
            return self.points.pop()
        return None

    def insertPoint(self, i, point, label=1):
        self.points.insert(i, point)
        self.point_labels.insert(i, label)

    def removePoint(self, i):
        if not self.canAddPoint():
            logger.warning(
                "Cannot remove point from: shape_type=%r",
                self.shape_type,
            )
            return

        if self.shape_type == "polygon" and len(self.points) <= 3:
            logger.warning(
                "Cannot remove point from: shape_type=%r, len(points)=%d",
                self.shape_type,
                len(self.points),
            )
            return

        if self.shape_type == "linestrip" and len(self.points) <= 2:
            logger.warning(
                "Cannot remove point from: shape_type=%r, len(points)=%d",
                self.shape_type,
                len(self.points),
            )
            return

        self.points.pop(i)
        self.point_labels.pop(i)

    def isClosed(self):
        return self._closed

    def setOpen(self):
        self._closed = False

    def setVisible(self, visible):
        """设置形状是否可见"""
        self.visible = visible

    def isVisible(self):
        """返回形状是否可见"""
        return self.visible

    def paint(self, painter):
        if self.mask is None and not self.points:
            return

        if not self.visible:
            return

        color = self.select_line_color if self.selected else self.line_color
        pen = QtGui.QPen(color)
        # 根据选中状态设置不同的线宽
        pen.setWidth(
            self.SELECT_PEN_WIDTH if self.selected else self.PEN_WIDTH)

        # 矩形选中时使用实线，其他形状使用虚线
        if self.selected:
            if self.shape_type == "point":
                # 点类型使用点线和圆头
                pen.setStyle(QtCore.Qt.DotLine)
                pen.setCapStyle(QtCore.Qt.RoundCap)
            elif self.shape_type == "rectangle":
                # 矩形使用实线，与悬停效果保持一致
                pen.setStyle(QtCore.Qt.SolidLine)
            else:
                pen.setStyle(QtCore.Qt.DashLine)
        # 添加悬停效果特殊处理
        elif self.hovered and self.shape_type == "point":
            # 悬停时点使用实线
            pen.setStyle(QtCore.Qt.SolidLine)
            pen.setWidth(self.SELECT_PEN_WIDTH)
            pen.setCapStyle(QtCore.Qt.RoundCap)

        painter.setPen(pen)

        if self.mask is not None:
            image_to_draw = np.zeros(self.mask.shape + (4,), dtype=np.uint8)
            fill_color = (
                self.select_fill_color.getRgb()
                if self.selected
                else self.fill_color.getRgb()
            )
            image_to_draw[self.mask] = fill_color
            qimage = QtGui.QImage.fromData(
                labelme.utils.img_arr_to_data(image_to_draw))
            qimage = qimage.scaled(
                qimage.size() * self.scale,
                QtCore.Qt.IgnoreAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )

            painter.drawImage(self._scale_point(point=self.points[0]), qimage)

            line_path = QtGui.QPainterPath()
            contours = skimage.measure.find_contours(
                np.pad(self.mask, pad_width=1))
            for contour in contours:
                contour += [self.points[0].y(), self.points[0].x()]
                line_path.moveTo(
                    self._scale_point(QtCore.QPointF(
                        contour[0, 1], contour[0, 0]))
                )
                for point in contour[1:]:
                    line_path.lineTo(
                        self._scale_point(QtCore.QPointF(point[1], point[0]))
                    )
            painter.drawPath(line_path)

        if self.points:
            line_path = QtGui.QPainterPath()
            vrtx_path = QtGui.QPainterPath()
            negative_vrtx_path = QtGui.QPainterPath()

            if self.shape_type in ["rectangle", "mask"]:
                assert len(self.points) in [1, 2]
                if len(self.points) == 2:
                    rectangle = QtCore.QRectF(
                        self._scale_point(self.points[0]),
                        self._scale_point(self.points[1]),
                    )
                    line_path.addRect(rectangle)

                    # 为矩形添加特殊的选中效果
                    if self.selected and self.shape_type == "rectangle":
                        # 添加内部发光效果
                        painter.save()

                        # 绘制内部轮廓（略小于原始矩形）
                        inner_rect = rectangle.adjusted(1, 1, -1, -1)
                        inner_pen = QtGui.QPen(self.select_line_color)
                        inner_pen.setWidth(1)
                        painter.setPen(inner_pen)
                        painter.drawRect(inner_rect)

                        # 绘制外部轮廓（略大于原始矩形）
                        outer_rect = rectangle.adjusted(-1, -1, 1, 1)
                        outer_pen = QtGui.QPen(self.select_line_color)
                        outer_pen.setWidth(1)
                        outer_pen.setStyle(QtCore.Qt.DotLine)
                        painter.setPen(outer_pen)
                        painter.drawRect(outer_rect)

                        painter.restore()

                if self.shape_type == "rectangle":
                    for i in range(len(self.points)):
                        self.drawVertex(vrtx_path, i)
            elif self.shape_type == "circle":
                assert len(self.points) in [1, 2]
                if len(self.points) == 2:
                    raidus = labelme.utils.distance(
                        self._scale_point(self.points[0] - self.points[1])
                    )
                    line_path.addEllipse(
                        self._scale_point(self.points[0]), raidus, raidus
                    )
                for i in range(len(self.points)):
                    self.drawVertex(vrtx_path, i)
            elif self.shape_type == "linestrip":
                line_path.moveTo(self._scale_point(self.points[0]))
                for i, p in enumerate(self.points):
                    line_path.lineTo(self._scale_point(p))
                    self.drawVertex(vrtx_path, i)
            elif self.shape_type == "points":
                assert len(self.points) == len(self.point_labels)
                for i, point_label in enumerate(self.point_labels):
                    if point_label == 1:
                        self.drawVertex(vrtx_path, i)
                    else:
                        self.drawVertex(negative_vrtx_path, i)
            else:
                line_path.moveTo(self._scale_point(self.points[0]))
                # Uncommenting the following line will draw 2 paths
                # for the 1st vertex, and make it non-filled, which
                # may be desirable.
                # self.drawVertex(vrtx_path, 0)

                for i, p in enumerate(self.points):
                    line_path.lineTo(self._scale_point(p))
                    self.drawVertex(vrtx_path, i)
                if self.isClosed():
                    line_path.lineTo(self._scale_point(self.points[0]))

            painter.drawPath(line_path)
            if vrtx_path.length() > 0:
                painter.drawPath(vrtx_path)
                painter.fillPath(vrtx_path, self._vertex_fill_color)
            if self.fill and self.mask is None:
                color = self.select_fill_color if self.selected else self.fill_color
                # 矩形选中时增强填充效果
                if self.selected and self.shape_type == "rectangle":
                    # 增加选中矩形的填充透明度
                    r, g, b, a = color.getRgb()
                    color = QtGui.QColor(r, g, b, min(a + 20, 255))  # 略微增加透明度
                painter.fillPath(line_path, color)

            pen.setColor(QtGui.QColor(255, 0, 0, 255))
            painter.setPen(pen)
            painter.drawPath(negative_vrtx_path)
            painter.fillPath(negative_vrtx_path, QtGui.QColor(255, 0, 0, 255))

            # 为点类型添加选中和悬停效果
            if self.shape_type == "point" and len(self.points) > 0:
                point = self._scale_point(self.points[0])

                # 选中状态的外部轮廓
                if self.selected:
                    # 保存当前画笔设置
                    painter.save()
                    # 创建外部轮廓环
                    outer_pen = QtGui.QPen(QtGui.QColor(255, 255, 0))
                    outer_pen.setWidth(2)
                    outer_pen.setStyle(QtCore.Qt.SolidLine)
                    painter.setPen(outer_pen)

                    # 计算外部环的尺寸（比点大一些）
                    radius = self.select_point_size * 1.2

                    # 绘制外部环
                    painter.drawEllipse(point, radius, radius)
                    painter.restore()
                # 悬停状态的效果
                elif self.hovered:
                    # 保存当前画笔设置
                    painter.save()

                    # 绘制外部发光效果
                    for i in range(2):
                        glow_size = self.select_point_size * (1.1 + i * 0.15)
                        alpha = 120 - i * 40  # 渐变透明度
                        glow_color = QtGui.QColor(255, 255, 255, alpha)
                        glow_pen = QtGui.QPen(glow_color)
                        glow_pen.setWidth(1)
                        painter.setPen(glow_pen)
                        painter.drawEllipse(point, glow_size/2, glow_size/2)

                    # 绘制内部高亮圆
                    highlight_color = self.vertex_fill_color.lighter(150)
                    highlight_color.setAlpha(180)
                    painter.setBrush(QtGui.QBrush(highlight_color))
                    painter.setPen(QtCore.Qt.NoPen)
                    painter.drawEllipse(
                        point, self.point_size * 0.7, self.point_size * 0.7)

                    painter.restore()

        # 绘制标签名称
        if Shape.show_label_names and self.label and len(self.points) > 0:
            painter.save()

            # 为不同类型的形状设置不同的标签位置
            if self.shape_type == "rectangle":
                # 矩形标签显示在左上角
                rect = QtCore.QRectF(self._scale_point(
                    self.points[0]), self._scale_point(self.points[1]))
                label_pos = QtCore.QPointF(rect.x(), rect.y() - 40)  # 稍微往上移动一点
            elif self.shape_type == "point":
                # 点标签显示在点右侧
                point = self._scale_point(self.points[0])
                label_pos = QtCore.QPointF(
                    point.x() + self.point_size * 1.5, point.y() - self.point_size)
            elif self.shape_type == "circle":
                # 圆形标签显示在圆心上方
                center = self._scale_point(self.points[0])
                radius = labelme.utils.distance(
                    self._scale_point(self.points[0] - self.points[1]))
                label_pos = QtCore.QPointF(
                    center.x() - radius/2, center.y() - radius - 20)
            elif self.shape_type == "line":
                # 线段标签显示在中点上方
                p1 = self._scale_point(self.points[0])
                p2 = self._scale_point(self.points[1])
                label_pos = QtCore.QPointF(
                    (p1.x() + p2.x())/2, (p1.y() + p2.y())/2 - 15)
            elif self.shape_type == "linestrip" or self.shape_type == "polygon":
                # 多边形和折线标签显示在第一个点上方
                p1 = self._scale_point(self.points[0])
                label_pos = QtCore.QPointF(p1.x(), p1.y() - 15)
            else:
                # 其他形状默认在第一个点上方
                p1 = self._scale_point(self.points[0])
                label_pos = QtCore.QPointF(p1.x(), p1.y() - 15)

            # 设置字体
            font = QtGui.QFont()
            font.setBold(True)
            font.setPointSize(9)
            painter.setFont(font)

            # 计算文本区域
            fm = painter.fontMetrics()
            text_rect = fm.boundingRect(self.label)

            # 创建标签背景矩形
            bg_rect = QtCore.QRectF(
                label_pos.x(),
                label_pos.y(),
                text_rect.width() + 10,
                text_rect.height() + 6
            )

            # 使用形状颜色，但半透明
            bg_color = QtGui.QColor(self.fill_color)
            if self.selected:
                # 选中状态下背景色更深
                bg_color.setAlpha(180)
            else:
                bg_color.setAlpha(120)

            # 绘制圆角矩形背景
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(bg_color)
            painter.drawRoundedRect(bg_rect, 5, 5)

            # 设置文本颜色 - 根据背景亮度自动调整
            r, g, b = bg_color.getRgb()[:3]
            brightness = (r * 299 + g * 587 + b * 114) / 1000
            if brightness > 128:
                # 深色文本用于浅色背景
                text_color = QtGui.QColor(40, 40, 40)
            else:
                # 浅色文本用于深色背景
                text_color = QtGui.QColor(250, 250, 250)

            # 绘制文本
            painter.setPen(text_color)
            painter.drawText(
                bg_rect,
                QtCore.Qt.AlignCenter,
                self.label
            )

            painter.restore()

    def drawVertex(self, path, i):
        # 选中状态时使用更大的顶点尺寸
        if self.selected and self.shape_type == "point":
            d = self.select_point_size * 1.5
        elif self.selected and self.shape_type == "rectangle":
            # 矩形选中时使用更大的顶点尺寸
            d = self.select_point_size * 1.2
        elif self.selected:
            d = self.select_point_size
        else:
            d = self.point_size

        shape = self.point_type
        point = self._scale_point(self.points[i])
        if i == self._highlightIndex:
            size, shape = self._highlightSettings[self._highlightMode]
            d *= size
        if self._highlightIndex is not None:
            self._vertex_fill_color = self.hvertex_fill_color
        else:
            self._vertex_fill_color = self.vertex_fill_color

        # 点类型选中时保持圆形
        if self.shape_type == "point" and self.selected:
            shape = self.P_ROUND
        # 矩形选中时使用方形顶点增强显示效果
        elif self.shape_type == "rectangle" and self.selected:
            shape = self.P_SQUARE

        if shape == self.P_SQUARE:
            path.addRect(point.x() - d / 2, point.y() - d / 2, d, d)
        elif shape == self.P_ROUND:
            path.addEllipse(point, d / 2.0, d / 2.0)
        else:
            assert False, "unsupported vertex shape"

    def nearestVertex(self, point, epsilon):
        min_distance = float("inf")
        min_i = None
        point = QtCore.QPointF(point.x() * self.scale, point.y() * self.scale)
        for i, p in enumerate(self.points):
            p = QtCore.QPointF(p.x() * self.scale, p.y() * self.scale)
            dist = labelme.utils.distance(p - point)
            if dist <= epsilon and dist < min_distance:
                min_distance = dist
                min_i = i
        return min_i

    def nearestEdge(self, point, epsilon):
        min_distance = float("inf")
        post_i = None
        point = QtCore.QPointF(point.x() * self.scale, point.y() * self.scale)
        for i in range(len(self.points)):
            start = self.points[i - 1]
            end = self.points[i]
            start = QtCore.QPointF(
                start.x() * self.scale, start.y() * self.scale)
            end = QtCore.QPointF(end.x() * self.scale, end.y() * self.scale)
            line = [start, end]
            dist = labelme.utils.distancetoline(point, line)
            if dist <= epsilon and dist < min_distance:
                min_distance = dist
                post_i = i
        return post_i

    def containsPoint(self, point):
        if self.mask is not None:
            y = np.clip(
                int(round(point.y() - self.points[0].y())),
                0,
                self.mask.shape[0] - 1,
            )
            x = np.clip(
                int(round(point.x() - self.points[0].x())),
                0,
                self.mask.shape[1] - 1,
            )
            return self.mask[y, x]
        return self.makePath().contains(point)

    def makePath(self):
        if self.shape_type in ["rectangle", "mask"]:
            path = QtGui.QPainterPath()
            if len(self.points) == 2:
                path.addRect(QtCore.QRectF(self.points[0], self.points[1]))
        elif self.shape_type == "circle":
            path = QtGui.QPainterPath()
            if len(self.points) == 2:
                raidus = labelme.utils.distance(
                    self.points[0] - self.points[1])
                path.addEllipse(self.points[0], raidus, raidus)
        else:
            path = QtGui.QPainterPath(self.points[0])
            for p in self.points[1:]:
                path.lineTo(p)
        return path

    def boundingRect(self):
        return self.makePath().boundingRect()

    def moveBy(self, offset):
        self.points = [p + offset for p in self.points]

    def moveVertexBy(self, i, offset):
        self.points[i] = self.points[i] + offset

    def highlightVertex(self, i, action):
        """Highlight a vertex appropriately based on the current action

        Args:
            i (int): The vertex index
            action (int): The action
            (see Shape.NEAR_VERTEX and Shape.MOVE_VERTEX)
        """
        self._highlightIndex = i
        self._highlightMode = action

    def highlightClear(self):
        """Clear the highlighted point"""
        self._highlightIndex = None

    def copy(self):
        shape = Shape(
            label=self.label,
            shape_type=self.shape_type,
            group_id=self.group_id,
            flags=self.flags,
            mask=self.mask.copy() if self.mask is not None else None,
            description=self.description,
        )
        shape.points = [p for p in self.points]
        shape.point_labels = [l for l in self.point_labels]
        shape._closed = self._closed
        shape.fill_color = self.fill_color
        shape.select_line_color = self.select_line_color
        shape.select_fill_color = self.select_fill_color
        shape._shape_raw = self._shape_raw
        shape._points_raw = self._points_raw
        if hasattr(self, '_point_labels_raw'):
            shape._point_labels_raw = self._point_labels_raw
        if hasattr(self, '_mask_raw'):
            shape._mask_raw = self._mask_raw
        return shape

    def __len__(self):
        return len(self.points)

    def __getitem__(self, key):
        return self.points[key]

    def __setitem__(self, key, value):
        self.points[key] = value

    def setHoverState(self, hovered):
        """设置形状的悬停状态"""
        self.hovered = hovered
