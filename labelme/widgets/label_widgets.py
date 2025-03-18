"""
标签视图模块 - 高效现代化的UI组件

该模块实现了与标签模型配合使用的视图组件，包括标签树形视图、分类树形视图
以及自定义的委托类用于实现现代化的UI风格。
"""

import html
from typing import Dict, List, Optional, Set, Any, Callable, Union

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal, QModelIndex, QSize, QRect, QPoint
from PyQt5.QtGui import (
    QPalette, QFont, QColor, QPainter, QTextDocument,
    QFontMetrics, QPen, QBrush, QLinearGradient
)
from PyQt5.QtWidgets import (
    QTreeView, QStyledItemDelegate, QStyle, QApplication,
    QAbstractItemView, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QSizePolicy, QSpacerItem, QPushButton
)

from .label_model import LabelModel


class ModernItemDelegate(QStyledItemDelegate):
    """
    现代化UI样式的自定义委托

    提供带自定义渲染的HTML文本支持、平滑圆角效果和微妙动画
    """

    def __init__(self, parent=None):
        """初始化委托"""
        super(ModernItemDelegate, self).__init__(parent)
        self.doc = QTextDocument(self)
        self._hover_index = QModelIndex()

    def set_hover_index(self, index):
        """设置当前鼠标悬停的索引"""
        self._hover_index = index

    def paint(self, painter: QPainter, option: QStyle.State, index: QModelIndex):
        """
        绘制项目

        Args:
            painter: 画笔
            option: 样式选项
            index: 模型索引
        """
        # 保存画笔状态
        painter.save()

        # 复制样式选项用于自定义修改
        options = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(options, index)

        # 获取是否为分类项
        is_category = index.data(Qt.UserRole + 2) if index.isValid() else False

        # 设置背景和选择状态
        if options.state & QStyle.State_Selected:
            if is_category:
                bg_color = QColor("#2979ff")
                text_color = QColor("#ffffff")
            else:
                bg_color = QColor("#e3f2fd")
                text_color = QColor("#0d47a1")
        elif options.state & QStyle.State_MouseOver or self._hover_index == index:
            if is_category:
                bg_color = QColor("#bbdefb")
                text_color = QColor("#1976d2")
            else:
                bg_color = QColor("#f5f5f5")
                text_color = QColor("#333333")
        else:
            if is_category:
                bg_color = QColor("#f5f5f5")
                text_color = QColor("#37474f")
            else:
                bg_color = QColor("#ffffff")
                text_color = QColor("#333333")

        # 绘制背景
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(bg_color))

        # 圆角矩形背景 - 分类项使用更大的圆角
        rect = option.rect
        radius = 8 if is_category else 5
        painter.drawRoundedRect(
            rect.adjusted(2, 1, -2, -1),
            radius, radius
        )

        # 设置文本
        self.doc.setHtml(options.text)

        # 设置文本颜色
        ctx = QtGui.QAbstractTextDocumentLayout.PaintContext()
        ctx.palette.setColor(QPalette.Text, text_color)

        # 调整文本位置
        text_rect = option.rect.adjusted(10, 2, -10, -2)
        painter.translate(text_rect.topLeft())

        # 绘制文本
        self.doc.documentLayout().draw(painter, ctx)

        # 恢复画笔状态
        painter.restore()

    def sizeHint(self, option, index):
        """调整项目尺寸，确保足够显示内容"""
        is_category = index.data(Qt.UserRole + 2) if index.isValid() else False
        self.doc.setHtml(index.data(Qt.DisplayRole) or "")

        # 分类项使用稍大的尺寸
        height = self.doc.size().height() + (10 if is_category else 6)
        width = self.doc.idealWidth() + 40

        return QSize(width, int(height))


class ModernTreeView(QTreeView):
    """
    现代化风格的树形视图基类

    提供平滑动画、微妙交互效果和一致的现代UI风格
    """

    def __init__(self, parent=None):
        """初始化树形视图"""
        super(ModernTreeView, self).__init__(parent)

        # 设置基本样式
        self.setHeaderHidden(True)
        self.setAnimated(True)
        self.setIndentation(20)
        self.setIconSize(QSize(20, 20))

        # 交互设置
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)

        # 创建自定义委托
        self.delegate = ModernItemDelegate()
        self.setItemDelegate(self.delegate)

        # 设置视觉效果
        self.setAlternatingRowColors(False)
        self.setStyleSheet("""
            ModernTreeView {
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 5px;
                outline: none;
            }
            ModernTreeView::item {
                padding: 4px;
                margin: 2px 0px;
            }
        """)

        # 鼠标追踪，用于悬停效果
        self.setMouseTracking(True)

    def mouseMoveEvent(self, event):
        """处理鼠标移动事件，更新悬停索引"""
        super(ModernTreeView, self).mouseMoveEvent(event)
        hover_index = self.indexAt(event.pos())
        self.delegate.set_hover_index(hover_index)
        self.viewport().update()


class LabelTreeView(ModernTreeView):
    """
    标签树形视图 - 显示所有标签项

    按分类组织和显示标签，支持选择、拖放等操作
    """

    # 自定义信号
    itemSelected = pyqtSignal(object)
    itemDeselected = pyqtSignal(object)
    itemDoubleClicked = pyqtSignal(object)
    contextMenuRequested = pyqtSignal(QPoint, object)
    itemCheckStateChanged = pyqtSignal(object, bool)  # 项目ID, 是否选中

    def __init__(self, parent=None):
        """初始化标签树视图"""
        super(LabelTreeView, self).__init__(parent)

        # 数据存储
        self._model = QtGui.QStandardItemModel(self)
        self._categories = {}  # 分类名称 -> 分类项目
        self._items = {}  # 项目ID -> 项目
        self._category_counts = {}  # 分类名称 -> 项目数量

        # 设置模型
        self.setModel(self._model)
        self.setUniformRowHeights(True)
        self.setHeaderHidden(True)

        # 连接信号
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._context_menu_requested)
        self.selectionModel().selectionChanged.connect(self._selection_changed)
        self.doubleClicked.connect(self._item_double_clicked)

        # 连接复选框状态变化信号
        self._model.itemChanged.connect(self._item_check_state_changed)

    def _context_menu_requested(self, point):
        """处理上下文菜单请求"""
        index = self.indexAt(point)
        if index.isValid():
            item = self._model.itemFromIndex(index)
            self.contextMenuRequested.emit(point, item.data(Qt.UserRole))

    def _selection_changed(self, selected, deselected):
        """处理选择变化"""
        # 处理选中的项
        for index in selected.indexes():
            item = self._model.itemFromIndex(index)
            data = item.data(Qt.UserRole)
            if data and not item.data(Qt.UserRole + 2):  # 不是分类项
                self.itemSelected.emit(data)

        # 处理取消选中的项
        for index in deselected.indexes():
            item = self._model.itemFromIndex(index)
            data = item.data(Qt.UserRole)
            if data and not item.data(Qt.UserRole + 2):  # 不是分类项
                self.itemDeselected.emit(data)

    def _item_double_clicked(self, index):
        """处理双击事件"""
        if index.isValid():
            item = self._model.itemFromIndex(index)
            data = item.data(Qt.UserRole)
            if data and not item.data(Qt.UserRole + 2):  # 不是分类项
                self.itemDoubleClicked.emit(data)

    def _item_check_state_changed(self, item):
        """处理复选框状态变化"""
        if item.data(Qt.UserRole + 2):  # 是分类项
            return

        data = item.data(Qt.UserRole)
        if data:
            self.itemCheckStateChanged.emit(
                data, item.checkState() == Qt.Checked)

    def clear(self):
        """清空视图"""
        self._model.clear()
        self._categories.clear()
        self._items.clear()
        self._category_counts.clear()

    def selected_items(self):
        """获取所有选中的项目数据"""
        result = []
        for index in self.selectedIndexes():
            item = self._model.itemFromIndex(index)
            data = item.data(Qt.UserRole)
            if data and not item.data(Qt.UserRole + 2):  # 不是分类项
                result.append(data)
        return result

    def get_category_item(self, name, display_name=None):
        """获取或创建分类项"""
        if name in self._categories:
            return self._categories[name]

        # 创建新分类项
        item = QtGui.QStandardItem(display_name or name)

        # 设置粗体字体
        font = item.font()
        font.setBold(True)
        item.setFont(font)

        # 存储分类标记
        item.setData(True, Qt.UserRole + 2)

        # 添加到模型
        self._model.appendRow(item)
        self._categories[name] = item

        return item

    def add_item(self, item_id, display_text, data, category_name, category_display_name=None):
        """
        添加项目到视图

        Args:
            item_id: 项目ID
            display_text: 显示文本（可包含HTML）
            data: 项目数据
            category_name: 分类名称
            category_display_name: 分类显示名称
        """
        # 确保数据类型正确
        if not isinstance(item_id, (int, str)):
            return None

        # 检查项目是否已存在
        if item_id in self._items:
            return self._items[item_id]

        # 获取或创建分类项
        category_item = self.get_category_item(
            category_name, category_display_name)

        # 创建项目
        item = QtGui.QStandardItem(display_text)

        # 设置复选框状态
        item.setCheckable(True)
        visible = data.get('visible', True) if isinstance(data, dict) else True
        item.setCheckState(Qt.Checked if visible else Qt.Unchecked)

        # 存储项目数据
        item.setData(data, Qt.UserRole)
        item.setData(item_id, Qt.UserRole + 1)
        item.setData(False, Qt.UserRole + 2)  # 不是分类项

        # 添加到分类下
        category_item.appendRow(item)

        # 存储项目映射
        self._items[item_id] = item

        # 更新分类计数
        self._update_category_count(category_name)

        return item

    def remove_item(self, item_id):
        """删除视图中的项目"""
        if item_id not in self._items:
            return False

        # 获取项目
        item = self._items[item_id]

        # 获取父项（分类）
        parent = item.parent() or self._model.invisibleRootItem()

        # 查找项目所在行
        row = item.row()

        # 从父项中移除
        parent.removeRow(row)

        # 从缓存中移除
        del self._items[item_id]

        # 尝试更新分类计数
        for category_name, category_item in self._categories.items():
            if category_item == parent:
                self._update_category_count(category_name)
                break

        return True

    def update_item(self, item_id, display_text=None, data=None,
                    category_name=None, category_display_name=None):
        """更新项目"""
        if item_id not in self._items:
            return False

        item = self._items[item_id]
        old_parent = item.parent() or self._model.invisibleRootItem()

        # 更新显示文本
        if display_text is not None:
            item.setText(display_text)

        # 更新数据
        if data is not None:
            item.setData(data, Qt.UserRole)

        # 更新分类
        if category_name is not None:
            if category_name in self._categories:
                # 检查是否需要移动项目
                category_item = self._categories[category_name]
                if category_item != old_parent:
                    # 从旧分类中移除
                    row = item.row()
                    old_parent.removeRow(row)

                    # 添加到新分类
                    category_item.appendRow(item)

                    # 更新两个分类的计数
                    for name, c_item in self._categories.items():
                        if c_item == old_parent or c_item == category_item:
                            self._update_category_count(name)
            else:
                # 创建新分类并移动项目
                new_category = self.get_category_item(
                    category_name, category_display_name)

                # 从旧分类中移除
                row = item.row()
                old_parent.removeRow(row)

                # 添加到新分类
                new_category.appendRow(item)

                # 更新两个分类的计数
                for name, c_item in self._categories.items():
                    if c_item == old_parent or c_item == new_category:
                        self._update_category_count(name)

        return True

    def select_item(self, item_id):
        """选中指定项目"""
        if item_id not in self._items:
            return False

        item = self._items[item_id]
        index = self._model.indexFromItem(item)

        # 选中项目
        self.selectionModel().select(
            index, QtCore.QItemSelectionModel.Select
        )

        # 滚动到选中的项目
        self.scrollTo(index)

        return True

    def deselect_item(self, item_id):
        """取消选中指定项目"""
        if item_id not in self._items:
            return False

        item = self._items[item_id]
        index = self._model.indexFromItem(item)

        # 取消选中项目
        self.selectionModel().select(
            index, QtCore.QItemSelectionModel.Deselect
        )

        return True

    def find_item_by_data(self, data_compare_func):
        """
        通过自定义比较函数查找项目

        Args:
            data_compare_func: 用于比较项目数据的函数
                接收一个参数（项目数据），返回布尔值

        Returns:
            找到的第一个项目ID，未找到则返回None
        """
        for item_id, item in self._items.items():
            data = item.data(Qt.UserRole)
            if data_compare_func(data):
                return item_id
        return None

    def _update_category_count(self, category_name):
        """更新分类项的计数显示"""
        if category_name not in self._categories:
            return

        category_item = self._categories[category_name]
        count = category_item.rowCount()

        # 特殊处理"未分类"
        display_name = category_name if category_name != "未分类" else "未分类"

        # 更新显示文本
        category_item.setText(f"{display_name} ({count})")


class UniqueLabelTreeView(ModernTreeView):
    """
    唯一标签树形视图 - 显示不重复的标签项

    按分类组织和显示不重复的标签，每个标签只显示一次
    """

    # 自定义信号
    itemSelected = pyqtSignal(object)
    itemDoubleClicked = pyqtSignal(object)

    def __init__(self, parent=None):
        """初始化唯一标签树形视图"""
        super(UniqueLabelTreeView, self).__init__(parent)

        # 设置模型和委托
        self._model = QtGui.QStandardItemModel()
        self.setModel(self._model)

        # 交互设置
        self.setSelectionMode(QAbstractItemView.SingleSelection)

        # 缓存
        self._categories = {}  # name -> item
        self._labels = {}      # label -> item

        # 连接信号
        self.doubleClicked.connect(self._item_double_clicked)
        self.selectionModel().selectionChanged.connect(self._selection_changed)

    def _selection_changed(self, selected, deselected):
        """处理选择变化"""
        # 只关注选中的项
        for index in selected.indexes():
            item = self._model.itemFromIndex(index)
            label = item.data(Qt.UserRole)
            if label and not item.data(Qt.UserRole + 2):  # 不是分类项
                self.itemSelected.emit(label)

    def _item_double_clicked(self, index):
        """处理双击事件"""
        if index.isValid():
            item = self._model.itemFromIndex(index)
            label = item.data(Qt.UserRole)
            if label and not item.data(Qt.UserRole + 2):  # 不是分类项
                self.itemDoubleClicked.emit(label)

    def clear(self):
        """清空视图"""
        self._model.clear()
        self._categories.clear()
        self._labels.clear()

    def get_category_item(self, name, display_name=None):
        """获取或创建分类项"""
        if name in self._categories:
            return self._categories[name]

        # 创建新分类项
        item = QtGui.QStandardItem(display_name or name)

        # 设置粗体字体
        font = item.font()
        font.setBold(True)
        item.setFont(font)

        # 存储分类标记
        item.setData(True, Qt.UserRole + 2)

        # 添加到模型
        self._model.appendRow(item)
        self._categories[name] = item

        return item

    def add_label(self, label, display_text, category_name, category_display_name=None):
        """
        添加标签到视图

        Args:
            label: 标签名称
            display_text: 显示文本（可包含HTML）
            category_name: 分类名称
            category_display_name: 分类显示名称
        """
        # 检查标签是否已存在
        if label in self._labels:
            # 更新已有标签
            self._labels[label].setText(display_text)
            return self._labels[label]

        # 获取或创建分类项
        category_item = self.get_category_item(
            category_name, category_display_name)

        # 创建标签项
        item = QtGui.QStandardItem(display_text)
        item.setData(label, Qt.UserRole)
        item.setData(False, Qt.UserRole + 2)  # 不是分类项

        # 添加到分类下
        category_item.appendRow(item)
        self._labels[label] = item

        # 更新分类项文本
        self._update_category_count(category_name)

        # 确保分类是展开状态
        self.expand(self._model.indexFromItem(category_item))

        return item

    def remove_label(self, label):
        """从视图中移除标签"""
        if label not in self._labels:
            return False

        # 获取标签项
        item = self._labels[label]

        # 获取父项（分类）
        parent = item.parent() or self._model.invisibleRootItem()

        # 查找项目所在行
        row = item.row()

        # 从父项中移除
        parent.removeRow(row)

        # 从缓存中移除
        del self._labels[label]

        # 更新分类计数
        for category_name, category_item in self._categories.items():
            if category_item == parent:
                self._update_category_count(category_name)
                break

        return True

    def update_label(self, label, display_text=None,
                     category_name=None, category_display_name=None):
        """更新标签"""
        if label not in self._labels:
            return False

        item = self._labels[label]
        old_parent = item.parent() or self._model.invisibleRootItem()

        # 更新显示文本
        if display_text is not None:
            item.setText(display_text)

        # 更新分类
        if category_name is not None:
            if category_name in self._categories:
                # 检查是否需要移动项目
                category_item = self._categories[category_name]
                if category_item != old_parent:
                    # 从旧分类中移除
                    row = item.row()
                    old_parent.removeRow(row)

                    # 添加到新分类
                    category_item.appendRow(item)

                    # 更新两个分类的计数
                    for name, c_item in self._categories.items():
                        if c_item == old_parent or c_item == category_item:
                            self._update_category_count(name)
            else:
                # 创建新分类并移动项目
                new_category = self.get_category_item(
                    category_name, category_display_name)

                # 从旧分类中移除
                row = item.row()
                old_parent.removeRow(row)

                # 添加到新分类
                new_category.appendRow(item)

                # 更新两个分类的计数
                for name, c_item in self._categories.items():
                    if c_item == old_parent or c_item == new_category:
                        self._update_category_count(name)

        return True

    def _update_category_count(self, category_name):
        """更新分类项的计数显示"""
        if category_name not in self._categories:
            return

        category_item = self._categories[category_name]
        count = category_item.rowCount()

        # 更新显示文本
        category_item.setText(f"{category_name} ({count})")


class LabelCountWidget(QWidget):
    """
    标签计数小部件

    显示标签统计信息的小部件，现代简约风格
    """

    def __init__(self, parent=None):
        """初始化标签计数小部件"""
        super(LabelCountWidget, self).__init__(parent)

        # 布局
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        # 标签图标
        self.icon_label = QLabel()
        try:
            # 尝试加载图标资源
            tag_pixmap = QtGui.QPixmap(":/icons/tag.png")
            if not tag_pixmap.isNull():
                self.icon_label.setPixmap(tag_pixmap.scaled(16, 16))
            else:
                # 如果资源不存在，使用文本替代
                self.icon_label.setText("🏷")
        except:
            # 出错时使用文本替代
            self.icon_label.setText("🏷")
        layout.addWidget(self.icon_label)

        # 标签数量
        self.count_label = QLabel("0 标签")
        font = self.count_label.font()
        font.setPointSize(9)
        self.count_label.setFont(font)
        layout.addWidget(self.count_label)

        # 右侧间隔
        layout.addItem(QSpacerItem(
            10, 10, QSizePolicy.Expanding, QSizePolicy.Minimum))

        # 设置样式
        self.setStyleSheet("""
            QLabel {
                color: #455a64;
            }
        """)

    def update_count(self, count):
        """更新标签计数"""
        self.count_label.setText(f"{count} 标签")


class CategoryHeaderWidget(QWidget):
    """
    分类标题小部件

    分类树视图的标题控件，显示统计信息并提供交互
    """

    def __init__(self, title="分类", parent=None):
        """初始化分类标题小部件"""
        super(CategoryHeaderWidget, self).__init__(parent)

        # 布局
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(10)

        # 标题
        self.title_label = QLabel(title)
        font = self.title_label.font()
        font.setPointSize(11)
        font.setBold(True)
        self.title_label.setFont(font)
        layout.addWidget(self.title_label)

        # 计数
        self.count_label = QLabel("(0)")
        font = self.count_label.font()
        font.setPointSize(10)
        self.count_label.setFont(font)
        layout.addWidget(self.count_label)

        # 右侧间隔
        layout.addItem(QSpacerItem(
            10, 10, QSizePolicy.Expanding, QSizePolicy.Minimum))

        # 样式
        self.setStyleSheet("""
            CategoryHeaderWidget {
                background-color: #f5f5f5;
                border-radius: 8px;
                border-bottom: 1px solid #e0e0e0;
            }
            QLabel {
                color: #37474f;
            }
        """)

    def update_count(self, count):
        """更新分类计数"""
        self.count_label.setText(f"({count})")
