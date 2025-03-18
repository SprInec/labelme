"""
标签面板模块 - 集成标签管理的面板组件

该模块实现了标签列表和多边形标签的面板组件，集成了标签模型和视图组件
提供完整的用户界面和交互功能。
"""

from typing import Dict, List, Optional, Set, Any, Callable, Union
import html

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal, QModelIndex, QSize, QRect, QPoint
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QMenu, QAction, QSplitter, QFrame,
    QScrollArea, QSizePolicy, QToolButton, QInputDialog
)

from .label_model import LabelModel, LabelCategory
from .label_widgets import (
    LabelTreeView, UniqueLabelTreeView, ModernTreeView,
    LabelCountWidget, CategoryHeaderWidget
)


class SearchBox(QWidget):
    """
    搜索框小部件

    提供带图标和动画效果的搜索输入框
    """

    # 自定义信号
    textChanged = pyqtSignal(str)
    searchCleared = pyqtSignal()

    def __init__(self, parent=None):
        """初始化搜索框小部件"""
        super(SearchBox, self).__init__(parent)

        # 主布局
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        # 搜索图标 - 使用内置资源或字符图标代替
        self.search_icon = QLabel()
        try:
            # 尝试加载图标资源
            search_pixmap = QtGui.QPixmap(":/icons/search.png")
            if not search_pixmap.isNull():
                self.search_icon.setPixmap(search_pixmap.scaled(16, 16))
            else:
                # 如果资源不存在，使用文本替代
                self.search_icon.setText("🔍")
        except:
            # 出错时使用文本替代
            self.search_icon.setText("🔍")
        layout.addWidget(self.search_icon)

        # 搜索输入框
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("搜索标签...")
        self.search_edit.textChanged.connect(self._text_changed)
        layout.addWidget(self.search_edit)

        # 清除按钮 - 使用内置资源或字符图标代替
        self.clear_button = QToolButton()
        try:
            # 尝试加载图标资源
            clear_pixmap = QtGui.QPixmap(":/icons/clear.png")
            if not clear_pixmap.isNull():
                self.clear_button.setIcon(
                    QtGui.QIcon(clear_pixmap.scaled(12, 12)))
            else:
                # 如果资源不存在，使用文本替代
                self.clear_button.setText("✖")
        except:
            # 出错时使用文本替代
            self.clear_button.setText("✖")
        self.clear_button.setCursor(Qt.PointingHandCursor)
        self.clear_button.clicked.connect(self._clear_search)
        self.clear_button.setVisible(False)
        layout.addWidget(self.clear_button)

        # 样式
        self.setStyleSheet("""
            SearchBox {
                background-color: #f5f5f5;
                border: 1px solid #e0e0e0;
                border-radius: 15px;
                padding: 5px 10px;
            }
            QLineEdit {
                background-color: transparent;
                border: none;
                font-size: 13px;
                color: #37474f;
            }
            QToolButton {
                border: none;
                background: transparent;
            }
        """)

    def _text_changed(self, text):
        """处理文本变化"""
        self.clear_button.setVisible(bool(text))
        self.textChanged.emit(text)

    def _clear_search(self):
        """清除搜索"""
        self.search_edit.clear()
        self.searchCleared.emit()

    def set_search_text(self, text):
        """设置搜索文本"""
        self.search_edit.setText(text)


class LabelListPanel(QWidget):
    """
    标签列表面板

    显示所有标签项的面板，按分类组织，提供搜索、过滤等功能
    """

    # 自定义信号
    itemSelected = pyqtSignal(object)
    itemDeselected = pyqtSignal(object)
    itemDoubleClicked = pyqtSignal(object)

    def __init__(self, label_model: LabelModel, parent=None):
        """
        初始化标签列表面板

        Args:
            label_model: 标签数据模型
            parent: 父窗口
        """
        super(LabelListPanel, self).__init__(parent)

        # 保存模型引用
        self.label_model = label_model

        # 初始化UI
        self._setup_ui()

        # 连接信号
        self._connect_signals()

        # 加载初始数据
        self._load_data()

    def _setup_ui(self):
        """设置UI组件"""
        # 主布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # 头部区域
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(10)

        # 标题
        self.title_label = QLabel("标签列表")
        font = self.title_label.font()
        font.setPointSize(12)
        font.setBold(True)
        self.title_label.setFont(font)
        self.title_label.setStyleSheet("color: #37474f;")
        header_layout.addWidget(self.title_label)

        # 计数小部件
        self.count_widget = LabelCountWidget()
        header_layout.addWidget(self.count_widget)

        # 右侧间隔
        header_layout.addStretch()

        # 头部菜单按钮
        self.menu_button = QToolButton()
        try:
            # 尝试加载图标
            menu_icon = QtGui.QIcon(":/icons/menu.png")
            if not menu_icon.isNull():
                self.menu_button.setIcon(menu_icon)
            else:
                # 如果资源不存在，使用文本替代
                self.menu_button.setText("≡")
        except:
            # 出错时使用文本替代
            self.menu_button.setText("≡")
        self.menu_button.setCursor(Qt.PointingHandCursor)
        self.menu_button.setPopupMode(QToolButton.InstantPopup)

        # 头部菜单
        self.header_menu = QMenu(self)
        self.action_expand_all = QAction("展开全部", self)
        self.action_collapse_all = QAction("折叠全部", self)
        self.action_refresh = QAction("刷新", self)

        self.header_menu.addAction(self.action_expand_all)
        self.header_menu.addAction(self.action_collapse_all)
        self.header_menu.addSeparator()
        self.header_menu.addAction(self.action_refresh)

        self.menu_button.setMenu(self.header_menu)
        header_layout.addWidget(self.menu_button)

        # 添加头部到主布局
        layout.addWidget(header_widget)

        # 搜索框
        self.search_box = SearchBox()
        layout.addWidget(self.search_box)

        # 标签树视图
        self.label_tree = LabelTreeView()
        layout.addWidget(self.label_tree)

        # 设置大小策略
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.setMinimumWidth(250)

        # 样式
        self.setStyleSheet("""
            LabelListPanel {
                background-color: white;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
            }
        """)

    def _connect_signals(self):
        """连接信号"""
        # 标签模型信号
        self.label_model.categoriesChanged.connect(self._refresh_view)
        self.label_model.labelsChanged.connect(self._refresh_view)
        self.label_model.itemsChanged.connect(self._refresh_view)

        # 搜索框信号
        self.search_box.textChanged.connect(self._filter_items)
        self.search_box.searchCleared.connect(self._refresh_view)

        # 树视图信号
        self.label_tree.itemSelected.connect(
            lambda item_data: self.itemSelected.emit(item_data))
        self.label_tree.itemDeselected.connect(
            lambda item_data: self.itemDeselected.emit(item_data))
        self.label_tree.itemDoubleClicked.connect(
            lambda item_data: self.itemDoubleClicked.emit(item_data))
        self.label_tree.contextMenuRequested.connect(
            self._show_context_menu)

        # 复选框状态变化
        self.label_tree.itemCheckStateChanged.connect(
            self._on_check_state_changed)

        # 菜单动作信号
        self.action_expand_all.triggered.connect(self.label_tree.expandAll)
        self.action_collapse_all.triggered.connect(self.label_tree.collapseAll)
        self.action_refresh.triggered.connect(self._refresh_view)

    def _load_data(self):
        """加载数据到树视图"""
        # 清空当前视图
        self.label_tree.clear()

        # 获取所有分类
        categories = self.label_model.get_categories()

        for category in categories:
            # 获取分类下的所有项目ID
            item_ids = self.label_model.get_items_by_category(category.name)

            # 添加每个项目到视图
            for item_id in item_ids:
                item_data = self.label_model.get_item(item_id)
                if item_data:
                    label = item_data["label"]
                    shape_type = item_data.get("shape_type", "")

                    # 创建显示文本
                    display_text = self._create_display_text(
                        item_data, shape_type)

                    # 添加到视图
                    self.label_tree.add_item(
                        item_id,
                        display_text,
                        item_data,
                        category.name,
                        category.display_name
                    )

        # 更新计数
        self._update_counts()

    def _create_display_text(self, item_data, shape_type=None):
        """创建显示文本，包含HTML格式"""
        # 获取基本信息
        label = item_data.get('label', '')
        shape_type = shape_type or item_data.get('shape_type', '')

        # 处理组ID
        if 'group_id' in item_data and item_data['group_id'] is not None:
            group_text = f" ({item_data['group_id']})"
        else:
            group_text = ""

        # 添加形状类型图标
        shape_icon = self._get_shape_type_icon(shape_type)

        # 获取颜色标记
        color_mark = ""
        if 'fill_color' in item_data and item_data['fill_color']:
            r, g, b = item_data['fill_color']
            color_mark = f'<font color="#{r:02x}{g:02x}{b:02x}">●</font> '

        # 组合显示文本
        return f"{shape_icon} {color_mark}{html.escape(label)}{group_text}"

    def _refresh_view(self):
        """刷新视图"""
        # 保存当前选中的项目
        selected_items = self.label_tree.selected_items()
        selected_ids = [item.get("id") for item in selected_items if item]

        # 重新加载数据
        self._load_data()

        # 恢复选中状态
        for item_id in selected_ids:
            self.label_tree.select_item(item_id)

    def _filter_items(self, text):
        """
        根据搜索文本过滤项目

        当文本为空时，显示所有项目；否则只显示匹配的项目
        """
        if not text:
            # 如果搜索文本为空，刷新视图显示全部
            self._refresh_view()
            return

        # 清空视图
        self.label_tree.clear()

        # 搜索文本（不区分大小写）
        text = text.lower()

        # 记录找到的分类
        found_categories = set()

        # 获取所有项目数据
        for item_id, item_data in self.label_model._item_map.items():
            label = item_data.get("label", "")
            shape_type = item_data.get("shape_type", "")

            # 检查标签是否匹配
            if text in label.lower():
                # 获取分类
                category_name = self.label_model._determine_category(item_data)

                # 记录找到的分类
                found_categories.add(category_name)

                # 创建显示文本
                display_text = self._create_display_text(item_data, shape_type)

                # 添加到视图
                self.label_tree.add_item(
                    item_id,
                    display_text,
                    item_data,
                    category_name
                )

        # 更新计数
        self._update_filtered_counts(found_categories)

    def _update_counts(self):
        """更新标签计数"""
        # 获取所有标签
        all_labels = self.label_model.get_labels()

        # 更新标签计数
        self.count_widget.update_count(len(all_labels))

    def _update_filtered_counts(self, categories):
        """更新过滤后的标签计数"""
        # 计算过滤后的标签数量
        count = 0
        for category_name in categories:
            items = self.label_model.get_items_by_category(category_name)
            count += len(items)

        # 更新标签计数
        self.count_widget.update_count(count)

    def _show_context_menu(self, pos, item_data):
        """显示上下文菜单"""
        if not item_data:
            return

        # 创建上下文菜单
        menu = QMenu(self)

        # 编辑标签动作
        edit_action = QAction("编辑标签", self)
        edit_action.triggered.connect(
            lambda: self._edit_label(item_data.get('id'), item_data))
        menu.addAction(edit_action)

        # 删除标签动作
        delete_action = QAction("删除标签", self)
        delete_action.triggered.connect(
            lambda: self._delete_item(item_data.get('id')))
        menu.addAction(delete_action)

        # 分隔线
        menu.addSeparator()

        # 可见性切换
        visible = item_data.get('visible', True)
        visibility_action = QAction(
            "隐藏" if visible else "显示", self)
        visibility_action.triggered.connect(
            lambda: self._toggle_visibility(item_data.get('id')))
        menu.addAction(visibility_action)

        # 执行菜单
        menu.exec_(self.label_tree.mapToGlobal(pos))

    def _edit_label(self, item_id, item_data):
        """编辑标签"""
        if not item_data:
            return

        # 获取当前标签
        current_label = item_data.get("label", "")

        # 显示输入对话框
        new_label, ok = QInputDialog.getText(
            self,
            "编辑标签",
            "标签名称:",
            QLineEdit.Normal,
            current_label
        )

        # 如果用户点击了确定并且输入了新标签
        if ok and new_label and new_label != current_label:
            # 更新模型中的标签
            self.label_model.update_item(item_id, label=new_label)

    def _delete_item(self, item_id):
        """删除项目"""
        # 移除视图中的项目
        self.label_tree.remove_item(item_id)

        # 从模型中移除项目
        self.label_model.remove_item(item_id)

    def select_items(self, item_ids):
        """选中指定的多个项目"""
        for item_id in item_ids:
            self.label_tree.select_item(item_id)

    def deselect_items(self, item_ids):
        """取消选中指定的多个项目"""
        for item_id in item_ids:
            self.label_tree.deselect_item(item_id)

    def select_all(self):
        """选中所有项目"""
        for item_id in self.label_model._item_map.keys():
            self.label_tree.select_item(item_id)

    def deselect_all(self):
        """取消选中所有项目"""
        for item_id in self.label_model._item_map.keys():
            self.label_tree.deselect_item(item_id)

    def _toggle_visibility(self, item_id):
        """切换标签项的可见性"""
        if not item_id or not self.label_model:
            return

        # 获取项目数据
        item_data = self.label_model.get_item(item_id)
        if not item_data:
            return

        # 切换可见性
        current_visibility = item_data.get('visible', True)
        new_visibility = not current_visibility

        # 更新模型数据
        item_data['visible'] = new_visibility

        # 更新关联的形状对象
        shape = item_data.get('shape')
        if shape and hasattr(shape, 'setVisible'):
            shape.setVisible(new_visibility)

        # 通知模型数据已更改
        self.label_model.itemsChanged.emit()

        # 刷新视图
        self._refresh_view()

    def _on_check_state_changed(self, item_data, checked):
        """处理复选框状态变化"""
        if not item_data or not isinstance(item_data, dict):
            return

        # 更新模型数据中的可见性
        item_data['visible'] = checked

        # 更新关联的形状对象
        shape = item_data.get('shape')
        if shape and hasattr(shape, 'setVisible'):
            shape.setVisible(checked)

        # 通知模型数据已更改
        self.label_model.itemsChanged.emit()

    def _get_shape_type_icon(self, shape_type):
        """获取形状类型对应的图标"""
        if shape_type == "polygon":
            return "⬡"  # 多边形图标
        elif shape_type == "rectangle":
            return "⬜"  # 矩形图标
        elif shape_type == "circle":
            return "⚫"  # 圆形图标
        elif shape_type == "line" or shape_type == "linestrip":
            return "⎯"  # 线段图标
        elif shape_type == "point" or shape_type == "points":
            return "⋅"  # 点图标
        elif shape_type == "ai_polygon" or shape_type == "ai_mask":
            return "🤖"  # AI形状图标
        else:
            return "◆"  # 默认图标


class LabelUniqueListPanel(QWidget):
    """
    唯一标签列表面板

    显示不重复的标签项的面板，提供快速选择标签的功能
    """

    # 自定义信号
    labelSelected = pyqtSignal(str)

    def __init__(self, label_model: LabelModel, parent=None):
        """
        初始化唯一标签列表面板

        Args:
            label_model: 标签数据模型
            parent: 父窗口
        """
        super(LabelUniqueListPanel, self).__init__(parent)

        # 保存模型引用
        self.label_model = label_model

        # 初始化UI
        self._setup_ui()

        # 连接信号
        self._connect_signals()

        # 加载初始数据
        self._load_data()

    def _setup_ui(self):
        """设置UI组件"""
        # 主布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # 头部区域
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(10)

        # 标题
        self.title_label = QLabel("多边形标签")
        font = self.title_label.font()
        font.setPointSize(12)
        font.setBold(True)
        self.title_label.setFont(font)
        self.title_label.setStyleSheet("color: #37474f;")
        header_layout.addWidget(self.title_label)

        # 计数小部件
        self.count_widget = LabelCountWidget()
        header_layout.addWidget(self.count_widget)

        # 右侧间隔
        header_layout.addStretch()

        # 添加头部到主布局
        layout.addWidget(header_widget)

        # 搜索框
        self.search_box = SearchBox()
        layout.addWidget(self.search_box)

        # 标签树视图
        self.label_tree = UniqueLabelTreeView()
        layout.addWidget(self.label_tree)

        # 设置大小策略
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.setMinimumWidth(250)

        # 样式
        self.setStyleSheet("""
            LabelUniqueListPanel {
                background-color: white;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
            }
        """)

    def _connect_signals(self):
        """连接信号"""
        # 标签模型信号
        self.label_model.categoriesChanged.connect(self._refresh_view)
        self.label_model.labelsChanged.connect(self._refresh_view)
        self.label_model.itemsChanged.connect(self._refresh_view)

        # 搜索框信号
        self.search_box.textChanged.connect(self._filter_labels)
        self.search_box.searchCleared.connect(self._refresh_view)

        # 树视图信号 - 修复信号连接问题
        # 将 itemSelected 信号连接到一个中间处理函数，确保类型转换正确
        self.label_tree.itemSelected.connect(self._on_label_selected)
        self.label_tree.itemDoubleClicked.connect(
            self._on_label_double_clicked)

    # 添加中间处理函数
    def _on_label_selected(self, label_obj):
        """处理标签选中事件，确保转换为字符串"""
        # 确保传递字符串类型
        if label_obj:
            self.labelSelected.emit(str(label_obj))

    def _on_label_double_clicked(self, label_obj):
        """处理标签双击事件，确保转换为字符串"""
        # 确保传递字符串类型
        if label_obj:
            self.labelSelected.emit(str(label_obj))

    def _load_data(self):
        """加载数据到树视图"""
        # 清空当前视图
        self.label_tree.clear()

        # 获取所有分类
        categories = self.label_model.get_categories()

        for category in categories:
            # 获取分类下的所有标签
            labels = self.label_model.get_labels_by_category(category.name)

            # 添加每个标签到视图
            for label in labels:
                # 获取标签的项目数
                items = self.label_model.get_items_by_label(label)

                # 创建显示文本
                display_text = self._create_display_text(label, len(items))

                # 添加到视图
                self.label_tree.add_label(
                    label,
                    display_text,
                    category.name,
                    category.display_name
                )

        # 更新计数
        self._update_counts()

    def _create_display_text(self, label, count):
        """创建标签的显示文本"""
        # HTML格式的显示文本
        label_text = html.escape(label)

        # 组合显示文本
        display_text = f"<span style='color:#0d47a1;'>{label_text}</span> <span style='color:#78909c; font-size:10px;'>({count})</span>"

        return display_text

    def _refresh_view(self):
        """刷新视图"""
        # 重新加载数据
        self._load_data()

    def _filter_labels(self, text):
        """
        根据搜索文本过滤标签

        当文本为空时，显示所有标签；否则只显示匹配的标签
        """
        if not text:
            # 如果搜索文本为空，刷新视图显示全部
            self._refresh_view()
            return

        # 清空视图
        self.label_tree.clear()

        # 搜索文本（不区分大小写）
        text = text.lower()

        # 遍历所有分类
        for category in self.label_model.get_categories():
            # 获取分类下的所有标签
            labels = self.label_model.get_labels_by_category(category.name)

            # 过滤匹配的标签
            filtered_labels = [
                label for label in labels if text in label.lower()]

            # 添加每个匹配的标签到视图
            for label in filtered_labels:
                # 获取标签的项目数
                items = self.label_model.get_items_by_label(label)

                # 创建显示文本
                display_text = self._create_display_text(label, len(items))

                # 添加到视图
                self.label_tree.add_label(
                    label,
                    display_text,
                    category.name,
                    category.display_name
                )

        # 更新计数
        filtered_count = sum(1 for label in self.label_model.get_labels()
                             if text in label.lower())
        self.count_widget.update_count(filtered_count)

    def _update_counts(self):
        """更新标签计数"""
        # 获取所有标签
        all_labels = self.label_model.get_labels()

        # 更新标签计数
        self.count_widget.update_count(len(all_labels))


class LabelPanelController:
    """
    标签面板控制器

    管理标签列表和唯一标签列表面板，协调它们之间的交互
    """

    def __init__(self, label_model: LabelModel):
        """
        初始化标签面板控制器

        Args:
            label_model: 标签数据模型
        """
        self.label_model = label_model

        # 创建面板
        self.label_list_panel = LabelListPanel(label_model)
        self.unique_label_panel = LabelUniqueListPanel(label_model)

        # 连接信号
        self._connect_signals()

    def _connect_signals(self):
        """连接信号"""
        # 唯一标签面板 -> 标签列表面板
        self.unique_label_panel.labelSelected.connect(
            self._on_unique_label_selected)

    def _on_unique_label_selected(self, label):
        """当唯一标签被选中时"""
        if not label or not isinstance(label, str):
            return

        # 获取所有具有该标签的项目ID
        items = self.label_model.get_items_by_label(label)

        # 先取消所有选择
        self.label_list_panel.deselect_all()

        # 然后选择匹配的项目
        self.label_list_panel.select_items(items)

    def get_label_list_panel(self) -> LabelListPanel:
        """获取标签列表面板"""
        return self.label_list_panel

    def get_unique_label_panel(self) -> LabelUniqueListPanel:
        """获取唯一标签列表面板"""
        return self.unique_label_panel
