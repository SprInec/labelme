"""
标签窗口模块 - 集成所有标签管理功能的窗口组件

该模块实现了完整的标签管理窗口，集成标签面板和控制器，
提供与主应用程序交互的接口。
"""

from typing import Dict, List, Optional, Set, Any, Callable, Union

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QToolBar, QAction, QDockWidget, QMenu, QShortcut, QMessageBox
)

from .label_model import LabelModel
from .label_panel import LabelPanelController, LabelListPanel, LabelUniqueListPanel


class LabelWindowToolBar(QToolBar):
    """
    标签窗口工具栏

    提供标签管理操作的工具栏
    """

    def __init__(self, parent=None):
        """初始化工具栏"""
        super(LabelWindowToolBar, self).__init__(parent)
        self.setIconSize(QSize(20, 20))
        self.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.setMovable(False)
        self.setStyleSheet("""
            QToolBar {
                border: none;
                background-color: #f5f5f5;
                spacing: 5px;
                padding: 5px;
            }
            QToolButton {
                border: none;
                background-color: transparent;
                border-radius: 4px;
                padding: 5px;
                color: #37474f;
            }
            QToolButton:hover {
                background-color: #e3f2fd;
            }
            QToolButton:pressed {
                background-color: #bbdefb;
            }
        """)

        # 创建操作 - 改用文本而不是可能不存在的图标资源
        self.action_refresh = QAction(
            "刷新",
            self
        )
        self.action_select_all = QAction(
            "全选",
            self
        )
        self.action_deselect_all = QAction(
            "取消全选",
            self
        )
        self.action_filter = QAction(
            "过滤",
            self
        )
        self.action_settings = QAction(
            "设置",
            self
        )

        # 添加到工具栏
        self.addAction(self.action_refresh)
        self.addSeparator()
        self.addAction(self.action_select_all)
        self.addAction(self.action_deselect_all)
        self.addSeparator()
        self.addAction(self.action_filter)
        self.addSeparator()
        self.addAction(self.action_settings)

    def setup_connections(self, window):
        """设置工具栏操作的连接"""
        self.action_refresh.triggered.connect(window.refresh_views)
        self.action_select_all.triggered.connect(window.select_all_items)
        self.action_deselect_all.triggered.connect(window.deselect_all_items)
        self.action_filter.triggered.connect(window.show_filter_menu)
        self.action_settings.triggered.connect(window.show_settings_dialog)


class LabelWindow(QMainWindow):
    """
    标签管理窗口

    集成所有标签管理功能的窗口组件，提供与主应用程序交互的接口
    """

    # 自定义信号
    itemSelected = pyqtSignal(object)
    itemDeselected = pyqtSignal(object)
    itemDoubleClicked = pyqtSignal(object)
    itemsChanged = pyqtSignal()

    def __init__(self, parent=None):
        """初始化标签窗口"""
        super(LabelWindow, self).__init__(parent)

        # 窗口设置
        self.setWindowTitle("标签管理")
        self.resize(800, 600)

        # 创建数据模型
        self.label_model = LabelModel()

        # 创建控制器
        self.controller = LabelPanelController(self.label_model)

        # 获取面板
        self.label_list_panel = self.controller.get_label_list_panel()
        self.unique_label_panel = self.controller.get_unique_label_panel()

        # 初始化UI
        self._setup_ui()

        # 连接信号
        self._connect_signals()

        # 设置快捷键
        self._setup_shortcuts()

    def _setup_ui(self):
        """设置UI组件"""
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 创建工具栏
        self.toolbar = LabelWindowToolBar(self)
        self.toolbar.setup_connections(self)
        main_layout.addWidget(self.toolbar)

        # 创建分割器
        self.splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.splitter)

        # 将面板添加到分割器
        self.splitter.addWidget(self.label_list_panel)
        self.splitter.addWidget(self.unique_label_panel)

        # 设置分割比例
        self.splitter.setSizes(
            [int(self.width() * 0.6), int(self.width() * 0.4)])

        # 样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: white;
            }
            QSplitter::handle {
                background-color: #e0e0e0;
                width: 1px;
            }
            QSplitter::handle:hover {
                background-color: #2979ff;
            }
        """)

    def _connect_signals(self):
        """连接信号"""
        # 标签列表面板信号
        self.label_list_panel.itemSelected.connect(self.itemSelected)
        self.label_list_panel.itemDeselected.connect(self.itemDeselected)
        self.label_list_panel.itemDoubleClicked.connect(self.itemDoubleClicked)

        # 标签模型信号
        self.label_model.itemsChanged.connect(self.itemsChanged)

    def _setup_shortcuts(self):
        """设置快捷键"""
        # 刷新 - F5
        refresh_shortcut = QShortcut(QKeySequence("F5"), self)
        refresh_shortcut.activated.connect(self.refresh_views)

        # 全选 - Ctrl+A
        select_all_shortcut = QShortcut(QKeySequence("Ctrl+A"), self)
        select_all_shortcut.activated.connect(self.select_all_items)

        # 取消全选 - Esc
        deselect_all_shortcut = QShortcut(QKeySequence("Esc"), self)
        deselect_all_shortcut.activated.connect(self.deselect_all_items)

    def add_item(self, item_id, label, shape_type, data=None):
        """
        添加项目到标签管理器

        Args:
            item_id: 项目唯一ID
            label: 标签文本
            shape_type: 形状类型
            data: 额外的项目数据

        Returns:
            bool: 是否成功添加
        """
        return self.label_model.add_item(item_id, label, shape_type, data)

    def remove_item(self, item_id):
        """
        从标签管理器中移除项目

        Args:
            item_id: 项目ID

        Returns:
            bool: 是否成功移除
        """
        return self.label_model.remove_item(item_id)

    def update_item(self, item_id, label=None, shape_type=None, data=None):
        """
        更新项目数据

        Args:
            item_id: 项目ID
            label: 新标签（如需更新）
            shape_type: 新形状类型（如需更新）
            data: 新数据（如需更新）

        Returns:
            bool: 是否成功更新
        """
        return self.label_model.update_item(item_id, label, shape_type, data)

    def get_item(self, item_id):
        """
        获取项目数据

        Args:
            item_id: 项目ID

        Returns:
            dict: 项目数据，不存在则返回None
        """
        return self.label_model.get_item(item_id)

    def get_items_by_label(self, label):
        """
        获取指定标签的所有项目ID

        Args:
            label: 标签文本

        Returns:
            List[int]: 项目ID列表
        """
        return self.label_model.get_items_by_label(label)

    def get_items_by_category(self, category_name):
        """
        获取指定分类的所有项目ID

        Args:
            category_name: 分类名称

        Returns:
            List[int]: 项目ID列表
        """
        return self.label_model.get_items_by_category(category_name)

    def get_all_labels(self):
        """
        获取所有标签

        Returns:
            Set[str]: 标签集合
        """
        return self.label_model.get_labels()

    def clear(self):
        """清空所有数据"""
        self.label_model.clear()

    def refresh_views(self):
        """刷新所有视图"""
        # 刷新标签列表面板
        self.label_list_panel._refresh_view()

        # 刷新唯一标签面板
        self.unique_label_panel._refresh_view()

    def select_all_items(self):
        """选中所有项目"""
        self.label_list_panel.select_all()

    def deselect_all_items(self):
        """取消选中所有项目"""
        self.label_list_panel.deselect_all()

    def show_filter_menu(self):
        """显示过滤菜单"""
        filter_menu = QMenu(self)

        # 创建过滤选项
        action_filter_polygon = QAction("仅显示多边形", self)
        action_filter_rectangle = QAction("仅显示矩形", self)
        action_filter_circle = QAction("仅显示圆形", self)
        action_filter_line = QAction("仅显示线段", self)
        action_filter_point = QAction("仅显示点", self)
        action_filter_clear = QAction("清除过滤", self)

        # 添加到菜单
        filter_menu.addAction(action_filter_polygon)
        filter_menu.addAction(action_filter_rectangle)
        filter_menu.addAction(action_filter_circle)
        filter_menu.addAction(action_filter_line)
        filter_menu.addAction(action_filter_point)
        filter_menu.addSeparator()
        filter_menu.addAction(action_filter_clear)

        # 连接操作信号
        action_filter_polygon.triggered.connect(
            lambda: self._apply_filter("polygon"))
        action_filter_rectangle.triggered.connect(
            lambda: self._apply_filter("rectangle"))
        action_filter_circle.triggered.connect(
            lambda: self._apply_filter("circle"))
        action_filter_line.triggered.connect(
            lambda: self._apply_filter("line"))
        action_filter_point.triggered.connect(
            lambda: self._apply_filter("point"))
        action_filter_clear.triggered.connect(
            self._clear_filter)

        # 显示菜单
        filter_menu.exec_(QtGui.QCursor.pos())

    def _apply_filter(self, shape_type):
        """
        应用形状类型过滤

        显示指定形状类型的项目
        """
        # 清空列表视图
        self.label_list_panel.label_tree.clear()

        # 获取所有项目数据
        for item_id, item_data in self.label_model._item_map.items():
            # 检查形状类型是否匹配
            if item_data.get("shape_type") == shape_type:
                label = item_data.get("label", "")

                # 获取分类
                category_name = self.label_model._determine_category(item_data)

                # 创建显示文本
                display_text = self.label_list_panel._create_display_text(
                    label, shape_type)

                # 添加到视图
                self.label_list_panel.label_tree.add_item(
                    item_id,
                    display_text,
                    item_data,
                    category_name
                )

    def _clear_filter(self):
        """清除过滤，显示所有项目"""
        # 刷新视图
        self.refresh_views()

    def show_settings_dialog(self):
        """显示设置对话框"""
        # TODO: 实现设置对话框
        QMessageBox.information(
            self,
            "设置",
            "标签管理设置将在后续版本中实现"
        )

    def closeEvent(self, event):
        """处理窗口关闭事件"""
        # 保存状态等操作
        super(LabelWindow, self).closeEvent(event)
