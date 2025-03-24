from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

from labelme.config import get_config
from labelme.config.config import save_config


class ShortcutsDialog(QtWidgets.QDialog):
    """快捷键设置对话框"""

    def __init__(self, parent=None):
        super(ShortcutsDialog, self).__init__(parent)
        self.parent = parent

        # 优先使用父窗口的配置（如果可用）
        if parent and hasattr(parent, '_config') and 'shortcuts' in parent._config:
            # 使用父窗口的最新配置
            self.config = parent._config
            self.shortcuts = self.config.get("shortcuts", {})
        else:
            # 如果父窗口配置不可用，则读取配置文件
            self.config = get_config()
            self.shortcuts = self.config.get("shortcuts", {})

        self.modified_shortcuts = self.shortcuts.copy()

        self.setWindowTitle(self.tr("快捷键设置"))
        self.setWindowFlags(
            self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint
        )
        self.setMinimumWidth(800)
        self.setMinimumHeight(1000)

        # 获取当前主题
        app = QtWidgets.QApplication.instance()
        self.is_dark = app.property("currentTheme") == "dark"

        # 加载样式表
        self.setThemeStyleSheet()

        self.initUI()

    def setThemeStyleSheet(self):
        """根据主题设置样式表"""
        if self.is_dark:
            self.setStyleSheet("""
                QDialog {
                    background-color: #252526;
                }
                QLabel {
                    font-size: 28px;
                    color: #ffffff;
                }
                QTableWidget {
                    background-color: #2d2d30;
                    border: 1px solid #3f3f46;
                    border-radius: 6px;
                    alternate-background-color: #333337;
                    gridline-color: #3f3f46;
                }
                QTableWidget::item {
                    padding: 10px;
                    border-bottom: 1px solid #3f3f46;
                    color: #ffffff;
                }
                QTableWidget::item:selected {
                    background-color: #37373d;
                    color: #ffffff;
                }
                QHeaderView::section {
                    background-color: #2d2d30;
                    padding: 12px;
                    border: none;
                    border-bottom: 1px solid #3f3f46;
                    font-weight: bold;
                    color: #ffffff;
                }
                QPushButton {
                    background-color: #2d2d30;
                    border: 1px solid #3f3f46;
                    border-radius: 6px;
                    padding: 8px 16px;
                    color: #ffffff;
                    font-weight: bold;
                    min-width: 100px;
                    min-height: 36px;
                }
                QPushButton:hover {
                    background-color: #37373d;
                    border-color: #4f4f56;
                }
                QPushButton:pressed {
                    background-color: #3f3f46;
                }
                QLineEdit {
                    border: 1px solid #3f3f46;
                    border-radius: 6px;
                    padding: 8px;
                    background-color: #2d2d30;
                    color: #ffffff;
                }
                QLineEdit:focus {
                    border-color: #007acc;
                }
                QScrollBar:vertical {
                    background-color: #252526;
                    width: 8px;
                    margin: 0px;
                }
                QScrollBar::handle:vertical {
                    background-color: #5a5a5c;
                    min-height: 30px;
                    border-radius: 4px;
                }
                QScrollBar::handle:vertical:hover {
                    background-color: #777779;
                }
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
                QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                    background: none;
                    height: 0px;
                }
            """)
        else:
            self.setStyleSheet("""
                QDialog {
                    background-color: #f5f5f5;
                }
                QLabel {
                    font-size: 28px;
                    color: #333333;
                }
                QTableWidget {
                    background-color: white;
                    border: 1px solid #e0e0e0;
                    border-radius: 6px;
                    alternate-background-color: #f8f8f8;
                    gridline-color: #e0e0e0;
                }
                QTableWidget::item {
                    padding: 10px;
                    border-bottom: 1px solid #f0f0f0;
                }
                QTableWidget::item:selected {
                    background-color: #e6f3ff;
                    color: #333333;
                }
                QHeaderView::section {
                    background-color: #f0f0f0;
                    padding: 12px;
                    border: none;
                    border-bottom: 1px solid #e0e0e0;
                    font-weight: bold;
                    color: #555555;
                }
                QPushButton {
                    background-color: #ffffff;
                    border: 1px solid #e0e0e0;
                    border-radius: 6px;
                    padding: 8px 16px;
                    color: #333333;
                    font-weight: bold;
                    min-width: 100px;
                    min-height: 36px;
                }
                QPushButton:hover {
                    background-color: #f0f0f0;
                    border-color: #d0d0d0;
                }
                QPushButton:pressed {
                    background-color: #e0e0e0;
                }
                QLineEdit {
                    border: 1px solid #e0e0e0;
                    border-radius: 6px;
                    padding: 8px;
                    background-color: white;
                }
                QLineEdit:focus {
                    border-color: #66afe9;
                }
                QScrollBar:vertical {
                    background-color: #f0f0f0;
                    width: 8px;
                    margin: 0px;
                }
                QScrollBar::handle:vertical {
                    background-color: #c0c0c0;
                    min-height: 30px;
                    border-radius: 4px;
                }
                QScrollBar::handle:vertical:hover {
                    background-color: #a0a0a0;
                }
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
                QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                    background: none;
                    height: 0px;
                }
            """)

    def showEvent(self, event):
        """窗口显示时更新主题"""
        super(ShortcutsDialog, self).showEvent(event)
        # 获取当前主题
        app = QtWidgets.QApplication.instance()
        current_theme = app.property("currentTheme")
        if current_theme != self.is_dark:
            self.is_dark = current_theme == "dark"
            self.setThemeStyleSheet()

    def initUI(self):
        """初始化UI"""
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)

        # 创建标题和说明
        title_label = QtWidgets.QLabel(self.tr("快捷键设置"))
        title_label.setStyleSheet(
            "font-size: 30px; font-weight: bold; margin-bottom: 10px;")

        desc_label = QtWidgets.QLabel(self.tr("您可以在下表中查看和自定义软件的快捷键："))
        desc_label.setStyleSheet(
            "font-size: 24px; color: #555555; margin-bottom: 15px;")

        # 搜索框
        search_layout = QtWidgets.QHBoxLayout()
        search_label = QtWidgets.QLabel(self.tr("搜索:"))
        self.search_edit = QtWidgets.QLineEdit()
        self.search_edit.setPlaceholderText(self.tr("输入关键词过滤功能或快捷键"))
        self.search_edit.textChanged.connect(self.filterTable)
        self.search_edit.setClearButtonEnabled(True)

        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_edit)

        # 创建表格
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels([self.tr("功能"), self.tr("快捷键")])
        self.table.setColumnWidth(0, 450)
        self.table.setColumnWidth(1, 250)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setDefaultSectionSize(42)
        self.table.setShowGrid(False)
        # 启用表格排序
        self.table.setSortingEnabled(True)
        # 双击编辑快捷键
        self.table.doubleClicked.connect(self.editShortcut)
        # 设置右键菜单
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.showContextMenu)

        # 填充表格
        self.populateTable()

        # 添加按钮
        button_layout = QtWidgets.QHBoxLayout()

        # 创建按钮容器小部件，用于添加阴影效果
        button_container = QtWidgets.QWidget()
        button_layout_inner = QtWidgets.QHBoxLayout(button_container)
        button_layout_inner.setContentsMargins(0, 0, 0, 0)

        self.edit_button = QtWidgets.QPushButton(self.tr("编辑"))
        self.edit_button.clicked.connect(self.editShortcut)
        self.edit_button.setIcon(QtGui.QIcon.fromTheme("edit"))

        self.reset_button = QtWidgets.QPushButton(self.tr("重置"))
        self.reset_button.clicked.connect(self.resetShortcut)
        self.reset_button.setIcon(QtGui.QIcon.fromTheme("edit-undo"))

        self.reset_all_button = QtWidgets.QPushButton(self.tr("全部重置"))
        self.reset_all_button.clicked.connect(self.resetAllShortcuts)
        self.reset_all_button.setIcon(QtGui.QIcon.fromTheme("edit-clear"))

        button_box = QtWidgets.QDialogButtonBox()
        ok_button = button_box.addButton(QtWidgets.QDialogButtonBox.Ok)
        cancel_button = button_box.addButton(QtWidgets.QDialogButtonBox.Cancel)
        ok_button.setText(self.tr("确定"))
        cancel_button.setText(self.tr("取消"))
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        button_layout_inner.addWidget(self.edit_button)
        button_layout_inner.addWidget(self.reset_button)
        button_layout_inner.addWidget(self.reset_all_button)
        button_layout_inner.addStretch()
        button_layout_inner.addWidget(button_box)

        # 添加到布局
        layout.addWidget(title_label)
        layout.addWidget(desc_label)
        layout.addLayout(search_layout)
        layout.addWidget(self.table)
        layout.addWidget(button_container)
        self.setLayout(layout)

    def filterTable(self):
        """根据搜索框内容过滤表格"""
        search_text = self.search_edit.text().lower()
        for row in range(self.table.rowCount()):
            function_name = self.table.item(row, 0).text().lower()
            shortcut_text = self.table.item(row, 1).text().lower()
            if search_text in function_name or search_text in shortcut_text:
                self.table.setRowHidden(row, False)
            else:
                self.table.setRowHidden(row, True)

    def showContextMenu(self, pos):
        """显示上下文菜单"""
        global_pos = self.table.mapToGlobal(pos)
        row = self.table.rowAt(pos.y())

        if row < 0:
            return

        self.table.selectRow(row)

        menu = QtWidgets.QMenu(self)
        edit_action = menu.addAction(self.tr("编辑快捷键"))
        reset_action = menu.addAction(self.tr("重置快捷键"))

        action = menu.exec_(global_pos)

        if action == edit_action:
            self.editShortcut()
        elif action == reset_action:
            self.resetShortcut()

    def populateTable(self):
        """填充表格数据"""
        self.table.setSortingEnabled(False)  # 暂时禁用排序
        self.table.setRowCount(0)

        # 功能名称映射
        function_names = {
            "close": self.tr("关闭"),
            "open": self.tr("打开"),
            "open_dir": self.tr("打开目录"),
            "quit": self.tr("退出"),
            "save": self.tr("保存"),
            "save_as": self.tr("另存为"),
            "save_to": self.tr("更改输出路径"),
            "delete_file": self.tr("删除文件"),
            "open_next": self.tr("下一张图片"),
            "open_prev": self.tr("上一张图片"),
            "zoom_in": self.tr("放大"),
            "zoom_out": self.tr("缩小"),
            "zoom_to_original": self.tr("原始大小"),
            "fit_window": self.tr("适应窗口"),
            "fit_width": self.tr("适应宽度"),
            "create_polygon": self.tr("创建多边形"),
            "create_rectangle": self.tr("创建矩形"),
            "create_circle": self.tr("创建圆形"),
            "create_line": self.tr("创建线条"),
            "create_point": self.tr("创建点"),
            "create_linestrip": self.tr("创建折线"),
            "edit_polygon": self.tr("编辑多边形"),
            "delete_polygon": self.tr("删除多边形"),
            "duplicate_polygon": self.tr("复制多边形"),
            "copy_polygon": self.tr("复制"),
            "paste_polygon": self.tr("粘贴"),
            "undo": self.tr("撤销"),
            "undo_last_point": self.tr("撤销上一个点"),
            "add_point_to_edge": self.tr("添加点到边缘"),
            "edit_label": self.tr("编辑标签"),
            "toggle_keep_prev_mode": self.tr("保持上一个模式"),
            "remove_selected_point": self.tr("删除选中的点"),
            "show_all_polygons": self.tr("显示所有多边形"),
            "hide_all_polygons": self.tr("隐藏所有多边形"),
            "toggle_all_polygons": self.tr("切换所有多边形"),
        }

        # 将快捷键分类
        categories = {
            self.tr("文件操作"): ["open", "open_dir", "save", "save_as", "save_to", "close", "quit", "delete_file"],
            self.tr("导航"): ["open_next", "open_prev"],
            self.tr("缩放与视图"): ["zoom_in", "zoom_out", "zoom_to_original", "fit_window", "fit_width",
                               "show_all_polygons", "hide_all_polygons", "toggle_all_polygons"],
            self.tr("创建与编辑"): ["create_polygon", "create_rectangle", "create_circle", "create_line",
                               "create_point", "create_linestrip", "edit_polygon", "delete_polygon",
                               "duplicate_polygon", "copy_polygon", "paste_polygon", "edit_label",
                               "undo", "undo_last_point", "add_point_to_edge", "toggle_keep_prev_mode",
                               "remove_selected_point"]
        }

        row = 0
        # 按分类添加项目
        for category, keys in categories.items():
            for key in keys:
                if key in self.modified_shortcuts:
                    self.table.insertRow(row)

                    # 功能名称
                    display_name = function_names.get(key, key)
                    name_item = QtWidgets.QTableWidgetItem(display_name)
                    self.table.setItem(row, 0, name_item)

                    # 快捷键
                    shortcut_text = self.formatShortcutText(
                        self.modified_shortcuts[key])
                    shortcut_item = QtWidgets.QTableWidgetItem(shortcut_text)
                    self.table.setItem(row, 1, shortcut_item)

                    # 存储原始键和分类
                    name_item.setData(Qt.UserRole, key)
                    name_item.setData(Qt.UserRole + 1, category)

                    row += 1

        # 添加未分类的项目
        for key, value in sorted(self.modified_shortcuts.items()):
            if not any(key in cat_keys for cat_keys in categories.values()):
                self.table.insertRow(row)

                # 功能名称
                name_item = QtWidgets.QTableWidgetItem(
                    function_names.get(key, key))
                self.table.setItem(row, 0, name_item)

                # 快捷键
                shortcut_text = self.formatShortcutText(value)
                shortcut_item = QtWidgets.QTableWidgetItem(shortcut_text)
                self.table.setItem(row, 1, shortcut_item)

                # 存储原始键
                name_item.setData(Qt.UserRole, key)
                name_item.setData(Qt.UserRole + 1, self.tr("其他"))

                row += 1

        self.table.setSortingEnabled(True)  # 重新启用排序

    def formatShortcutText(self, shortcut):
        """格式化快捷键文本"""
        if shortcut is None:
            return self.tr("无")
        elif isinstance(shortcut, list):
            return ", ".join([str(s) for s in shortcut])
        else:
            return str(shortcut)

    def editShortcut(self):
        """编辑快捷键"""
        current_row = self.table.currentRow()
        if current_row < 0:
            QtWidgets.QMessageBox.information(
                self, self.tr("提示"), self.tr("请先选择一个功能"))
            return

        key = self.table.item(current_row, 0).data(Qt.UserRole)
        current_shortcut = self.modified_shortcuts.get(key)
        display_name = self.table.item(current_row, 0).text()

        dialog = ShortcutEditDialog(self, display_name, current_shortcut)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            # 获取新的快捷键，保持原始格式（列表或普通）
            new_shortcut = dialog.current_shortcut

            # 检查快捷键冲突
            conflict_found = False
            if new_shortcut is not None:
                # 确定要检查的快捷键值
                check_shortcut = new_shortcut
                if isinstance(new_shortcut, list):
                    # 如果是列表，只检查第一个元素
                    check_shortcut = new_shortcut[0] if new_shortcut else None

                for k, v in self.modified_shortcuts.items():
                    # 跳过当前正在编辑的项
                    if k == key:
                        continue

                    # 确定要比较的值
                    compare_value = v
                    if isinstance(v, list):
                        # 如果是列表，只比较第一个元素
                        compare_value = v[0] if v else None

                    if compare_value == check_shortcut and check_shortcut is not None:
                        conflict_key = k
                        # 查找显示名称
                        conflict_display_name = conflict_key
                        for row in range(self.table.rowCount()):
                            if self.table.item(row, 0).data(Qt.UserRole) == conflict_key:
                                conflict_display_name = self.table.item(
                                    row, 0).text()
                                break

                        reply = QtWidgets.QMessageBox.question(
                            self,
                            self.tr("快捷键冲突"),
                            self.tr("快捷键 '{0}' 已被 '{1}' 使用。\n\n是否仍要分配此快捷键？").format(
                                check_shortcut, conflict_display_name),
                            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                            QtWidgets.QMessageBox.No
                        )

                        if reply == QtWidgets.QMessageBox.No:
                            conflict_found = True
                            break
                        else:
                            # 如果用户确认，则移除之前的快捷键分配
                            self.modified_shortcuts[conflict_key] = None

            if not conflict_found:
                # 更新修改后的快捷键
                self.modified_shortcuts[key] = new_shortcut

                # 更新表格
                self.populateTable()

    def resetShortcut(self):
        """重置当前选中的快捷键"""
        current_row = self.table.currentRow()
        if current_row < 0:
            QtWidgets.QMessageBox.information(
                self, self.tr("提示"), self.tr("请先选择一个功能"))
            return

        key = self.table.item(current_row, 0).data(Qt.UserRole)
        original_shortcut = self.shortcuts.get(key)

        # 恢复原始快捷键
        self.modified_shortcuts[key] = original_shortcut

        # 更新表格
        self.populateTable()

    def resetAllShortcuts(self):
        """重置所有快捷键"""
        reply = QtWidgets.QMessageBox.question(
            self,
            self.tr("确认重置"),
            self.tr("确定要重置所有快捷键吗？"),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )

        if reply == QtWidgets.QMessageBox.Yes:
            # 恢复所有原始快捷键
            self.modified_shortcuts = self.shortcuts.copy()

            # 更新表格
            self.populateTable()

    def accept(self):
        """确认修改"""
        # 更新配置
        self.config["shortcuts"] = self.modified_shortcuts

        # 如果父窗口存在且有applyCustomShortcuts方法，优先调用它
        # 它会负责保存配置到文件并应用快捷键
        if self.parent and hasattr(self.parent, 'applyCustomShortcuts'):
            self.parent.applyCustomShortcuts(self.modified_shortcuts)

            # 提示用户
            QtWidgets.QMessageBox.information(
                self,
                self.tr("提示"),
                self.tr("快捷键设置已保存并应用。")
            )
        else:
            # 如果无法通过父窗口保存，则尝试直接保存配置
            save_config(self.config)

            # 尝试立即应用快捷键设置
            self.applyShortcuts()

            # 提示用户
            QtWidgets.QMessageBox.information(
                self,
                self.tr("提示"),
                self.tr("快捷键设置已保存并应用，部分更改可能需要重启应用后生效。")
            )

        super(ShortcutsDialog, self).accept()

    def applyShortcuts(self):
        """立即应用快捷键设置"""
        if not self.parent or not hasattr(self.parent, 'actions'):
            return

        try:
            # 更新快捷键
            for key, shortcut in self.modified_shortcuts.items():
                self.updateActionShortcut(key, shortcut)

            # 显示状态栏消息
            if hasattr(self.parent, 'status'):
                self.parent.status(self.tr("快捷键已更新"), 5000)
        except Exception as e:
            print(f"应用快捷键设置时出错: {e}")

    def updateActionShortcut(self, key, shortcut):
        """更新指定快捷键对应的动作"""
        if not self.parent or not hasattr(self.parent, 'actions'):
            return False

        try:
            # 创建快捷键序列
            if shortcut is None:
                shortcut_seq = QtGui.QKeySequence()
            elif isinstance(shortcut, list):
                # 如果是列表，只使用第一个快捷键
                if shortcut and len(shortcut) > 0:
                    shortcut_seq = QtGui.QKeySequence(str(shortcut[0]))
                else:
                    shortcut_seq = QtGui.QKeySequence()
            else:
                shortcut_seq = QtGui.QKeySequence(str(shortcut))

            # 获取动作集合
            actions = self.parent.actions

            # 常见的动作映射
            action_mapping = {
                "close": getattr(actions, "close", None),
                "open": getattr(actions, "open", None),
                "save": getattr(actions, "save", None),
                "save_as": getattr(actions, "saveAs", None),
                "quit": getattr(actions, "quit", None),
                "delete_file": getattr(actions, "deleteFile", None),
                "open_next": getattr(actions, "openNextImg", None),
                "open_prev": getattr(actions, "openPrevImg", None),
                "zoom_in": getattr(actions, "zoomIn", None),
                "zoom_out": getattr(actions, "zoomOut", None),
                "zoom_to_original": getattr(actions, "zoomOrg", None),
                "fit_window": getattr(actions, "fitWindow", None),
                "fit_width": getattr(actions, "fitWidth", None),
                "create_polygon": getattr(actions, "createMode", None),
                "create_rectangle": getattr(actions, "createRectangleMode", None),
                "create_circle": getattr(actions, "createCircleMode", None),
                "create_line": getattr(actions, "createLineMode", None),
                "create_point": getattr(actions, "createPointMode", None),
                "create_linestrip": getattr(actions, "createLineStripMode", None),
                "edit_polygon": getattr(actions, "editMode", None),
                "delete_polygon": getattr(actions, "delete", None),
                "duplicate_polygon": getattr(actions, "duplicate", None),
                "copy_polygon": getattr(actions, "copy", None),
                "paste_polygon": getattr(actions, "paste", None),
                "undo": getattr(actions, "undo", None),
                "undo_last_point": getattr(actions, "undoLastPoint", None),
                "edit_label": getattr(actions, "edit", None),
                "toggle_keep_prev_mode": getattr(actions, "toggleKeepPrevMode", None),
                "remove_selected_point": getattr(actions, "removePoint", None),
            }

            # 如果在映射表中找到了对应的动作，则设置快捷键
            if key in action_mapping and action_mapping[key]:
                action_mapping[key].setShortcut(shortcut_seq)
                return True

            return False
        except Exception as e:
            print(f"更新动作 {key} 的快捷键时出错: {e}")
            return False


class ShortcutEditDialog(QtWidgets.QDialog):
    """快捷键编辑对话框"""

    def __init__(self, parent=None, display_name=None, current_shortcut=None):
        super(ShortcutEditDialog, self).__init__(parent)
        self.display_name = display_name
        self.current_shortcut = current_shortcut
        # 记录原始快捷键是否为列表格式
        self.is_list_format = isinstance(current_shortcut, list)

        self.setWindowTitle(self.tr("编辑快捷键"))
        self.setFixedSize(500, 500)

        # 获取当前主题
        app = QtWidgets.QApplication.instance()
        self.is_dark = app.property("currentTheme") == "dark"

        # 设置样式
        self.setThemeStyleSheet()

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)

        # 显示功能名称
        title_label = QtWidgets.QLabel(self.tr("为以下功能设置快捷键"))
        title_label.setObjectName("titleLabel")

        function_label = QtWidgets.QLabel(display_name)
        function_label.setStyleSheet(
            "font-weight: bold; font-size: 14pt; margin: 5px 0;")
        function_label.setAlignment(QtCore.Qt.AlignCenter)

        # 快捷键编辑控件
        self.keySequenceEdit = QtWidgets.QKeySequenceEdit()
        if current_shortcut:
            if isinstance(current_shortcut, list):
                # 如果是列表，只使用第一个元素
                if current_shortcut and len(current_shortcut) > 0:
                    self.keySequenceEdit.setKeySequence(
                        QtGui.QKeySequence(str(current_shortcut[0])))
            else:
                self.keySequenceEdit.setKeySequence(
                    QtGui.QKeySequence(str(current_shortcut)))
        self.keySequenceEdit.setFocus()

        # 提示信息
        tip_label = QtWidgets.QLabel(self.tr("请按下新的快捷键组合，或点击清除按钮移除快捷键"))
        tip_label.setWordWrap(True)
        tip_label.setAlignment(QtCore.Qt.AlignLeft)
        tip_label.setStyleSheet("color: #666666; font-size: 24px;")

        # 按钮
        button_layout = QtWidgets.QHBoxLayout()

        clear_button = QtWidgets.QPushButton(self.tr("清除"))
        clear_button.setIcon(QtGui.QIcon.fromTheme("edit-clear"))
        clear_button.clicked.connect(self.keySequenceEdit.clear)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_box.button(
            QtWidgets.QDialogButtonBox.Cancel).setText(self.tr("取消"))

        button_layout.addWidget(clear_button)
        button_layout.addStretch()
        button_layout.addWidget(button_box)

        # 添加所有组件到主布局
        layout.addWidget(title_label)
        layout.addWidget(function_label)
        layout.addWidget(self.keySequenceEdit)
        layout.addWidget(tip_label)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def keyPressEvent(self, event):
        """处理按键事件"""
        # 处理Escape键
        if event.key() == QtCore.Qt.Key_Escape:
            self.reject()
        else:
            super(ShortcutEditDialog, self).keyPressEvent(event)

    def accept(self):
        """确认修改"""
        # 检查用户是否清除了快捷键
        if not self.keySequenceEdit.keySequence().toString():
            self.current_shortcut = None
        else:
            # 获取新的快捷键
            new_shortcut = self.keySequenceEdit.keySequence().toString()

            # 如果原始快捷键是列表格式，保持该格式
            if self.is_list_format and self.current_shortcut:
                if len(self.current_shortcut) > 0:
                    # 替换第一个元素，保留其余元素
                    self.current_shortcut[0] = new_shortcut
                else:
                    self.current_shortcut = [new_shortcut]
            else:
                self.current_shortcut = new_shortcut

        super(ShortcutEditDialog, self).accept()

    def showEvent(self, event):
        """窗口显示时更新主题"""
        super(ShortcutEditDialog, self).showEvent(event)
        # 获取当前主题
        app = QtWidgets.QApplication.instance()
        current_theme = app.property("currentTheme")
        if current_theme != self.is_dark:
            self.is_dark = current_theme == "dark"
            self.setThemeStyleSheet()

    def setThemeStyleSheet(self):
        """根据主题设置样式表"""
        if self.is_dark:
            self.setStyleSheet("""
                QDialog {
                    background-color: #252526;
                }
                QLabel {
                    font-size: 28px;
                    color: #ffffff;
                }
                QLabel#titleLabel {
                    font-size: 30px;
                    font-weight: bold;
                    color: #333333;
                }
                QPushButton {
                    background-color: #ffffff;
                    border: 1px solid #e0e0e0;
                    border-radius: 6px;
                    padding: 8px 16px;
                    color: #333333;
                    font-weight: bold;
                    min-width: 80px;
                    min-height: 32px;
                }
                QPushButton:hover {
                    background-color: #f0f0f0;
                    border-color: #d0d0d0;
                }
                QPushButton:pressed {
                    background-color: #e0e0e0;
                }
                QKeySequenceEdit {
                    border: 1px solid #e0e0e0;
                    border-radius: 6px;
                    padding: 10px;
                    background-color: white;
                    font-size: 20px;
                    min-height: 45px;
                }
                QKeySequenceEdit:focus {
                    border-color: #66afe9;
                }
            """)
        else:
            self.setStyleSheet("""
                QDialog {
                    background-color: #f5f5f5;
                }
                QLabel {
                    font-size: 28px;
                    color: #333333;
                }
                QLabel#titleLabel {
                    font-size: 30px;
                    font-weight: bold;
                    color: #333333;
                }
                QPushButton {
                    background-color: #ffffff;
                    border: 1px solid #e0e0e0;
                    border-radius: 6px;
                    padding: 8px 16px;
                    color: #333333;
                    font-weight: bold;
                    min-width: 80px;
                    min-height: 32px;
                }
                QPushButton:hover {
                    background-color: #f0f0f0;
                    border-color: #d0d0d0;
                }
                QPushButton:pressed {
                    background-color: #e0e0e0;
                }
                QKeySequenceEdit {
                    border: 1px solid #e0e0e0;
                    border-radius: 6px;
                    padding: 10px;
                    background-color: white;
                    font-size: 20px;
                    min-height: 45px;
                }
                QKeySequenceEdit:focus {
                    border-color: #66afe9;
                }
            """)
