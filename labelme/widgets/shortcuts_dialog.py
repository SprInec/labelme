from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

from labelme.config import get_config
from labelme.config.config import save_config


class ShortcutsDialog(QtWidgets.QDialog):
    """快捷键设置对话框"""

    def __init__(self, parent=None):
        super(ShortcutsDialog, self).__init__(parent)
        self.parent = parent
        self.config = get_config()
        self.shortcuts = self.config.get("shortcuts", {})
        self.modified_shortcuts = self.shortcuts.copy()

        self.setWindowTitle(self.tr("快捷键设置"))
        self.setWindowFlags(
            self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint
        )
        self.setMinimumWidth(750)
        self.setMinimumHeight(650)

        self.initUI()

    def initUI(self):
        """初始化UI"""
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # 创建说明标签
        desc_label = QtWidgets.QLabel(self.tr("在下表中可以查看和修改软件的快捷键设置："))
        layout.addWidget(desc_label)

        # 创建表格
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels([self.tr("功能"), self.tr("快捷键")])
        self.table.setColumnWidth(0, 400)
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
        self.table.setStyleSheet("QTableView::item { padding: 8px; }")
        self.table.verticalHeader().setDefaultSectionSize(36)

        # 填充表格
        self.populateTable()

        # 添加按钮
        button_layout = QtWidgets.QHBoxLayout()

        self.edit_button = QtWidgets.QPushButton(self.tr("编辑"))
        self.edit_button.clicked.connect(self.editShortcut)

        self.reset_button = QtWidgets.QPushButton(self.tr("重置"))
        self.reset_button.clicked.connect(self.resetShortcut)

        self.reset_all_button = QtWidgets.QPushButton(self.tr("全部重置"))
        self.reset_all_button.clicked.connect(self.resetAllShortcuts)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        button_layout.addWidget(self.edit_button)
        button_layout.addWidget(self.reset_button)
        button_layout.addWidget(self.reset_all_button)
        button_layout.addStretch()
        button_layout.addWidget(button_box)

        layout.addWidget(self.table)
        layout.addLayout(button_layout)
        self.setLayout(layout)

    def populateTable(self):
        """填充表格数据"""
        self.table.setRowCount(0)

        # 功能名称映射
        function_names = {
            "close": "关闭",
            "open": "打开",
            "open_dir": "打开目录",
            "quit": "退出",
            "save": "保存",
            "save_as": "另存为",
            "save_to": "更改输出路径",
            "delete_file": "删除文件",
            "open_next": "下一张图片",
            "open_prev": "上一张图片",
            "zoom_in": "放大",
            "zoom_out": "缩小",
            "zoom_to_original": "原始大小",
            "fit_window": "适应窗口",
            "fit_width": "适应宽度",
            "create_polygon": "创建多边形",
            "create_rectangle": "创建矩形",
            "create_circle": "创建圆形",
            "create_line": "创建线条",
            "create_point": "创建点",
            "create_linestrip": "创建折线",
            "edit_polygon": "编辑多边形",
            "delete_polygon": "删除多边形",
            "duplicate_polygon": "复制多边形",
            "copy_polygon": "复制",
            "paste_polygon": "粘贴",
            "undo": "撤销",
            "undo_last_point": "撤销上一个点",
            "add_point_to_edge": "添加点到边缘",
            "edit_label": "编辑标签",
            "toggle_keep_prev_mode": "保持上一个模式",
            "remove_selected_point": "删除选中的点",
            "show_all_polygons": "显示所有多边形",
            "hide_all_polygons": "隐藏所有多边形",
            "toggle_all_polygons": "切换所有多边形",
        }

        row = 0
        for key, value in sorted(self.modified_shortcuts.items()):
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

            row += 1

    def formatShortcutText(self, shortcut):
        """格式化快捷键文本"""
        if shortcut is None:
            return "无"
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

        dialog = ShortcutEditDialog(self, key, current_shortcut)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            new_shortcut = dialog.keySequenceEdit.keySequence().toString()
            if not new_shortcut:
                new_shortcut = None

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

        # 保存配置
        save_config(self.config)

        # 提示用户重启应用
        QtWidgets.QMessageBox.information(
            self,
            self.tr("提示"),
            self.tr("快捷键设置已保存，部分更改可能需要重启应用后生效。")
        )

        super(ShortcutsDialog, self).accept()


class ShortcutEditDialog(QtWidgets.QDialog):
    """快捷键编辑对话框"""

    def __init__(self, parent=None, key=None, current_shortcut=None):
        super(ShortcutEditDialog, self).__init__(parent)
        self.key = key
        self.current_shortcut = current_shortcut

        self.setWindowTitle(self.tr("编辑快捷键"))
        self.setFixedSize(500, 300)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # 显示功能名称
        function_label = QtWidgets.QLabel(self.tr("功能: {}").format(key))
        function_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        layout.addWidget(function_label)

        # 快捷键编辑控件
        self.keySequenceEdit = QtWidgets.QKeySequenceEdit()
        if current_shortcut:
            self.keySequenceEdit.setKeySequence(
                QtGui.QKeySequence(current_shortcut))

        layout.addWidget(self.keySequenceEdit)

        # 提示信息
        tip_label = QtWidgets.QLabel(self.tr("请按下新的快捷键组合"))
        tip_label.setStyleSheet("color: gray;")
        layout.addWidget(tip_label)

        # 按钮
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        clear_button = QtWidgets.QPushButton(self.tr("清除"))
        clear_button.clicked.connect(self.keySequenceEdit.clear)
        button_box.addButton(
            clear_button, QtWidgets.QDialogButtonBox.ResetRole)

        layout.addWidget(button_box)

        self.setLayout(layout)
