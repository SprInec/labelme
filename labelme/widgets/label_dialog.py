import re

from loguru import logger
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets

import labelme.utils

# TODO(unknown):
# - Calculate optimal position so as not to go out of screen area.


class LabelQLineEdit(QtWidgets.QLineEdit):
    def setListWidget(self, list_widget):
        self.list_widget = list_widget

    def keyPressEvent(self, e):
        if e.key() in [QtCore.Qt.Key_Up, QtCore.Qt.Key_Down]:
            self.list_widget.keyPressEvent(e)
        else:
            super(LabelQLineEdit, self).keyPressEvent(e)


class LabelDialog(QtWidgets.QDialog):
    def __init__(
        self,
        text="Enter object label",
        parent=None,
        labels=None,
        sort_labels=True,
        show_text_field=True,
        completion="startswith",
        fit_to_content=None,
        flags=None,
        app=None,
    ):
        if fit_to_content is None:
            fit_to_content = {"row": False, "column": True}
        self._fit_to_content = fit_to_content

        # 保存对主应用程序的引用
        self.app = app

        super(LabelDialog, self).__init__(parent)
        self.edit = LabelQLineEdit()
        self.edit.setPlaceholderText(text)
        self.edit.setValidator(labelme.utils.labelValidator())
        self.edit.editingFinished.connect(self.postProcess)
        if flags:
            self.edit.textChanged.connect(self.updateFlags)
        self.edit_group_id = QtWidgets.QLineEdit()
        self.edit_group_id.setPlaceholderText("GID")
        self.edit_group_id.setValidator(
            QtGui.QRegExpValidator(QtCore.QRegExp(r"\d*"), None)
        )
        layout = QtWidgets.QVBoxLayout()
        if show_text_field:
            layout_edit = QtWidgets.QHBoxLayout()
            layout_edit.addWidget(self.edit, 6)
            layout_edit.addWidget(self.edit_group_id, 2)
            layout.addLayout(layout_edit)

        # label_list
        self.labelList = QtWidgets.QListWidget()
        if self._fit_to_content["row"]:
            self.labelList.setHorizontalScrollBarPolicy(
                QtCore.Qt.ScrollBarAlwaysOff)
        if self._fit_to_content["column"]:
            self.labelList.setVerticalScrollBarPolicy(
                QtCore.Qt.ScrollBarAlwaysOff)
        self._sort_labels = sort_labels
        if labels:
            self.labelList.addItems(labels)
        if self._sort_labels:
            self.labelList.sortItems()
        else:
            self.labelList.setDragDropMode(
                QtWidgets.QAbstractItemView.InternalMove)
        self.labelList.currentItemChanged.connect(self.labelSelected)
        self.labelList.itemDoubleClicked.connect(self.labelDoubleClicked)
        # 设置最小高度，确保初始显示合理
        self.labelList.setMinimumHeight(200)  # 增加最小高度
        # 设置大小策略为垂直方向可扩展
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding,  # 水平方向也可扩展
            QtWidgets.QSizePolicy.Expanding
        )
        self.labelList.setSizePolicy(sizePolicy)

        self.edit.setListWidget(self.labelList)
        layout.addWidget(self.labelList)

        # label_flags
        if flags is None:
            flags = {}
        self._flags = flags
        self.flagsLayout = QtWidgets.QVBoxLayout()
        self.resetFlags()
        layout.addItem(self.flagsLayout)
        self.edit.textChanged.connect(self.updateFlags)

        # 添加description输入框
        self.editDescription = QtWidgets.QLineEdit()
        self.editDescription.setPlaceholderText("Description (optional)")

        layout.addWidget(self.editDescription)

        # 创建底部布局
        bottom_layout = QtWidgets.QHBoxLayout()

        # 添加颜色选择功能
        color_layout = QtWidgets.QHBoxLayout()
        color_layout.setAlignment(QtCore.Qt.AlignLeft)  # 设置左对齐
        self.color_button = QtWidgets.QPushButton()
        self.color_button.setObjectName("color_button")
        self.color_button.setFixedSize(32, 32)
        self.selected_color = QtGui.QColor(0, 255, 0)  # 默认绿色
        self.update_color_button()
        self.color_button.clicked.connect(self.choose_color)
        color_layout.addWidget(self.color_button, 0, QtCore.Qt.AlignVCenter)

        # 添加按钮
        self.buttonBox = bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal,
            self,
        )
        bb.button(bb.Ok).setIcon(labelme.utils.newIcon("icons8-done-48"))
        bb.button(bb.Cancel).setIcon(labelme.utils.newIcon("icons8-undo-60"))
        bb.accepted.connect(self.validate)
        bb.rejected.connect(self.reject)

        # 将颜色选择和按钮添加到底部布局
        bottom_layout.addLayout(color_layout, 1)  # 设置拉伸因子为1，使其左对齐
        bottom_layout.addStretch(2)  # 添加弹性空间
        bottom_layout.addWidget(bb, 1)  # 设置拉伸因子为1

        # 将底部布局添加到主布局
        layout.addLayout(bottom_layout)

        self.setLayout(layout)

        # 设置初始大小
        self.resize(580, 650)

        # 设置对话框的最小尺寸
        self.setMinimumWidth(580)
        self.setMinimumHeight(500)

        # completion
        completer = QtWidgets.QCompleter()
        if completion == "startswith":
            completer.setCompletionMode(QtWidgets.QCompleter.InlineCompletion)
            completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        elif completion == "contains":
            completer.setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
            completer.setFilterMode(QtCore.Qt.MatchContains)
            completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        else:
            raise ValueError("Unsupported completion: {}".format(completion))
        stringListModel = QtCore.QStringListModel()
        if labels:
            stringListModel.setStringList(labels)
        completer.setModel(stringListModel)
        self.edit.setCompleter(completer)

        self.setStyleSheet("""
    
            QPushButton {
                padding: 8px 20px;
                border-radius: 6px;
                font-size: 9pt;
                margin: 5px;
                min-width: 110px;
            }
            QPushButton#color_button {
                padding: 0px;
                min-width: 30px;
                border-radius: 0px;
                border: 1px solid #3d3d3d;
                margin-top: 0px;
            }
        """)

    def addLabelHistory(self, label):
        if self.labelList.findItems(label, QtCore.Qt.MatchExactly):
            return
        self.labelList.addItem(label)
        if self._sort_labels:
            self.labelList.sortItems()

    def labelSelected(self, item):
        self.edit.setText(item.text())
        text = item.text().strip()

        # 如果app对象存在，使用app的颜色管理
        if self.app:
            clean_text = text.replace("●", "").strip()
            # 提取纯文本标签名，去除任何HTML标记
            if '<font' in clean_text:
                clean_text = re.sub(r'<[^>]*>|</[^>]*>',
                                    '', clean_text).strip()

            # 使用app的颜色获取方法
            color = self.app._get_rgb_by_label(clean_text)
            if color:
                # 转换成QColor
                self.selected_color = QtGui.QColor(*color)
                self.update_color_button()
                return

        # 降级处理：如果无法从app获取颜色，尝试从标签文本提取
        if "●" in text:
            try:
                # 尝试提取颜色代码
                color_str = text.split('color="')[1].split('">')[0]
                # 解析十六进制颜色值
                r = int(color_str[1:3], 16)
                g = int(color_str[3:5], 16)
                b = int(color_str[5:7], 16)
                self.selected_color = QtGui.QColor(r, g, b)
                self.update_color_button()
                return
            except (IndexError, ValueError):
                pass

        # 如果没有找到颜色，保持当前颜色
        self.update_color_button()

    def validate(self):
        text = self.edit.text()
        if hasattr(text, "strip"):
            text = text.strip()
        else:
            text = text.trimmed()
        if text:
            self.accept()

    def labelDoubleClicked(self, item):
        self.validate()

    def postProcess(self):
        text = self.edit.text()
        if hasattr(text, "strip"):
            text = text.strip()
        else:
            text = text.trimmed()
        self.edit.setText(text)

    def updateFlags(self, label_new):
        # keep state of shared flags
        flags_old = self.getFlags()

        flags_new = {}
        for pattern, keys in self._flags.items():
            if re.match(pattern, label_new):
                for key in keys:
                    flags_new[key] = flags_old.get(key, False)
        self.setFlags(flags_new)

    def deleteFlags(self):
        for i in reversed(range(self.flagsLayout.count())):
            item = self.flagsLayout.itemAt(i).widget()
            self.flagsLayout.removeWidget(item)
            item.setParent(None)

    def resetFlags(self, label=""):
        flags = {}
        for pattern, keys in self._flags.items():
            if re.match(pattern, label):
                for key in keys:
                    flags[key] = False
        self.setFlags(flags)

    def setFlags(self, flags):
        self.deleteFlags()
        for key in flags:
            item = QtWidgets.QCheckBox(key, self)
            item.setChecked(flags[key])
            self.flagsLayout.addWidget(item)
            item.show()

    def getFlags(self):
        flags = {}
        for i in range(self.flagsLayout.count()):
            item = self.flagsLayout.itemAt(i).widget()
            flags[item.text()] = item.isChecked()
        return flags

    def getGroupId(self):
        group_id = self.edit_group_id.text()
        if group_id:
            return int(group_id)
        return None

    def getDescription(self):
        return self.editDescription.text()

    def popUp(self, text=None, move=True, flags=None, group_id=None, description=None, color=None):
        # 移除这些限制，允许窗口自由调整大小
        if self._fit_to_content["row"]:
            pass
        if self._fit_to_content["column"]:
            pass

        # if text is None, the previous label in self.edit is kept
        if text is None:
            text = self.edit.text()
        else:
            text = text.strip()

        # description is always initialized by empty text c.f., self.edit.text
        if description is None:
            description = ""
        self.editDescription.setText(description)

        # 如果没有提供颜色或提供的是默认绿色，尝试查找标签对应的颜色
        has_found_color = False
        if color is None or color.getRgb()[:3] == (0, 255, 0):  # 默认绿色
            # 1. 首先从主应用程序获取颜色
            clean_text = text.replace("●", "").strip()
            # 移除HTML标记
            if '<font' in clean_text:
                clean_text = re.sub(r'<[^>]*>|</[^>]*>',
                                    '', clean_text).strip()

            if self.app:
                rgb_color = self.app._get_rgb_by_label(clean_text)
                if rgb_color:
                    color = QtGui.QColor(*rgb_color)
                    has_found_color = True

            # 2. 如果从app获取不到，尝试从标签文本提取
            if not has_found_color:
                if "●" in text and 'color="' in text:
                    try:
                        color_str = text.split('color="')[1].split('">')[0]
                        r = int(color_str[1:3], 16)
                        g = int(color_str[3:5], 16)
                        b = int(color_str[5:7], 16)
                        color = QtGui.QColor(r, g, b)
                        has_found_color = True
                    except (IndexError, ValueError):
                        pass

        # 设置颜色按钮
        if color is not None and isinstance(color, QtGui.QColor):
            self.selected_color = color
            self.update_color_button()

        if flags:
            self.setFlags(flags)
        else:
            self.resetFlags(text)
        self.edit.setText(text)
        self.edit.setSelection(0, len(text))
        if group_id is None:
            self.edit_group_id.clear()
        else:
            self.edit_group_id.setText(str(group_id))
        items = self.labelList.findItems(text, QtCore.Qt.MatchFixedString)
        if items:
            if len(items) != 1:
                logger.warning("Label list has duplicate '{}'".format(text))
            self.labelList.setCurrentItem(items[0])
            row = self.labelList.row(items[0])
            self.edit.completer().setCurrentRow(row)
        self.edit.setFocus(QtCore.Qt.PopupFocusReason)

        # 确保对话框显示在屏幕中央
        if move:
            # 获取屏幕几何信息
            screen = QtWidgets.QApplication.desktop().screenGeometry()
            # 获取对话框大小
            size = self.sizeHint()
            # 计算居中位置
            x = (screen.width() - size.width()) // 2
            y = (screen.height() - size.height()) // 2
            self.move(x, y)

        if self.exec_():
            return (
                self.edit.text(),
                self.getFlags(),
                self.getGroupId(),
                self.getDescription(),
                self.get_color(),
            )
        else:
            return None

    def choose_color(self):
        """打开颜色选择对话框"""
        color = QtWidgets.QColorDialog.getColor(
            self.selected_color, self, "选择标签颜色"
        )
        if color.isValid():
            self.selected_color = color
            self.update_color_button()

    def update_color_button(self):
        """更新颜色按钮的样式"""
        style = f"background-color: {self.selected_color.name()}; border: 1px solid #888888; border-radius: 14px;"
        self.color_button.setStyleSheet(style)

    def get_color(self):
        """获取选择的颜色"""
        return self.selected_color

    def resizeEvent(self, event):
        """处理窗口大小变化事件，确保标签列表控件能正确调整大小"""
        super(LabelDialog, self).resizeEvent(event)
        # 窗口大小变化时，标签列表控件会自动调整大小，因为我们已经设置了合适的大小策略

    def labelSelectionChanged(self):
        # 处理选择变化
        if self.labelList.currentItem():
            self.labelSelected(self.labelList.currentItem())

    def changeColor(self):
        # 确保向后兼容
        self.choose_color()
