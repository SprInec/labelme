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
    ):
        if fit_to_content is None:
            fit_to_content = {"row": False, "column": True}
        self._fit_to_content = fit_to_content

        super(LabelDialog, self).__init__(parent)
        self.edit = LabelQLineEdit()
        self.edit.setPlaceholderText(text)
        self.edit.setValidator(labelme.utils.labelValidator())
        self.edit.editingFinished.connect(self.postProcess)
        if flags:
            self.edit.textChanged.connect(self.updateFlags)
        self.edit_group_id = QtWidgets.QLineEdit()
        self.edit_group_id.setPlaceholderText("Group ID")
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
        # 移除固定高度，允许标签列表随窗口大小变化
        # self.labelList.setFixedHeight(150)
        # 设置最小高度，确保初始显示合理
        self.labelList.setMinimumHeight(150)
        # 设置大小策略为垂直方向可扩展
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding
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
        layout.addWidget(QtWidgets.QLabel("描述:"))
        layout.addWidget(self.editDescription)

        # 创建底部布局
        bottom_layout = QtWidgets.QHBoxLayout()

        # 添加颜色选择功能
        color_layout = QtWidgets.QHBoxLayout()
        color_layout.setAlignment(QtCore.Qt.AlignLeft)  # 设置左对齐
        color_label = QtWidgets.QLabel("标签颜色:")
        self.color_button = QtWidgets.QPushButton()
        self.color_button.setFixedWidth(50)
        self.color_button.setFixedHeight(20)
        self.selected_color = QtGui.QColor(0, 255, 0)  # 默认绿色
        self.update_color_button()
        self.color_button.clicked.connect(self.choose_color)
        color_layout.addWidget(color_label)
        color_layout.addWidget(self.color_button)

        # 添加按钮
        self.buttonBox = bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal,
            self,
        )
        bb.button(bb.Ok).setIcon(labelme.utils.newIcon("done"))
        bb.button(bb.Cancel).setIcon(labelme.utils.newIcon("undo"))
        bb.accepted.connect(self.validate)
        bb.rejected.connect(self.reject)

        # 将颜色选择和按钮添加到底部布局
        bottom_layout.addLayout(color_layout, 1)  # 设置拉伸因子为1，使其左对齐
        bottom_layout.addStretch(2)  # 添加弹性空间
        bottom_layout.addWidget(bb, 1)  # 设置拉伸因子为1

        # 将底部布局添加到主布局
        layout.addLayout(bottom_layout)

        self.setLayout(layout)

        # 设置对话框的大小策略
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Expanding
        )

        # 设置初始大小
        self.resize(500, 400)

        # completion
        completer = QtWidgets.QCompleter()
        if completion == "startswith":
            completer.setCompletionMode(QtWidgets.QCompleter.InlineCompletion)
        elif completion == "contains":
            completer.setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
            completer.setFilterMode(QtCore.Qt.MatchContains)
        else:
            raise ValueError("Unsupported completion: {}".format(completion))
        completer.setModel(self.labelList.model())
        self.edit.setCompleter(completer)

    def addLabelHistory(self, label):
        if self.labelList.findItems(label, QtCore.Qt.MatchExactly):
            return
        self.labelList.addItem(label)
        if self._sort_labels:
            self.labelList.sortItems()

    def labelSelected(self, item):
        self.edit.setText(item.text())

    def validate(self):
        if not self.edit.isEnabled():
            self.accept()
            return

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
            # self.labelList.setMinimumHeight(
            #     self.labelList.sizeHintForRow(0) * self.labelList.count() + 2
            # )
            pass
        if self._fit_to_content["column"]:
            # self.labelList.setMinimumWidth(
            #     self.labelList.sizeHintForColumn(0) + 2)
            pass

        # if text is None, the previous label in self.edit is kept
        if text is None:
            text = self.edit.text()
        # description is always initialized by empty text c.f., self.edit.text
        if description is None:
            description = ""
        self.editDescription.setText(description)
        # 设置颜色
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
        style = f"background-color: {self.selected_color.name()}; border: 1px solid #888888;"
        self.color_button.setStyleSheet(style)

    def get_color(self):
        """获取选择的颜色"""
        return self.selected_color

    def edit(self, value):
        """编辑标签并返回是否接受"""
        self.edit.setText(value)
        self.edit.setSelection(0, len(value))
        result = self.exec_()
        return result == QtWidgets.QDialog.Accepted

    def get_value(self):
        """获取标签文本"""
        return self.edit.text()

    def resizeEvent(self, event):
        """处理窗口大小变化事件，确保标签列表控件能正确调整大小"""
        super(LabelDialog, self).resizeEvent(event)
        # 窗口大小变化时，标签列表控件会自动调整大小，因为我们已经设置了合适的大小策略
