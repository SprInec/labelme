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


class LabelItemDelegate(QtWidgets.QStyledItemDelegate):
    """自定义标签项的代理，用于呈现更美观的样式"""

    def __init__(self, parent=None):
        super(LabelItemDelegate, self).__init__(parent)

    def paint(self, painter, option, index):
        # 保存画笔状态
        painter.save()

        # 圆角绘制准备
        radius = 8  # 定义圆角半径

        # 调整矩形区域，缩小更多以形成更大的间隙效果
        option_rect = option.rect.adjusted(5, 5, -5, -5)

        # 获取项
        color_data = index.data(QtCore.Qt.UserRole+1)

        # 创建一个路径来绘制圆角矩形
        path = QtGui.QPainterPath()
        path.addRoundedRect(
            QtCore.QRectF(option_rect),
            radius,
            radius
        )

        # 背景色绘制
        if color_data and isinstance(color_data, QtGui.QColor):
            # 使用标签颜色创建更淡的半透明背景 (10%透明度)
            bg_color = QtGui.QColor(color_data)
            bg_color.setAlpha(25)  # 10%透明度 (255*0.1≈25)

            # 使用路径来填充圆角背景
            painter.fillPath(path, bg_color)

            # 左边框宽度
            border_width = 12  # 增加左边框宽度

            # 绘制左边框 (使用圆角)
            border_path = QtGui.QPainterPath()
            border_rect = QtCore.QRectF(
                option_rect.left(),
                option_rect.top(),
                border_width,
                option_rect.height()
            )
            # 只对左边使用圆角
            border_path.addRoundedRect(border_rect, radius, radius)
            # 裁剪掉右边的圆角
            clip_path = QtGui.QPainterPath()
            clip_path.addRect(
                option_rect.left(),
                option_rect.top(),
                border_width / 2,  # 只显示左边的一半
                option_rect.height()
            )
            # 应用裁剪
            border_path = border_path.intersected(clip_path)

            # 填充左边框
            border_color = QtGui.QColor(color_data)
            painter.fillPath(border_path, border_color)

        # 选中状态高亮 (也使用圆角)
        if option.state & QtWidgets.QStyle.State_Selected:
            highlight_color = QtGui.QColor(0, 120, 215, 178)  # 70%透明度
            painter.fillPath(path, highlight_color)
            painter.setPen(QtGui.QColor(255, 255, 255))
        elif option.state & QtWidgets.QStyle.State_MouseOver:
            hover_color = QtGui.QColor(0, 0, 0, 13)  # 5%透明度
            painter.fillPath(path, hover_color)
            painter.setPen(QtGui.QColor(0, 0, 0))
        else:
            painter.setPen(QtGui.QColor(0, 0, 0))

        # 文本绘制区域 (增加左边距)
        text_rect = QtCore.QRect(
            option_rect.left() + border_width + 12,  # 左边框宽度 + 额外间距
            option_rect.top(),
            option_rect.width() - (border_width + 20),  # 适当调整右边距
            option_rect.height()
        )

        # 设置字体
        font = painter.font()
        font.setPointSize(10)
        painter.setFont(font)

        # 绘制文本
        text = index.data(QtCore.Qt.DisplayRole)
        # 移除HTML标记后绘制纯文本
        if text and '<font' in text:
            clean_text = re.sub(r'<[^>]*>●|</font>', '', text).strip()
            painter.drawText(text_rect, QtCore.Qt.AlignVCenter, clean_text)
        else:
            painter.drawText(text_rect, QtCore.Qt.AlignVCenter, text)

        # 恢复画笔状态
        painter.restore()

    def sizeHint(self, option, index):
        # 增大项高度以增强呼吸感
        size = super(LabelItemDelegate, self).sizeHint(option, index)
        size.setHeight(60)  # 进一步增大项高度
        return size


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
        self.edit_group_id.setAlignment(QtCore.Qt.AlignCenter)
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

        # 设置标签列表样式
        self.labelList.setStyleSheet("""
            QListWidget {
                background-color: #FFFFFF;
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 10px;
                outline: none;
                padding-right: 15px; /* 为滚动条留出空间 */
            }
            QListWidget::item {
                border-radius: 8px;
                padding: 10px;
                margin: 10px 5px;  /* 增加垂直和水平间距 */
            }
            QListWidget::item:selected {
                color: white;
                border: none;
            }
            /* 滚动条样式 */
            QScrollBar:vertical {
                background-color: #f0f0f0;
                width: 8px;
                margin: 10px 0 10px 0;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background-color: #c0c0c0;
                min-height: 30px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #a0a0a0;
            }
            QScrollBar::add-line:vertical, 
            QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, 
            QScrollBar::sub-page:vertical {
                background: none;
            }
        """)

        # 设置自定义代理
        self.label_delegate = LabelItemDelegate(self.labelList)
        self.labelList.setItemDelegate(self.label_delegate)
        self._sort_labels = sort_labels
        if labels:
            # 添加标签并应用样式
            for label in labels:
                item = QtWidgets.QListWidgetItem(label)
                self.labelList.addItem(item)
                self._set_label_item_style(item, label)
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
        item = QtWidgets.QListWidgetItem(label)
        self.labelList.addItem(item)
        # 设置样式
        self._set_label_item_style(item, label)
        if self._sort_labels:
            self.labelList.sortItems()

    def labelSelected(self, item):
        self.edit.setText(item.text())
        text = item.text().strip()

        # 重新应用样式以确保显示正确
        self._set_label_item_style(item, text)

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

    def _set_label_item_style(self, item, label_text):
        """设置标签项的样式，添加左边框和背景色

        Args:
            item: QListWidgetItem 对象
            label_text: 标签文本
        """
        # 获取纯文本标签（移除HTML和特殊字符）
        clean_text = label_text.replace("●", "").strip()
        # 移除HTML标记
        if '<font' in clean_text:
            clean_text = re.sub(r'<[^>]*>|</[^>]*>', '', clean_text).strip()

        # 获取标签颜色
        rgb_color = None
        if self.app:
            rgb_color = self.app._get_rgb_by_label(clean_text)

        if not rgb_color:
            # 如果获取不到颜色，尝试从标签文本提取
            if "●" in label_text and 'color="' in label_text:
                try:
                    color_str = label_text.split('color="')[1].split('">')[0]
                    r = int(color_str[1:3], 16)
                    g = int(color_str[3:5], 16)
                    b = int(color_str[5:7], 16)
                    rgb_color = (r, g, b)
                except (IndexError, ValueError):
                    # 使用默认绿色
                    rgb_color = (0, 255, 0)
            else:
                # 使用默认绿色
                rgb_color = (0, 255, 0)

        # 创建QColor对象
        r, g, b = rgb_color
        color = QtGui.QColor(r, g, b)

        # 设置背景透明度与代理一致
        background = QtGui.QBrush(QtGui.QColor(r, g, b, 25))  # 10%透明度

        # 设置项的背景色
        item.setBackground(background)

        # 使用自定义数据保存边框颜色
        item.setData(QtCore.Qt.UserRole+1, color)
