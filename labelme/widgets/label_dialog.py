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
        sort_labels=False,
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

        # 获取标签云布局配置
        self._use_cloud_layout = False
        if app and hasattr(app, '_config'):
            self._use_cloud_layout = app._config.get(
                'label_cloud_layout', False)

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

        # 创建主布局
        self.main_layout = layout

        # 始终创建标准列表布局和流式标签云布局
        self.createStandardListLayout(layout, labels, sort_labels)
        self.createCloudLayout(layout, labels)

        # 根据设置显示或隐藏相应的布局
        if self._use_cloud_layout:
            # 如果启用标签云布局，则显示流式布局，隐藏标准列表
            self.labelList.setVisible(False)
            self.scrollArea.setVisible(True)
        else:
            # 否则显示标准列表，隐藏流式布局
            self.labelList.setVisible(True)
            self.scrollArea.setVisible(False)

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

        # 添加布局切换按钮
        self.layout_toggle_button = QtWidgets.QPushButton()
        self.layout_toggle_button.setObjectName("layout_toggle_button")
        self.layout_toggle_button.setFixedSize(36, 36)
        self.layout_toggle_button.setToolTip(self.tr("切换标签布局模式"))
        self.layout_toggle_button.clicked.connect(self.onLayoutToggleClicked)
        # 设置图标
        if self._use_cloud_layout:
            self.layout_toggle_button.setIcon(
                labelme.utils.newIcon("icons8-grid-view-48"))
        else:
            self.layout_toggle_button.setIcon(
                labelme.utils.newIcon("icons8-list-view-48"))
        color_layout.addWidget(self.layout_toggle_button,
                               0, QtCore.Qt.AlignVCenter)

        # 在布局按钮和颜色按钮之间添加一定的间距
        color_layout.addSpacing(5)

        # 添加颜色选择按钮
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
            QPushButton#layout_toggle_button {
                padding: 0px;
                min-width: 36px;
                min-height: 36px;
                border-radius: 0px;
                border: none;
                background-color: transparent;
                margin: 0px;
            }
            QPushButton#layout_toggle_button:hover {
                background-color: rgba(0, 0, 0, 0.05);
            }
            QPushButton#layout_toggle_button:pressed {
                background-color: rgba(0, 0, 0, 0.1);
            }
        """)

    def createStandardListLayout(self, layout, labels, sort_labels):
        """创建标准的列表布局"""
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
            QListWidget::item:hover {
                cursor: grab;  /* 鼠标悬停时显示抓取光标 */
            }
            QListWidget::item:pressed {
                cursor: grabbing;  /* 鼠标按下时显示抓取中光标 */
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

        # 启用拖放功能
        self.labelList.setDragEnabled(True)
        self.labelList.setAcceptDrops(True)
        self.labelList.setDragDropMode(
            QtWidgets.QAbstractItemView.InternalMove)
        self.labelList.setDefaultDropAction(QtCore.Qt.MoveAction)

        if labels:
            # 添加标签并应用样式
            for label in labels:
                item = QtWidgets.QListWidgetItem(label)
                self.labelList.addItem(item)
                self._set_label_item_style(item, label)
        if self._sort_labels:
            self.labelList.sortItems()
        # 不需要else语句，因为上面已经设置了拖放模式

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

    def createCloudLayout(self, layout, labels):
        """创建标签云流式布局"""
        # 创建一个滚动区域，用于包含流式布局的标签
        self.scrollArea = QtWidgets.QScrollArea()
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.scrollArea.setStyleSheet("""
            QScrollArea {
                background-color: #FFFFFF;
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 8px;
            }
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

        # 创建一个容器窗口
        self.cloudContainer = LabelCloudContainer(self)
        # 创建流式布局 - 设置更大的间距
        self.cloudLayout = FlowLayout()
        self.cloudLayout.setSpacing(12)  # 设置项目间距为12像素
        self.cloudLayout.setContentsMargins(10, 10, 10, 10)  # 设置边距
        self.cloudContainer.setLayout(self.cloudLayout)

        # 添加标签到流式布局
        if labels:
            for label in labels:
                self.addLabelToCloud(label)

        # 将容器添加到滚动区域
        self.scrollArea.setWidget(self.cloudContainer)
        # 将滚动区域添加到主布局
        layout.addWidget(self.scrollArea)
        # 最低高度设置
        self.scrollArea.setMinimumHeight(240)  # 增加最小高度从200到240

    def addLabelToCloud(self, label_text):
        """添加标签到标签云布局"""
        # 避免重复添加标签
        for item in self.cloudContainer.label_items:
            # 清理标签文本以进行比较
            item_clean_text = item.clean_text
            if '<font' in item_clean_text:
                item_clean_text = re.sub(
                    r'<[^>]*>|</[^>]*>', '', item_clean_text).strip()

            label_clean_text = label_text
            if '<font' in label_clean_text:
                label_clean_text = re.sub(
                    r'<[^>]*>|</[^>]*>', '', label_clean_text).strip()

            # 如果标签已存在，则不添加
            if item_clean_text == label_clean_text:
                return

        # 创建一个标签项小部件
        label_widget = LabelCloudItem(label_text, self.cloudContainer)

        # 获取标签颜色
        clean_text = label_text.replace("●", "").strip()
        if '<font' in clean_text:
            clean_text = re.sub(r'<[^>]*>|</[^>]*>',
                                '', clean_text).strip()

        # 检查是否为当前编辑的标签，如果是，使用当前选择的颜色
        rgb_color = None
        use_current_color = False

        # 检查当前编辑的标签是否与添加的标签相同
        if self.edit.text().strip() == label_text:
            use_current_color = True
            current_color = self.get_color()
            rgb_color = (current_color.red(),
                         current_color.green(), current_color.blue())

        # 如果不使用当前选择的颜色，从应用程序获取
        if not use_current_color and self.app:
            rgb_color = self.app._get_rgb_by_label(clean_text)

        if not rgb_color:
            # 默认使用绿色
            rgb_color = (0, 255, 0)

        # 设置标签项颜色
        label_widget.setLabelColor(QtGui.QColor(*rgb_color))

        # 将标签小部件添加到流式布局
        self.cloudLayout.addWidget(label_widget)

        # 将标签项添加到容器的跟踪列表中
        self.cloudContainer.addLabelItem(label_widget)

        # 连接双击信号
        label_widget.doubleClicked.connect(
            lambda: self.cloudItemDoubleClicked(label_text))
        # 连接选中信号
        label_widget.clicked.connect(
            lambda: self.cloudItemSelected(label_text))

        # 强制更新布局
        self.cloudContainer.updateGeometry()
        self.scrollArea.updateGeometry()

    def toggleCloudLayout(self, use_cloud=None):
        """切换布局模式"""
        if use_cloud is None:
            self._use_cloud_layout = not self._use_cloud_layout
        else:
            self._use_cloud_layout = use_cloud

        # 根据布局模式显示/隐藏对应的控件
        if hasattr(self, 'scrollArea'):
            self.scrollArea.setVisible(self._use_cloud_layout)
        if hasattr(self, 'labelList'):
            self.labelList.setVisible(not self._use_cloud_layout)

        # 更新布局切换按钮图标
        if hasattr(self, 'layout_toggle_button'):
            if self._use_cloud_layout:
                self.layout_toggle_button.setIcon(
                    labelme.utils.newIcon("icons8-grid-view-48"))
                self.layout_toggle_button.setToolTip(self.tr("切换为列表布局"))
            else:
                self.layout_toggle_button.setIcon(
                    labelme.utils.newIcon("icons8-list-view-48"))
                self.layout_toggle_button.setToolTip(self.tr("切换为流式布局"))

        # 同步主应用程序的布局设置
        if self.app and hasattr(self.app, '_config') and hasattr(self.app, 'cloud_layout_action'):
            # 仅当设置与应用不一致时更新应用设置
            if self.app._config.get('label_cloud_layout', False) != self._use_cloud_layout:
                self.app._config['label_cloud_layout'] = self._use_cloud_layout
                self.app.cloud_layout_action.setChecked(self._use_cloud_layout)

                # 保存到配置文件
                try:
                    from labelme.config import save_config
                    save_config(self.app._config)
                except Exception as e:
                    logger.exception("保存标签云布局配置失败: %s", e)

    def cloudItemSelected(self, label_text):
        """流式布局中的标签被选中"""
        self.edit.setText(label_text)

        # 更新颜色按钮与标签选中时的颜色一致
        clean_text = label_text.replace("●", "").strip()
        # 提取纯文本标签名，去除任何HTML标记
        if '<font' in clean_text:
            clean_text = re.sub(r'<[^>]*>|</[^>]*>', '', clean_text).strip()

        # 使用app的颜色获取方法
        if self.app:
            rgb_color = self.app._get_rgb_by_label(clean_text)
            if rgb_color:
                # 转换成QColor
                self.selected_color = QtGui.QColor(*rgb_color)
                self.update_color_button()
                return

        # 降级处理：如果无法从app获取颜色，尝试从标签文本提取
        if "●" in label_text and 'color="' in label_text:
            try:
                # 尝试提取颜色代码
                color_str = label_text.split('color="')[1].split('">')[0]
                # 解析十六进制颜色值
                r = int(color_str[1:3], 16)
                g = int(color_str[3:5], 16)
                b = int(color_str[5:7], 16)
                self.selected_color = QtGui.QColor(r, g, b)
                self.update_color_button()
            except (IndexError, ValueError):
                pass

    def cloudItemDoubleClicked(self, label_text):
        """流式布局中的标签被双击"""
        self.edit.setText(label_text)

        # 确保清除所有标签的选中状态，防止视觉上的混乱
        if hasattr(self, 'cloudContainer'):
            self.cloudContainer.clearAllSelection()

        self.validate()

    def addLabelHistory(self, label):
        # 添加到列表视图
        if self.labelList.findItems(label, QtCore.Qt.MatchExactly):
            return
        item = QtWidgets.QListWidgetItem(label)
        self.labelList.addItem(item)

        # 获取标签颜色 - 优先使用当前选择的颜色
        clean_text = label.replace("●", "").strip()
        if '<font' in clean_text:
            clean_text = re.sub(r'<[^>]*>|</[^>]*>', '', clean_text).strip()

        # 使用当前选择的颜色
        current_color = self.get_color()
        use_current_color = False

        # 检查当前编辑的标签是否与添加的标签相同
        if self.edit.text().strip() == label:
            use_current_color = True

        # 如果app对象存在且当前未编辑标签，从app获取颜色
        if not use_current_color and self.app:
            rgb_color = self.app._get_rgb_by_label(clean_text)
            if rgb_color:
                current_color = QtGui.QColor(*rgb_color)

        # 设置标签项样式，应用获取到的颜色
        background = QtGui.QBrush(QtGui.QColor(current_color.red(
        ), current_color.green(), current_color.blue(), 25))  # 10%透明度
        item.setBackground(background)
        item.setData(QtCore.Qt.UserRole+1, current_color)

        if self._sort_labels:
            self.labelList.sortItems()

        # 添加到标签云视图
        if hasattr(self, 'cloudLayout'):
            self.addLabelToCloud(label)

            # 如果使用当前选择的颜色，同步到云布局中的标签项
            if use_current_color:
                for label_item in self.cloudContainer.label_items:
                    if label_item.clean_text == clean_text:
                        label_item.setLabelColor(current_color)
                        label_item.update()

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
            # 在接受对话框前，确保当前标签的颜色被正确同步到主应用程序
            if self.app and hasattr(self.app, '_update_same_label_colors'):
                clean_text = text.replace("●", "").strip()
                if '<font' in clean_text:
                    clean_text = re.sub(
                        r'<[^>]*>|</[^>]*>', '', clean_text).strip()

                # 获取当前选中的颜色并同步到应用程序
                color = self.get_color()
                self.app._update_same_label_colors(clean_text, color)

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

        # 检查布局模式配置是否变化
        if self.app and hasattr(self.app, '_config'):
            new_layout_mode = self.app._config.get('label_cloud_layout', False)
            if new_layout_mode != self._use_cloud_layout:
                self.toggleCloudLayout(new_layout_mode)

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

        # 刷新所有标签项的颜色
        self.refreshLabelColors()

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

        # 根据当前布局模式选择设置选中项
        if self._use_cloud_layout:
            # 在标签云视图中查找并选中项
            # 注意：流式布局中没有直接的选中状态，但我们可以确保文本输入框显示正确的文本
            pass
        else:
            # 在标准列表视图中查找并选中项
            items = self.labelList.findItems(text, QtCore.Qt.MatchFixedString)
            if items:
                if len(items) != 1:
                    logger.warning(
                        "Label list has duplicate '{}'".format(text))
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
            # 保存标签排序结果
            self.saveLabelOrder()
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

            # 如果标签文本已经输入，尝试更新当前选中的标签项颜色
            current_text = self.edit.text().strip()
            if current_text:
                # 更新当前标签在列表视图中的颜色
                items = self.labelList.findItems(
                    current_text, QtCore.Qt.MatchFixedString)
                if items:
                    for item in items:
                        # 更新标签项样式
                        item.setBackground(QtGui.QBrush(QtGui.QColor(
                            color.red(), color.green(), color.blue(), 25)))  # 10%透明度
                        item.setData(QtCore.Qt.UserRole+1, color)

                # 更新当前标签在云布局中的颜色
                if hasattr(self, 'cloudContainer') and self.cloudContainer:
                    for label_item in self.cloudContainer.label_items:
                        # 检查是否为当前编辑的标签
                        if label_item.clean_text == current_text:
                            label_item.setLabelColor(color)
                            label_item.update()

                # 如果主应用程序存在，同步颜色到应用程序
                if self.app and hasattr(self.app, '_update_same_label_colors'):
                    clean_text = current_text.replace("●", "").strip()
                    if '<font' in clean_text:
                        clean_text = re.sub(
                            r'<[^>]*>|</[^>]*>', '', clean_text).strip()

                    # 更新主应用程序中的标签颜色映射
                    self.app._update_same_label_colors(clean_text, color)

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

    def saveLabelOrder(self):
        """保存当前标签排序顺序到应用程序"""
        if self.app and hasattr(self.app, 'save_label_order'):
            labels = []
            # 从列表视图或流式布局中获取标签顺序
            if not self._use_cloud_layout:
                # 从标准列表获取标签
                for i in range(self.labelList.count()):
                    item = self.labelList.item(i)
                    # 从标签文本中提取纯文本标签名
                    text = item.text()
                    clean_text = text.replace("●", "").strip()
                    if '<font' in clean_text:
                        clean_text = re.sub(
                            r'<[^>]*>|</[^>]*>', '', clean_text).strip()
                    labels.append(clean_text)
            else:
                # 从流式布局中获取标签 - 流式布局只在当前会话中生效
                for item in self.cloudContainer.label_items:
                    text = item.text
                    clean_text = text.replace("●", "").strip()
                    if '<font' in clean_text:
                        clean_text = re.sub(
                            r'<[^>]*>|</[^>]*>', '', clean_text).strip()
                    labels.append(clean_text)

            # 保存标签顺序 - 只在会话中保存，不写入配置文件
            try:
                if hasattr(self.app, 'updateLabelList'):
                    # 如果应用程序有更新标签列表的方法，直接调用
                    self.app.updateLabelList(labels)
                else:
                    # 否则调用基本的保存方法，但设置临时标志
                    self.app.save_label_order(labels, temporary=True)
            except Exception as e:
                logger.warning(f"无法更新标签顺序: {e}")

    def onLayoutToggleClicked(self):
        """处理布局切换按钮的点击事件"""
        self.toggleCloudLayout()

    def refreshLabelColors(self):
        """刷新所有标签项的颜色，确保与主应用程序中的颜色一致"""
        if not self.app:
            return

        # 刷新标准列表中的标签颜色
        for i in range(self.labelList.count()):
            item = self.labelList.item(i)
            if item:
                self._set_label_item_style(item, item.text())

        # 刷新流式布局中的标签颜色
        if hasattr(self, 'cloudContainer') and self.cloudContainer:
            for label_item in self.cloudContainer.label_items:
                clean_text = label_item.clean_text
                rgb_color = self.app._get_rgb_by_label(clean_text)
                if rgb_color:
                    label_item.setLabelColor(QtGui.QColor(*rgb_color))


class FlowLayout(QtWidgets.QLayout):
    """流式布局实现，自动将部件排列在一行，超出则换行"""

    def __init__(self, parent=None, margin=0, spacing=-1):
        super(FlowLayout, self).__init__(parent)
        if parent is not None:
            self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)
        self._items = []
        self._is_updating = False

    def __del__(self):
        while self.count():
            self.takeAt(0)

    def addItem(self, item):
        self._items.append(item)
        self.invalidate()  # 添加项后立即刷新布局

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):
        return QtCore.Qt.Orientations(QtCore.Qt.Orientation(0))

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        height = self._doLayout(QtCore.QRect(0, 0, width, 0), True)
        return height

    def setGeometry(self, rect):
        if self._is_updating:
            return

        self._is_updating = True
        super(FlowLayout, self).setGeometry(rect)
        self._doLayout(rect, False)
        self._is_updating = False

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QtCore.QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margin = self.contentsMargins().left() + self.contentsMargins().right()
        margin += self.contentsMargins().top() + self.contentsMargins().bottom()
        size += QtCore.QSize(margin, margin)
        return size

    def _doLayout(self, rect, testOnly):
        x = rect.x()
        y = rect.y()
        lineHeight = 0
        # 增加项之间的间距为12像素
        spaceX = max(self.spacing(), 12)
        spaceY = max(self.spacing(), 8)

        # 获取左右边距
        left = self.contentsMargins().left()
        right = self.contentsMargins().right()
        top = self.contentsMargins().top()
        bottom = self.contentsMargins().bottom()

        # 调整可用区域
        effectiveRect = QtCore.QRect(
            rect.x() + left,
            rect.y() + top,
            rect.width() - left - right,
            rect.height() - top - bottom
        )

        x = effectiveRect.x()
        y = effectiveRect.y()
        right_bound = effectiveRect.right()

        # 记录已放置的项以检测和避免重叠
        placed_rects = []

        for item in self._items:
            wid = item.widget()
            if wid and not wid.isVisible():
                continue  # 跳过不可见的小部件

            item_width = item.sizeHint().width()
            item_height = item.sizeHint().height()

            # 检查当前行是否还有足够空间
            nextX = x + item_width
            if nextX > right_bound and lineHeight > 0:
                # 如果不够空间，换行
                x = effectiveRect.x()
                y = y + lineHeight + spaceY
                nextX = x + item_width
                lineHeight = 0

            if not testOnly:
                # 创建当前项的放置矩形
                item_rect = QtCore.QRect(x, y, item_width, item_height)

                # 检查是否与已放置的项重叠
                overlaps = False
                for placed_rect in placed_rects:
                    if item_rect.intersects(placed_rect):
                        overlaps = True
                        break

                # 如果重叠，尝试调整位置
                if overlaps:
                    # 找到新的位置 - 移到下一行
                    x = effectiveRect.x()
                    y = y + lineHeight + spaceY
                    lineHeight = 0
                    item_rect = QtCore.QRect(x, y, item_width, item_height)

                # 记录放置位置
                placed_rects.append(item_rect)

                # 设置实际几何位置
                item.setGeometry(item_rect)

            # 更新位置和行高
            x = nextX + spaceX
            lineHeight = max(lineHeight, item_height)

        # 返回布局总高度
        return y + lineHeight - rect.y() + bottom


class LabelCloudItem(QtWidgets.QWidget):
    """流式布局中的标签项小部件"""

    # 定义自定义信号
    clicked = QtCore.pyqtSignal()
    doubleClicked = QtCore.pyqtSignal()
    dragStarted = QtCore.pyqtSignal(object)  # 发送自身引用

    def __init__(self, text, parent=None):
        super(LabelCloudItem, self).__init__(parent)
        self.text = text
        self.selected = False
        self.color = QtGui.QColor(0, 255, 0)  # 默认绿色
        self.dragging = False
        self.hover = False
        self.drop_hover = False  # 拖拽悬停状态
        self._drag_start_position = None  # 拖拽起始位置

        # 清理文本，移除HTML标记
        if '<font' in text:
            self.clean_text = re.sub(r'<[^>]*>●|</font>', '', text).strip()
        else:
            self.clean_text = text

        # 设置固定高度 - 增大高度
        self.setFixedHeight(52)  # 增加高度从44px到52px

        # 计算文本宽度并设置宽度 - 增大宽度
        fm = QtGui.QFontMetrics(self.font())
        text_width = fm.width(self.clean_text)
        self.setFixedWidth(text_width + 65)  # 增加宽度边距从45px到65px

        # 鼠标样式
        self.setCursor(QtCore.Qt.PointingHandCursor)

        # 启用鼠标跟踪以捕获悬停事件
        self.setMouseTracking(True)

        # 允许拖拽
        self.setAcceptDrops(True)

    def setLabelColor(self, color):
        """设置标签颜色"""
        self.color = color
        self.update()

    def paintEvent(self, event):
        """绘制标签项"""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # 圆角半径
        radius = 8

        # 标签区域 - 增加内边距
        rect = self.rect().adjusted(4, 4, -4, -4)  # 从2,2,-2,-2增加到4,4,-4,-4

        # 创建路径
        path = QtGui.QPainterPath()
        path.addRoundedRect(QtCore.QRectF(rect), radius, radius)

        # 绘制背景
        bg_color = QtGui.QColor(self.color)
        bg_color.setAlpha(25)  # 10%透明度
        painter.fillPath(path, bg_color)

        # 左边框宽度
        border_width = 14  # 从12px增加到14px

        # 绘制左边框
        border_path = QtGui.QPainterPath()
        border_rect = QtCore.QRectF(
            rect.left(),
            rect.top(),
            border_width,
            rect.height()
        )
        # 只对左边使用圆角
        border_path.addRoundedRect(border_rect, radius, radius)
        # 裁剪掉右边的圆角
        clip_path = QtGui.QPainterPath()
        clip_path.addRect(
            rect.left(),
            rect.top(),
            border_width / 2,  # 只显示左边的一半
            rect.height()
        )
        # 应用裁剪
        border_path = border_path.intersected(clip_path)

        # 填充左边框
        painter.fillPath(border_path, self.color)

        # 视觉效果增强：拖拽的目标位置显示接收指示
        if self.drop_hover:
            # 绘制更明显的接收指示边框
            drop_color = QtGui.QColor(0, 120, 215, 100)
            drop_pen = QtGui.QPen(drop_color, 2, QtCore.Qt.DashLine)
            painter.setPen(drop_pen)
            painter.drawRoundedRect(
                rect.adjusted(1, 1, -1, -1),
                radius, radius
            )

        # 选中状态或悬停状态高亮
        if self.selected:
            highlight_color = QtGui.QColor(0, 120, 215, 178)  # 70%透明度
            painter.fillPath(path, highlight_color)
            painter.setPen(QtGui.QColor(255, 255, 255))
        elif self.hover or self.dragging:
            hover_color = QtGui.QColor(0, 0, 0, 13)  # 5%透明度
            painter.fillPath(path, hover_color)
            painter.setPen(QtGui.QColor(0, 0, 0))
        else:
            painter.setPen(QtGui.QColor(0, 0, 0))

        # 文本区域 - 增加左边距
        text_rect = QtCore.QRect(
            rect.left() + border_width + 10,  # 左边框宽度 + 增加间距从6px到10px
            rect.top(),
            rect.width() - (border_width + 16),  # 增加右边距从12px到16px
            rect.height()
        )

        # 绘制文本
        # 设置字体
        font = painter.font()
        font.setPointSize(10)  # 确保字体大小合适
        painter.setFont(font)
        painter.drawText(text_rect, QtCore.Qt.AlignVCenter, self.clean_text)

    def mousePressEvent(self, event):
        """鼠标按下事件"""
        if event.button() == QtCore.Qt.LeftButton:
            self.selected = True
            self.update()
            self.clicked.emit()

            # 保存拖拽起始位置
            self._drag_start_position = event.pos()

    def mouseMoveEvent(self, event):
        """鼠标移动事件，处理拖拽"""
        if not (event.buttons() & QtCore.Qt.LeftButton) or self._drag_start_position is None:
            return

        # 计算移动距离，超过阈值则开始拖拽
        if (event.pos() - self._drag_start_position).manhattanLength() < QtWidgets.QApplication.startDragDistance():
            return

        # 开始拖拽
        self.dragging = True

        # 创建拖拽对象
        drag = QtGui.QDrag(self)

        # 设置拖拽的数据
        mime_data = QtCore.QMimeData()
        mime_data.setText(self.text)
        # 添加自定义数据以在拖放时识别
        mime_data.setData("application/x-labelcloud-item",
                          QtCore.QByteArray(b"1"))
        drag.setMimeData(mime_data)

        # 设置拖拽时的半透明预览图像
        pixmap = QtGui.QPixmap(self.size())
        pixmap.fill(QtCore.Qt.transparent)

        # 在pixmap上绘制当前小部件
        painter = QtGui.QPainter(pixmap)
        self.render(painter)
        painter.end()

        # 设置拖拽的图像和热点
        drag.setPixmap(pixmap)
        drag.setHotSpot(event.pos())

        # 通知父控件拖拽开始
        self.dragStarted.emit(self)

        # 执行拖拽
        result = drag.exec_(QtCore.Qt.MoveAction)

        # 拖拽结束，重置状态
        self.dragging = False
        self.selected = False  # 确保拖拽后不保持选中状态
        self._drag_start_position = None
        self.update()

    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        if event.button() == QtCore.Qt.LeftButton:
            # 如果不是拖放操作结束，则保持选中状态
            # 拖放操作的结束通过drag.exec_后的代码处理
            if not self.dragging:
                # 点击操作保持选中状态
                pass
            self.update()

    def mouseDoubleClickEvent(self, event):
        """鼠标双击事件"""
        if event.button() == QtCore.Qt.LeftButton:
            self.doubleClicked.emit()

    def enterEvent(self, event):
        """鼠标进入事件"""
        self.hover = True
        self.update()

    def leaveEvent(self, event):
        """鼠标离开事件"""
        self.hover = False
        self.update()

    def dragEnterEvent(self, event):
        """拖拽进入事件"""
        if event.mimeData().hasText() and event.mimeData().hasFormat("application/x-labelcloud-item"):
            event.acceptProposedAction()
            self.drop_hover = True
            self.update()

    def dragLeaveEvent(self, event):
        """拖拽离开事件"""
        event.accept()
        self.drop_hover = False
        self.update()

    def dropEvent(self, event):
        """放置事件"""
        if event.mimeData().hasText() and event.mimeData().hasFormat("application/x-labelcloud-item"):
            event.acceptProposedAction()
            self.drop_hover = False
            self.update()

            # 获取父控件（FlowLayout的容器）
            parent = self.parent()
            if parent and hasattr(parent, 'handleLabelDrop'):
                # 调用父控件的处理方法
                parent.handleLabelDrop(event.mimeData().text(), self)

            # 确保自身的选中状态被清除
            self.selected = False
            self.update()

    def sizeHint(self):
        """尺寸提示"""
        return QtCore.QSize(self.width(), self.height())


class LabelCloudContainer(QtWidgets.QWidget):
    """标签云容器，用于管理标签项的拖放操作"""

    def __init__(self, dialog):
        super(LabelCloudContainer, self).__init__()
        self.dialog = dialog
        self.setAcceptDrops(True)
        self.dragging_item = None
        self.label_items = []  # 保存所有标签项的引用
        self.update_timer = None  # 用于延迟更新的定时器

    def addLabelItem(self, item):
        """添加标签项到容器"""
        self.label_items.append(item)
        item.dragStarted.connect(self.onItemDragStarted)
        item.clicked.connect(lambda: self.clearSelectionExcept(item))

    def clearAllSelection(self):
        """清除所有标签项的选中状态"""
        for item in self.label_items:
            if item.selected:
                item.selected = False
                item.update()

    def clearSelectionExcept(self, current_item):
        """清除除当前项外的所有选中状态"""
        for item in self.label_items:
            if item != current_item and item.selected:
                item.selected = False
                item.update()

    def onItemDragStarted(self, item):
        """标签项开始拖拽时的处理"""
        self.dragging_item = item

    def handleLabelDrop(self, text, target_item):
        """处理标签项的放置"""
        if not self.dragging_item or self.dragging_item == target_item:
            return

        # 获取目标项和源项在布局中的索引
        layout = self.layout()
        if not isinstance(layout, FlowLayout):
            return

        # 找到拖拽项和目标项的索引
        source_index = -1
        target_index = -1

        for i in range(len(self.label_items)):
            if self.label_items[i] == self.dragging_item:
                source_index = i
            elif self.label_items[i] == target_item:
                target_index = i

        if source_index == -1 or target_index == -1:
            return

        # 移动项
        item = self.label_items.pop(source_index)
        self.label_items.insert(target_index, item)

        # 重新排列布局中的所有项
        self.updateLayout()

        # 重置拖拽状态
        self.dragging_item = None

        # 清除所有选中状态
        self.clearAllSelection()

        # 如果有保存标签顺序的方法，则调用
        if hasattr(self.dialog, 'saveLabelOrder'):
            self.dialog.saveLabelOrder()

    def updateLayout(self):
        """更新布局，根据标签项的新顺序重新排列"""
        layout = self.layout()
        if not isinstance(layout, FlowLayout):
            return

        # 清空布局
        while layout.count():
            item = layout.takeAt(0)
            # 不要删除小部件，只从布局中移除
            if item.widget():
                item.widget().setParent(None)

        # 重新添加所有项
        for item in self.label_items:
            layout.addWidget(item)

        # 强制更新布局 - 关键修复
        layout.invalidate()
        layout.activate()

        # 确保更新生效
        self.updateGeometry()
        self.parentWidget().updateGeometry()

        # 触发布局重计算
        self.adjustSize()
        self.update()

        # 使用计时器延迟再次更新，解决某些情况下第一次更新不完全的问题
        if self.update_timer is None:
            self.update_timer = QtCore.QTimer(self)
            self.update_timer.setSingleShot(True)
            self.update_timer.timeout.connect(self.delayedUpdate)

        self.update_timer.start(10)  # 10毫秒后再次更新

    def delayedUpdate(self):
        """延迟更新，确保布局正确显示"""
        layout = self.layout()
        if isinstance(layout, FlowLayout):
            layout.invalidate()
            layout.activate()

        # 触发滚动区域的更新
        scroll_area = self.parentWidget()
        if isinstance(scroll_area, QtWidgets.QScrollArea):
            scroll_area.updateGeometry()

        # 强制重绘
        self.update()

    def dragEnterEvent(self, event):
        """拖拽进入事件"""
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        """拖拽移动事件"""
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dropEvent(self, event):
        """放置事件 - 处理拖放到容器空白区域的情况"""
        if event.mimeData().hasText() and self.dragging_item:
            event.acceptProposedAction()

            # 将拖拽的项移动到末尾
            source_index = -1

            for i, item in enumerate(self.label_items):
                if item == self.dragging_item:
                    source_index = i
                    break

            if source_index != -1:
                item = self.label_items.pop(source_index)
                self.label_items.append(item)
                self.updateLayout()

            # 重置拖拽状态
            self.dragging_item = None

            # 清除所有选中状态
            self.clearAllSelection()

            # 如果有保存标签顺序的方法，则调用
            if hasattr(self.dialog, 'saveLabelOrder'):
                self.dialog.saveLabelOrder()

    def resizeEvent(self, event):
        """处理容器调整大小事件"""
        super(LabelCloudContainer, self).resizeEvent(event)
        # 容器大小改变时，重新计算流式布局
        layout = self.layout()
        if isinstance(layout, FlowLayout):
            layout.invalidate()
            # 适当延迟更新以确保完全重新布局
            if self.update_timer is None:
                self.update_timer = QtCore.QTimer(self)
                self.update_timer.setSingleShot(True)
                self.update_timer.timeout.connect(self.delayedUpdate)

            self.update_timer.start(10)
