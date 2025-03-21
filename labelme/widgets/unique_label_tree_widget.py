import html

from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QSize, QRect
from PyQt5.QtGui import QPalette, QStandardItem, QStandardItemModel, QColor, QBrush, QFont, QPainter, QPen
from PyQt5.QtWidgets import QStyle, QTreeView, QStyledItemDelegate, QWidget, QHBoxLayout, QAbstractItemView


class UniqueLabelCategoryDelegate(QStyledItemDelegate):
    """分类项的自定义代理，用于呈现更现代的分类项样式"""

    def __init__(self, parent=None, is_dark=False):
        super(UniqueLabelCategoryDelegate, self).__init__(parent)
        self.is_dark = is_dark

    def paint(self, painter, option, index):
        if not index.parent().isValid() and index.column() == 0:  # 顶层项(分类)
            option = QtWidgets.QStyleOptionViewItem(option)
            self.initStyleOption(option, index)

            painter.save()

            # 绘制背景
            if self.is_dark:
                if option.state & QStyle.State_Selected:
                    painter.fillRect(option.rect, QColor(45, 90, 120))
                elif option.state & QStyle.State_MouseOver:
                    painter.fillRect(option.rect, QColor(50, 50, 55))
                else:
                    painter.fillRect(option.rect, QColor(35, 35, 40))
            else:
                if option.state & QStyle.State_Selected:
                    painter.fillRect(option.rect, QColor(220, 235, 252))
                elif option.state & QStyle.State_MouseOver:
                    painter.fillRect(option.rect, QColor(240, 244, 249))
                else:
                    painter.fillRect(option.rect, QColor(248, 249, 250))

            # 获取图标
            icon = index.data(Qt.DecorationRole)
            if icon and not icon.isNull():
                icon_size = option.decorationSize
                icon_rect = QRect(option.rect.left() + 10,
                                  option.rect.top() + (option.rect.height() - icon_size.height()) // 2,
                                  icon_size.width(), icon_size.height())
                icon.paint(painter, icon_rect, Qt.AlignCenter)
                text_left = icon_rect.right() + 10
            else:
                text_left = option.rect.left() + 10

            # 绘制文本
            text = index.data(Qt.DisplayRole)
            if isinstance(text, str):
                # 分解文本提取分类名和计数
                if "(" in text:
                    try:
                        parts = text.rsplit(" (", 1)
                        category_name = parts[0]
                        count = parts[1][:-1]  # 移除右括号
                    except:
                        category_name = text
                        count = ""
                else:
                    category_name = text
                    count = ""

                # 设置文本颜色
                if self.is_dark:
                    text_color = "#ffffff" if option.state & QStyle.State_Selected else "#e0e0e0"
                    count_color = "#aaaaaa"
                else:
                    text_color = "#000000" if option.state & QStyle.State_Selected else "#333333"
                    count_color = "#777777"

                # 绘制分类名
                font = painter.font()
                font.setBold(True)
                font.setPointSize(10)
                painter.setFont(font)
                painter.setPen(QColor(text_color))
                text_rect = QRect(text_left, option.rect.top(),
                                  option.rect.width() - text_left - 50, option.rect.height())
                painter.drawText(text_rect, Qt.AlignVCenter, category_name)

                # 绘制计数
                if count:
                    font.setBold(False)
                    font.setPointSize(9)
                    painter.setFont(font)
                    painter.setPen(QColor(count_color))
                    count_rect = QRect(text_rect.right(), option.rect.top(),
                                       40, option.rect.height())
                    painter.drawText(count_rect, Qt.AlignVCenter, f"({count})")

            painter.restore()
        else:
            # 非分类项使用默认绘制方法
            super(UniqueLabelCategoryDelegate, self).paint(
                painter, option, index)

    def sizeHint(self, option, index):
        if not index.parent().isValid():  # 分类项
            return QSize(option.rect.width(), 38)  # 分类项高度增加
        else:
            return super(UniqueLabelCategoryDelegate, self).sizeHint(option, index)


class UniqueLabelItemDelegate(QStyledItemDelegate):
    """子项的自定义代理，用于呈现更现代的子项样式"""

    def __init__(self, parent=None, is_dark=False):
        super(UniqueLabelItemDelegate, self).__init__(parent)
        self.is_dark = is_dark
        self.doc = QtGui.QTextDocument(self)

    def paint(self, painter, option, index):
        if index.parent().isValid() and index.column() == 0:  # 子项
            option = QtWidgets.QStyleOptionViewItem(option)
            self.initStyleOption(option, index)

            painter.save()

            # 获取自定义样式
            custom_style = index.data(Qt.UserRole + 10)
            is_unused = custom_style and "font-style: italic" in custom_style

            # 提取颜色信息
            text = index.data(Qt.DisplayRole)
            label_color = None
            if "<font" in text:
                import re
                # 提取颜色信息
                color_match = re.search(r'color=[\'"]([^\'"]*)[\'"]', text)
                if color_match:
                    label_color = QColor(color_match.group(1))

            # 绘制背景
            if self.is_dark:
                if option.state & QStyle.State_Selected:
                    if is_unused:
                        # 未使用的标签被选中时，背景稍微加深，其余与未被选中时的样式一致
                        painter.fillRect(option.rect, QColor(100, 40, 40, 70))
                        # 绘制左边框标记
                        mark_color = QColor(255, 109, 109)
                        painter.setPen(Qt.NoPen)
                        painter.setBrush(QBrush(mark_color))
                        # 左侧标记宽度，并使用圆角矩形
                        border_width = 8
                        painter.drawRoundedRect(
                            option.rect.left(),
                            option.rect.top() + 2,
                            border_width,
                            option.rect.height() - 4,
                            3, 3
                        )
                    elif label_color:
                        # 使用标签颜色的10%透明度作为背景
                        bg_color = QColor(label_color)
                        bg_color.setAlpha(25)  # 10%透明度
                        painter.fillRect(option.rect, bg_color)
                    else:
                        painter.fillRect(option.rect, QColor(0, 120, 212))
                elif is_unused:
                    # 未使用标签的暗色主题背景
                    painter.fillRect(option.rect, QColor(80, 30, 30, 50))
                    # 绘制左边框标记
                    mark_color = QColor(255, 109, 109)
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(QBrush(mark_color))
                    # 左侧标记宽度，并使用圆角矩形
                    border_width = 8
                    painter.drawRoundedRect(
                        option.rect.left(),
                        option.rect.top() + 2,
                        border_width,
                        option.rect.height() - 4,
                        3, 3
                    )
                elif option.state & QStyle.State_MouseOver:
                    painter.fillRect(option.rect, QColor(60, 60, 65))
                else:
                    painter.fillRect(
                        option.rect, QColor(40, 40, 45, 0))  # 透明背景
            else:
                if option.state & QStyle.State_Selected:
                    if is_unused:
                        # 未使用的标签被选中时，背景稍微加深，其余与未被选中时的样式一致
                        painter.fillRect(option.rect, QColor(255, 230, 230))
                        # 绘制左边框标记
                        mark_color = QColor(255, 109, 109)
                        painter.setPen(Qt.NoPen)
                        painter.setBrush(QBrush(mark_color))
                        # 左侧标记宽度，并使用圆角矩形
                        border_width = 8
                        painter.drawRoundedRect(
                            option.rect.left(),
                            option.rect.top() + 2,
                            border_width,
                            option.rect.height() - 4,
                            3, 3
                        )
                    elif label_color:
                        # 使用标签颜色的10%透明度作为背景
                        bg_color = QColor(label_color)
                        bg_color.setAlpha(25)  # 10%透明度
                        painter.fillRect(option.rect, bg_color)
                    else:
                        painter.fillRect(option.rect, QColor(210, 228, 255))
                elif is_unused:
                    # 未使用标签的亮色主题背景
                    painter.fillRect(option.rect, QColor(255, 240, 240))
                    # 绘制左边框标记
                    mark_color = QColor(255, 109, 109)
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(QBrush(mark_color))
                    # 左侧标记宽度，并使用圆角矩形
                    border_width = 8
                    painter.drawRoundedRect(
                        option.rect.left(),
                        option.rect.top() + 2,
                        border_width,
                        option.rect.height() - 4,
                        3, 3
                    )
                elif option.state & QStyle.State_MouseOver:
                    painter.fillRect(option.rect, QColor(235, 243, 254))
                else:
                    painter.fillRect(option.rect, QColor(
                        255, 255, 255, 0))  # 透明背景

            # 绘制文本和颜色指示器
            if text:
                # 设置默认文本颜色
                if self.is_dark:
                    text_color = QColor(
                        255, 255, 255) if option.state & QStyle.State_Selected else QColor(220, 220, 220)
                else:
                    text_color = QColor(
                        0, 0, 0) if option.state & QStyle.State_Selected else QColor(60, 60, 60)

                # 如果是未使用标签，设置特殊颜色
                if is_unused and not option.state & QStyle.State_Selected:
                    text_color = QColor(255, 109, 109)

                # 提取纯文本内容
                if "<font" in text:
                    import re
                    # 提取纯文本内容，同时移除 "●" 字符和</font>标签
                    content = re.sub(r'<[^>]*>●|</font>', '', text).strip()
                else:
                    content = text

                # 设置字体
                font = painter.font()
                font.setFamily("Microsoft YaHei")
                font.setPointSize(10)  # 保持现有大小
                font.setBold(False)
                painter.setFont(font)
                painter.setPen(text_color)

                # 设置文本起始位置
                text_left = option.rect.left() + 15

                # 如果是未使用标签，增加额外的左侧padding
                if is_unused:
                    text_left += 5  # 增加额外的padding，与边框宽度相协调

                # 绘制文本，直接使用纯文本内容
                text_rect = QRect(
                    text_left,
                    option.rect.top(),
                    option.rect.width() - text_left - 40,  # 留出右侧空间给颜色指示器
                    option.rect.height()
                )

                painter.drawText(text_rect, Qt.AlignVCenter, content)

                # 在文本右侧绘制颜色指示器
                if label_color:
                    # 计算颜色指示器位置 - 放在文本右侧固定距离处
                    content_width = option.fontMetrics.width(content)
                    color_rect = QRect(
                        text_left + content_width + 25,  # 文本宽度加间距
                        option.rect.top() + (option.rect.height() - 20) // 2,  # 垂直居中
                        20,  # 颜色圆点大小
                        20   # 颜色圆点大小
                    )

                    # 绘制颜色圆点
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(QBrush(label_color))
                    painter.drawEllipse(color_rect)

                    # 添加细边框使颜色圆点更加清晰
                    painter.setPen(QPen(QColor(200, 200, 200, 100), 1.0))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawEllipse(color_rect)

            painter.restore()
        else:
            # 非子项使用默认绘制方法
            super(UniqueLabelItemDelegate, self).paint(painter, option, index)

    def sizeHint(self, option, index):
        if index.parent().isValid():  # 子项
            # 确保有足够大的高度
            return QSize(option.rect.width(), 42)
        else:
            return super(UniqueLabelItemDelegate, self).sizeHint(option, index)


class ModernUniqueTreeView(QTreeView):
    """现代化的树视图控件，样式类似文件资源管理器"""

    def __init__(self, parent=None, is_dark=False):
        super(ModernUniqueTreeView, self).__init__(parent)
        self.is_dark = is_dark
        self.setRootIsDecorated(True)  # 显示根项的展开/折叠控件
        self.setItemsExpandable(True)
        self.setAnimated(True)
        self.setIndentation(20)
        self.setUniformRowHeights(False)  # 允许不同高度的行
        self.setIconSize(QSize(20, 20))
        self.setHeaderHidden(True)
        self.setAlternatingRowColors(False)
        self.setSelectionMode(QAbstractItemView.SingleSelection)  # 保持原有的单选模式
        self.setDragEnabled(False)  # 不需要拖拽功能
        self.setFocusPolicy(Qt.StrongFocus)

        # 启用平滑滚动
        self.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)

        # 设置代理
        self.category_delegate = UniqueLabelCategoryDelegate(self, is_dark)
        self.item_delegate = UniqueLabelItemDelegate(self, is_dark)
        self.setItemDelegate(self.category_delegate)
        self.setItemDelegateForColumn(0, self.item_delegate)

        # 设置样式
        self.setThemeStyleSheet()

    def setThemeStyleSheet(self):
        """根据主题设置样式表"""
        if self.is_dark:
            self.setStyleSheet("""
                QTreeView {
                    background-color: #252526;
                    border: none;
                    outline: none;
                    padding: 5px;
                    selection-background-color: transparent;
                }
                QTreeView::item {
                    border-radius: 4px;
                    margin: 2px 4px;
                    padding: 4px;
                    font-size: 14px;
                    font-family: "Microsoft YaHei UI", "Segoe UI", Arial, sans-serif;
                }
                QTreeView::branch {
                    background-color: transparent;
                }
                QTreeView::branch:has-children:!has-siblings:closed,
                QTreeView::branch:closed:has-children:has-siblings {
                    image: url(:/icons/right-arrow-dark.png);
                }
                QTreeView::branch:open:has-children:!has-siblings,
                QTreeView::branch:open:has-children:has-siblings {
                    image: url(:/icons/down-arrow-dark.png);
                }
                QScrollBar:vertical {
                    background-color: #2a2a2b;
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
                QScrollBar:horizontal {
                    background-color: #2a2a2b;
                    height: 8px;
                    margin: 0px;
                }
                QScrollBar::handle:horizontal {
                    background-color: #5a5a5c;
                    min-width: 30px;
                    border-radius: 4px;
                }
                QScrollBar::handle:horizontal:hover {
                    background-color: #777779;
                }
                QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
                QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                    background: none;
                    width: 0px;
                }
            """)
        else:
            self.setStyleSheet("""
                QTreeView {
                    background-color: white;
                    border: none;
                    outline: none;
                    padding: 5px;
                    selection-background-color: transparent;
                }
                QTreeView::item {
                    border-radius: 4px;
                    margin: 2px 4px;
                    padding: 4px;
                    font-size: 14px;
                    font-family: "Microsoft YaHei UI", "Segoe UI", Arial, sans-serif;
                }
                QTreeView::branch {
                    background-color: transparent;
                }
                QTreeView::branch:has-children:!has-siblings:closed,
                QTreeView::branch:closed:has-children:has-siblings {
                    image: url(:/icons/right-arrow.png);
                }
                QTreeView::branch:open:has-children:!has-siblings,
                QTreeView::branch:open:has-children:has-siblings {
                    image: url(:/icons/down-arrow.png);
                }
                QScrollBar:vertical {
                    background-color: #f6f6f6;
                    width: 8px;
                    margin: 0px;
                }
                QScrollBar::handle:vertical {
                    background-color: #d0d0d0;
                    min-height: 30px;
                    border-radius: 4px;
                }
                QScrollBar::handle:vertical:hover {
                    background-color: #b0b0b0;
                }
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
                QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                    background: none;
                    height: 0px;
                }
                QScrollBar:horizontal {
                    background-color: #f6f6f6;
                    height: 8px;
                    margin: 0px;
                }
                QScrollBar::handle:horizontal {
                    background-color: #d0d0d0;
                    min-width: 30px;
                    border-radius: 4px;
                }
                QScrollBar::handle:horizontal:hover {
                    background-color: #b0b0b0;
                }
                QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
                QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                    background: none;
                    width: 0px;
                }
            """)

    def drawBranches(self, painter, rect, index):
        """自定义分支绘制，使用更现代的展开/折叠图标"""
        if not index.parent().isValid():  # 顶层项
            item = self.model().itemFromIndex(index)
            if item and hasattr(item, 'rowCount') and item.rowCount() > 0:
                painter.save()

                # 绘制展开/折叠图标 - 改用圆形背景加箭头的现代样式
                mid_y = rect.top() + rect.height() // 2
                right_x = rect.left() + rect.width() - 10

                # 绘制圆形背景
                if self.is_dark:
                    bg_color = QColor(60, 60, 66)
                    arrow_color = QColor(220, 220, 220)
                else:
                    bg_color = QColor(240, 240, 245)
                    arrow_color = QColor(100, 100, 100)

                # 圆形背景
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(bg_color))
                painter.drawEllipse(right_x - 8, mid_y - 8, 16, 16)

                # 箭头
                painter.setPen(
                    QPen(arrow_color, 2.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))

                if self.isExpanded(index):
                    # 绘制展开图标（向下箭头）
                    painter.drawLine(right_x - 4, mid_y -
                                     1, right_x, mid_y + 3)
                    painter.drawLine(right_x, mid_y + 3,
                                     right_x + 4, mid_y - 1)
                else:
                    # 绘制折叠图标（向右箭头）
                    painter.drawLine(right_x - 2, mid_y -
                                     4, right_x + 3, mid_y)
                    painter.drawLine(right_x + 3, mid_y,
                                     right_x - 2, mid_y + 4)

                painter.restore()
        # 不绘制子项的分支线

    def setDarkMode(self, is_dark):
        """切换暗色/亮色主题"""
        self.is_dark = is_dark
        self.category_delegate.is_dark = is_dark
        self.item_delegate.is_dark = is_dark
        self.setThemeStyleSheet()
        self.viewport().update()


class UniqueLabelTreeWidgetItem(QStandardItem):
    def __init__(self, text=None, label=None, is_category=False, shape_type=None, is_dark=False):
        super(UniqueLabelTreeWidgetItem, self).__init__()
        self.setText(text or "")
        self.setLabel(label)
        self.is_category = is_category
        self.shape_type = shape_type
        self.is_dark = is_dark

        # 存储是否为分类项的标志
        self.setData(is_category, Qt.UserRole + 2)

        self.setEditable(False)
        self.setSelectable(True)

    def setLabel(self, label):
        self.setData(label, Qt.UserRole)

    def label(self):
        return self.data(Qt.UserRole)

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return '{}("{}")'.format(self.__class__.__name__, self.text())


class UniqueLabelTreeWidget(QWidget):
    itemDoubleClicked = QtCore.pyqtSignal(UniqueLabelTreeWidgetItem)
    itemSelectionChanged = QtCore.pyqtSignal(str, str)  # 新增信号：标签名称, 形状类型

    def __init__(self, is_dark=False):
        super(UniqueLabelTreeWidget, self).__init__()
        self.categories = {}  # 存储所有分类
        self.labels_by_category = {}  # 按类别存储标签
        self.is_dark = is_dark
        self.unused_labels = set()  # 新增：存储未使用标签的集合

        # 创建现代化的树状视图
        self.treeView = ModernUniqueTreeView(self, is_dark)
        self.model = QStandardItemModel()
        self.treeView.setModel(self.model)

        # 连接信号
        self.treeView.doubleClicked.connect(self.itemDoubleClickedEvent)
        # 连接选择变化信号
        self.treeView.selectionModel().selectionChanged.connect(
            self.itemSelectionChangedEvent)

        # 设置布局
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.treeView)
        self.setLayout(layout)

        # 设置窗口样式
        if self.is_dark:
            self.setStyleSheet("background-color: #252526; border: none;")
        else:
            self.setStyleSheet("background-color: white; border: none;")

    def setDarkMode(self, is_dark):
        """切换暗色/亮色主题"""
        self.is_dark = is_dark
        self.treeView.setDarkMode(is_dark)

        # 更新窗口样式
        if self.is_dark:
            self.setStyleSheet("background-color: #252526; border: none;")
        else:
            self.setStyleSheet("background-color: white; border: none;")

        # 更新现有项的样式
        self.updateAllItemsTheme()

    def updateAllItemsTheme(self):
        """更新所有项的主题样式"""
        for category_name, category_item in self.categories.items():
            category_item.is_dark = self.is_dark

            # 更新分类下的所有子项
            for i in range(category_item.rowCount()):
                child_item = category_item.child(i, 0)
                if isinstance(child_item, UniqueLabelTreeWidgetItem):
                    child_item.is_dark = self.is_dark

        # 更新分类数量显示
        self.updateAllCategoryCounts()
        # 刷新视图
        self.treeView.viewport().update()

    def count(self):
        # 计算所有标签项的数量（不包括分类项）
        count = 0
        for i in range(self.model.rowCount()):
            category_item = self.model.item(i, 0)
            count += category_item.rowCount()
        return count

    def itemDoubleClickedEvent(self, index):
        item = self.model.itemFromIndex(index)
        # 确保项是UniqueLabelTreeWidgetItem类型并且不是分类
        if isinstance(item, UniqueLabelTreeWidgetItem) and not item.is_category:
            self.itemDoubleClicked.emit(item)

    def findItemByLabel(self, label):
        """根据标签查找项"""
        for i in range(self.model.rowCount()):
            category_item = self.model.item(i, 0)
            for j in range(category_item.rowCount()):
                item = category_item.child(j, 0)
                if item and item.label() == label:
                    return item
        return None

    def getCategoryItem(self, category_name):
        """获取或创建分类项"""
        if category_name in self.categories:
            return self.categories[category_name]

        # 创建新分类
        category_item = UniqueLabelTreeWidgetItem(
            category_name, None, True, None, self.is_dark)
        self.model.appendRow(category_item)
        self.categories[category_name] = category_item
        if category_name not in self.labels_by_category:
            self.labels_by_category[category_name] = []
        self.treeView.expand(category_item.index())  # 展开分类项
        return category_item

    def createItemFromLabel(self, label, shape_type=None):
        """从标签创建项"""
        if self.findItemByLabel(label):
            raise ValueError(f"Item for label '{label}' already exists")

        item = UniqueLabelTreeWidgetItem(
            label=label, shape_type=shape_type, is_dark=self.is_dark)
        return item

    def addItem(self, item):
        """添加项到对应的分类中"""
        if not isinstance(item, UniqueLabelTreeWidgetItem):
            raise TypeError("item must be UniqueLabelTreeWidgetItem")

        # 设置项的颜色主题
        item.is_dark = self.is_dark

        label = item.label()
        if not label:
            return

        # 从app中获取该标签对应的形状类型
        # 暂时使用默认分类"标签"
        category_name = "标签"

        # 如果item有shape_type属性，则按照shape_type分类
        shape_type = getattr(item, 'shape_type', None)
        if shape_type:
            shape_type_names = {
                "polygon": "多边形",
                "rectangle": "矩形",
                "circle": "圆形",
                "line": "线段",
                "point": "点",
                "linestrip": "折线",
                "points": "多点",
                "mask": "蒙版"
            }
            category_name = shape_type_names.get(shape_type, shape_type)

        # 获取或创建分类项
        category_item = self.getCategoryItem(category_name)

        # 添加到分类下
        category_item.appendRow(item)
        if label not in self.labels_by_category[category_name]:
            self.labels_by_category[category_name].append(label)

        self.treeView.expand(category_item.index())  # 确保展开显示
        self.updateCategoryCount(category_name)

    def setItemLabel(self, item, label, color=None):
        """设置项的标签和颜色"""
        if color is None:
            item.setText(f"{label}")
        else:
            item.setText(
                '{} <font color="#{:02x}{:02x}{:02x}">●</font>'.format(
                    html.escape(label), *color
                )
            )

    def selectedItems(self):
        """获取选中的项"""
        items = []
        for index in self.treeView.selectedIndexes():
            item = self.model.itemFromIndex(index)
            # 确保项是UniqueLabelTreeWidgetItem类型并且不是分类
            if isinstance(item, UniqueLabelTreeWidgetItem) and not item.is_category:
                items.append(item)
        return items

    def clear(self):
        """清空所有项"""
        self.model.clear()
        self.categories = {}
        self.labels_by_category = {}

    def updateCategoryCount(self, category_name):
        """更新分类数量显示"""
        if category_name in self.categories:
            category_item = self.categories[category_name]
            count = category_item.rowCount()
            category_item.setText(f"{category_name} ({count})")

    def updateAllCategoryCounts(self):
        """更新所有分类的数量显示"""
        for category_name in list(self.categories.keys()):
            self.updateCategoryCount(category_name)

    def expandAll(self):
        """展开所有项"""
        self.treeView.expandAll()

    def highlightUnusedLabels(self, label_tree_widget):
        """
        检查哪些标签未在当前图片中使用，并对其进行强调显示

        Args:
            label_tree_widget: LabelTreeWidget实例，包含当前图片中已使用的标签
        """
        # 重置之前的未使用标签记录
        self.unused_labels.clear()

        # 获取当前图片中使用的所有标签
        used_labels = set()
        for i in range(label_tree_widget.model.rowCount()):
            category_item = label_tree_widget.model.item(i, 0)
            for j in range(category_item.rowCount()):
                item = category_item.child(j, 0)
                if item and hasattr(item, 'shape') and item.shape():
                    shape = item.shape()
                    if hasattr(shape, 'label'):
                        used_labels.add(shape.label)

        # 检查每个可用标签是否被使用
        unused_style = """
            font-style: italic;
            font-weight: bold;
            color: #FF6D6D;
            border-radius: 6px;
            border-left: 8px solid #FF6D6D;
            background-color: rgba(255, 109, 109, 0.1);
            padding-left: 8px;
            margin: 2px 0px;
        """

        used_style = ""

        # 遍历所有标签项，检查它们是否被使用
        for i in range(self.model.rowCount()):
            category_item = self.model.item(i, 0)
            for j in range(category_item.rowCount()):
                item = category_item.child(j, 0)
                if item and hasattr(item, 'label'):
                    label = item.label()
                    if label not in used_labels:
                        # 未使用的标签，添加到集合并应用强调样式
                        self.unused_labels.add(label)
                        item.setData(unused_style, Qt.UserRole + 10)  # 保存样式数据
                    else:
                        # 已使用的标签，恢复正常样式
                        item.setData(used_style, Qt.UserRole + 10)

        # 更新视图
        self.treeView.viewport().update()

    def resetHighlights(self):
        """重置所有标签的高亮状态"""
        self.unused_labels.clear()

        for i in range(self.model.rowCount()):
            category_item = self.model.item(i, 0)
            for j in range(category_item.rowCount()):
                item = category_item.child(j, 0)
                if item:
                    item.setData("", Qt.UserRole + 10)  # 清除样式数据

        # 更新视图
        self.treeView.viewport().update()

    def itemSelectionChangedEvent(self, selected, deselected):
        """处理项选择变化的事件"""
        # 获取当前选择的项
        selected_items = self.selectedItems()
        if selected_items:
            item = selected_items[0]  # 获取第一个选中项（单选模式）
            label = item.label()
            if not label:
                return

            # 获取形状类型（如果有）
            shape_type = getattr(item, 'shape_type', None)

            # 如果项本身没有形状类型，则从其所属分类推断
            if not shape_type:
                # 获取父项（分类）
                parent = item.parent()
                if parent:
                    category_name = parent.text()
                    if "(" in category_name:
                        category_name = category_name.split(" (")[0]

                    # 中文分类名转换为英文形状类型
                    shape_type_mapping = {
                        "多边形": "polygon",
                        "矩形": "rectangle",
                        "圆形": "circle",
                        "线段": "line",
                        "点": "point",
                        "折线": "linestrip",
                        "多点": "points",
                        "蒙版": "mask",
                        "标签": "polygon"  # 默认为多边形
                    }
                    shape_type = shape_type_mapping.get(
                        category_name, "polygon")

            # 发送信号
            if label and shape_type:
                self.itemSelectionChanged.emit(label, shape_type)
