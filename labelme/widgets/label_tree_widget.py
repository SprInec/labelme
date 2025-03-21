from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QSize, QRect, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QPalette, QStandardItem, QStandardItemModel, QColor, QBrush, QFont, QPainter, QPen, QPainterPath, QPixmap, QIcon
from PyQt5.QtWidgets import QStyle, QTreeView, QStyleFactory, QProxyStyle, QStyledItemDelegate, QWidget, QLabel, QHBoxLayout, QAbstractItemView

import html


class LabelTreeCategoryDelegate(QStyledItemDelegate):
    """分类项的自定义代理，用于呈现更现代的分类项样式"""

    def __init__(self, parent=None, is_dark=False):
        super(LabelTreeCategoryDelegate, self).__init__(parent)
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
            super(LabelTreeCategoryDelegate, self).paint(
                painter, option, index)

    def sizeHint(self, option, index):
        if not index.parent().isValid():  # 分类项
            return QSize(option.rect.width(), 38)  # 分类项高度增加
        else:
            return super(LabelTreeCategoryDelegate, self).sizeHint(option, index)


class LabelTreeItemDelegate(QStyledItemDelegate):
    """子项的自定义代理，用于呈现更现代的子项样式"""

    def __init__(self, parent=None, is_dark=False):
        super(LabelTreeItemDelegate, self).__init__(parent)
        self.is_dark = is_dark
        self.doc = QtGui.QTextDocument(self)

    def paint(self, painter, option, index):
        if index.parent().isValid() and index.column() == 0:  # 子项
            option = QtWidgets.QStyleOptionViewItem(option)
            self.initStyleOption(option, index)

            painter.save()

            # 绘制背景
            if self.is_dark:
                if option.state & QStyle.State_Selected:
                    painter.fillRect(option.rect, QColor(0, 120, 212))
                elif option.state & QStyle.State_MouseOver:
                    painter.fillRect(option.rect, QColor(60, 60, 65))
                else:
                    painter.fillRect(
                        option.rect, QColor(40, 40, 45, 0))  # 透明背景
            else:
                if option.state & QStyle.State_Selected:
                    painter.fillRect(option.rect, QColor(210, 228, 255))
                elif option.state & QStyle.State_MouseOver:
                    painter.fillRect(option.rect, QColor(235, 243, 254))
                else:
                    painter.fillRect(option.rect, QColor(
                        255, 255, 255, 0))  # 透明背景

            # 绘制复选框
            check_state = index.data(Qt.CheckStateRole)
            if check_state is not None:
                checkbox_style = QtWidgets.QStyleOptionButton()
                checkbox_style.rect = QRect(
                    option.rect.left() + 8,
                    option.rect.top() + (option.rect.height() - 28) // 2,
                    28, 28  # 稍微增大复选框
                )

                if self.is_dark:
                    if check_state == Qt.Checked:
                        # 绘制选中状态
                        painter.setPen(Qt.NoPen)
                        painter.setBrush(QBrush(QColor(0, 120, 212)))
                        painter.drawRoundedRect(checkbox_style.rect, 5, 5)

                        # 绘制居中对勾
                        painter.setPen(QPen(QColor(255, 255, 255), 3))
                        # 计算对勾的中心点
                        center_x = checkbox_style.rect.left() + checkbox_style.rect.width() // 2
                        center_y = checkbox_style.rect.top() + checkbox_style.rect.height() // 2

                        # 绘制对勾的两条线段，以中心点为基准
                        painter.drawLine(
                            center_x - 5,
                            center_y + 2,
                            center_x - 1,
                            center_y + 6
                        )
                        painter.drawLine(
                            center_x - 1,
                            center_y + 6,
                            center_x + 7,
                            center_y - 4
                        )
                    else:
                        # 绘制未选中状态
                        painter.setPen(QPen(QColor(150, 150, 150), 1.5))
                        painter.setBrush(Qt.NoBrush)
                        painter.drawRoundedRect(checkbox_style.rect, 5, 5)
                else:
                    if check_state == Qt.Checked:
                        # 绘制选中状态
                        painter.setPen(Qt.NoPen)
                        painter.setBrush(QBrush(QColor(0, 120, 212)))
                        painter.drawRoundedRect(checkbox_style.rect, 5, 5)

                        # 绘制居中对勾
                        painter.setPen(QPen(QColor(255, 255, 255), 2.5))
                        # 计算对勾的中心点
                        center_x = checkbox_style.rect.left() + checkbox_style.rect.width() // 2
                        center_y = checkbox_style.rect.top() + checkbox_style.rect.height() // 2

                        # 绘制对勾的两条线段，以中心点为基准
                        painter.drawLine(
                            center_x - 5,
                            center_y + 2,
                            center_x - 1,
                            center_y + 6
                        )
                        painter.drawLine(
                            center_x - 1,
                            center_y + 6,
                            center_x + 7,
                            center_y - 4
                        )
                    else:
                        # 绘制未选中状态
                        painter.setPen(QPen(QColor(170, 170, 170), 1.5))
                        painter.setBrush(Qt.NoBrush)
                        painter.drawRoundedRect(checkbox_style.rect, 5, 5)

                text_left = checkbox_style.rect.right() + 15  # 增加更多间距
            else:
                text_left = option.rect.left() + 8

            # 绘制文本和颜色指示器
            text = index.data(Qt.DisplayRole)
            if text:
                # 设置默认文本颜色
                if self.is_dark:
                    text_color = QColor(
                        255, 255, 255) if option.state & QStyle.State_Selected else QColor(220, 220, 220)
                else:
                    text_color = QColor(
                        0, 0, 0) if option.state & QStyle.State_Selected else QColor(60, 60, 60)

                # 提取颜色信息和文本内容
                label_color = None
                if "<font" in text:
                    import re
                    # 提取颜色信息
                    color_match = re.search(r'color=[\'"]([^\'"]*)[\'"]', text)
                    if color_match:
                        label_color = QColor(color_match.group(1))

                    # 提取纯文本内容，同时移除 "●" 字符和</font>标签
                    content = re.sub(r'<[^>]*>●|</font>', '', text).strip()
                else:
                    content = text

                # 绘制标签名称 - 保持你的字体设置不变
                font = painter.font()
                font.setFamily("Microsoft YaHei")
                font.setPointSize(10)  # 保持现有大小
                font.setBold(False)
                painter.setFont(font)
                painter.setPen(text_color)

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
                        20,  # 颜色圆点大小，保持现有大小
                        20   # 颜色圆点大小，保持现有大小
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
            super(LabelTreeItemDelegate, self).paint(painter, option, index)

    def sizeHint(self, option, index):
        if index.parent().isValid():  # 子项
            # 确保有足够大的高度
            return QSize(option.rect.width(), 48)
        else:
            return super(LabelTreeItemDelegate, self).sizeHint(option, index)


class ModernTreeView(QTreeView):
    """现代化的树视图控件，样式类似文件资源管理器"""

    def __init__(self, parent=None, is_dark=False):
        super(ModernTreeView, self).__init__(parent)
        self.is_dark = is_dark
        self.setRootIsDecorated(True)  # 显示根项的展开/折叠控件
        self.setItemsExpandable(True)
        self.setAnimated(True)
        self.setIndentation(20)
        self.setUniformRowHeights(False)  # 允许不同高度的行
        self.setIconSize(QSize(20, 20))
        self.setHeaderHidden(True)
        self.setAlternatingRowColors(False)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setFocusPolicy(Qt.StrongFocus)

        # 启用平滑滚动
        self.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)

        # 设置代理
        self.category_delegate = LabelTreeCategoryDelegate(self, is_dark)
        self.item_delegate = LabelTreeItemDelegate(self, is_dark)
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


class LabelTreeWidgetItem(QStandardItem):
    def __init__(self, text=None, shape=None, is_category=False, is_dark=False):
        super(LabelTreeWidgetItem, self).__init__()
        self.setText(text or "")
        self.setShape(shape)
        self.is_category = is_category
        self.is_dark = is_dark

        # 存储是否为分类项的标志
        self.setData(is_category, Qt.UserRole + 2)

        # 如果不是分类项，设置可选中
        if not is_category and shape:
            self.setCheckable(True)
            self.setCheckState(
                Qt.Checked if shape.isVisible() else Qt.Unchecked)

        self.setEditable(False)
        self.setSelectable(True)

    def clone(self):
        return LabelTreeWidgetItem(self.text(), self.shape(), self.is_category, self.is_dark)

    def setShape(self, shape):
        self.setData(shape, Qt.UserRole)

    def shape(self):
        return self.data(Qt.UserRole)

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return '{}("{}")'.format(self.__class__.__name__, self.text())


class LabelTreeWidget(QWidget):
    itemDoubleClicked = QtCore.pyqtSignal(LabelTreeWidgetItem)
    itemSelectionChanged = QtCore.pyqtSignal(list, list)

    def __init__(self, is_dark=False):
        super(LabelTreeWidget, self).__init__()
        self._selectedItems = []
        self.categories = {}  # 存储所有分类
        self.is_dark = is_dark

        # 创建现代化的树状视图
        self.treeView = ModernTreeView(self, is_dark)
        self.model = QStandardItemModel()
        self.treeView.setModel(self.model)

        # 连接信号
        self.treeView.doubleClicked.connect(self.itemDoubleClickedEvent)
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
                if isinstance(child_item, LabelTreeWidgetItem):
                    child_item.is_dark = self.is_dark

        # 更新分类数量显示
        self.updateAllCategoryCounts()
        # 刷新视图
        self.treeView.viewport().update()

    def __len__(self):
        # 计算所有形状项的数量（不包括分类项）
        count = 0
        for i in range(self.model.rowCount()):
            category_item = self.model.item(i, 0)
            count += category_item.rowCount()
        return count

    def __iter__(self):
        # 遍历所有形状项（不包括分类项）
        for i in range(self.model.rowCount()):
            category_item = self.model.item(i, 0)
            for j in range(category_item.rowCount()):
                yield category_item.child(j, 0)

    def itemSelectionChangedEvent(self, selected, deselected):
        selected_items = []
        for index in selected.indexes():
            item = self.model.itemFromIndex(index)
            # 确保项是LabelTreeWidgetItem类型并且不是分类
            if isinstance(item, LabelTreeWidgetItem) and not item.is_category:
                selected_items.append(item)

        deselected_items = []
        for index in deselected.indexes():
            item = self.model.itemFromIndex(index)
            # 确保项是LabelTreeWidgetItem类型并且不是分类
            if isinstance(item, LabelTreeWidgetItem) and not item.is_category:
                deselected_items.append(item)

        self.itemSelectionChanged.emit(selected_items, deselected_items)

    def itemDoubleClickedEvent(self, index):
        item = self.model.itemFromIndex(index)
        # 确保项是LabelTreeWidgetItem类型并且不是分类
        if isinstance(item, LabelTreeWidgetItem) and not item.is_category:
            self.itemDoubleClicked.emit(item)

    def selectedItems(self):
        items = []
        for index in self.treeView.selectedIndexes():
            item = self.model.itemFromIndex(index)
            # 确保项是LabelTreeWidgetItem类型并且不是分类
            if isinstance(item, LabelTreeWidgetItem) and not item.is_category:
                items.append(item)
        return items

    def getCategoryItem(self, category_name):
        """获取或创建分类项"""
        if category_name in self.categories:
            return self.categories[category_name]

        # 创建新分类
        category_item = LabelTreeWidgetItem(
            category_name, None, True, self.is_dark)

        # 设置分类图标
        icon_map = {
            "多边形": ":/icons/polygon.png",
            "矩形": ":/icons/rectangle.png",
            "圆形": ":/icons/circle.png",
            "线段": ":/icons/line.png",
            "点": ":/icons/point.png",
            "折线": ":/icons/linestrip.png",
            "多点": ":/icons/points.png",
            "蒙版": ":/icons/mask.png"
        }

        # 如果有对应图标，设置图标
        if category_name in icon_map:
            category_item.setIcon(QtGui.QIcon(icon_map[category_name]))

        self.model.appendRow(category_item)
        self.categories[category_name] = category_item
        self.treeView.expand(category_item.index())  # 展开分类项
        return category_item

    def addItem(self, item):
        """添加项到对应的分类中"""
        if not isinstance(item, LabelTreeWidgetItem):
            raise TypeError("item must be LabelTreeWidgetItem")

        # 设置项的颜色主题
        item.is_dark = self.is_dark

        shape = item.shape()
        if not shape:
            return

        # 根据形状类型进行分类
        shape_type = shape.shape_type
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
        self.treeView.expand(category_item.index())  # 确保展开显示

        # 更新分类数量
        self.updateCategoryCount(category_name)

    def removeItem(self, item):
        """从树中移除项"""
        if not isinstance(item, LabelTreeWidgetItem):
            # 如果不是正确的类型，尝试找到对应的形状项
            shape = getattr(item, 'shape', lambda: None)()
            if shape:
                item = self.findItemByShape(shape)
                if not item:
                    return

        parent = item.parent()
        if parent:
            row = item.row()
            parent.removeRow(row)

            # 找到并更新对应的分类数量
            for category_name, category_item in self.categories.items():
                if category_item == parent:
                    self.updateCategoryCount(category_name)
                    break
        else:
            # 如果没有父项（这不应该发生），从模型中移除
            index = self.model.indexFromItem(item)
            if index.isValid():
                self.model.removeRow(index.row(), index.parent())

    def selectItem(self, item):
        """选择一个项"""
        if not isinstance(item, LabelTreeWidgetItem):
            return

        index = self.model.indexFromItem(item)
        if index.isValid():
            self.treeView.selectionModel().select(index, QtCore.QItemSelectionModel.Select)

    def scrollToItem(self, item):
        """滚动到指定项"""
        if not isinstance(item, LabelTreeWidgetItem):
            return

        index = self.model.indexFromItem(item)
        if index.isValid():
            self.treeView.scrollTo(index)

    def findItemByShape(self, shape):
        """根据形状查找项"""
        for i in range(self.model.rowCount()):
            category_item = self.model.item(i, 0)
            for j in range(category_item.rowCount()):
                item = category_item.child(j, 0)
                if isinstance(item, LabelTreeWidgetItem) and item.shape() == shape:
                    return item
        return None

    def clear(self):
        """清空所有项"""
        self.model.clear()
        self.categories = {}  # 重置分类字典

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

    def clearSelection(self):
        """清除所有选择"""
        self.treeView.clearSelection()

    def expandAll(self):
        """展开所有项"""
        self.treeView.expandAll()

    def expandAll(self):
        """展开所有项"""
        self.treeView.expandAll()
