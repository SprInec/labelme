from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QStandardItem, QStandardItemModel
from PyQt5.QtWidgets import QStyle, QTreeView

import html


class HTMLDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, parent=None):
        super(HTMLDelegate, self).__init__()
        self.doc = QtGui.QTextDocument(self)

    def paint(self, painter, option, index):
        painter.save()

        options = QtWidgets.QStyleOptionViewItem(option)

        self.initStyleOption(options, index)
        self.doc.setHtml(options.text)
        options.text = ""

        style = (
            QtWidgets.QApplication.style()
            if options.widget is None
            else options.widget.style()
        )
        style.drawControl(QStyle.CE_ItemViewItem, options, painter)

        ctx = QtGui.QAbstractTextDocumentLayout.PaintContext()

        if option.state & QStyle.State_Selected:
            ctx.palette.setColor(
                QPalette.Text,
                option.palette.color(
                    QPalette.Active, QPalette.HighlightedText),
            )
        else:
            ctx.palette.setColor(
                QPalette.Text,
                option.palette.color(QPalette.Active, QPalette.Text),
            )

        textRect = style.subElementRect(QStyle.SE_ItemViewItemText, options)

        if index.column() != 0:
            textRect.adjust(5, 0, 0, 0)

        margin = (option.rect.height() - options.fontMetrics.height()) // 2
        margin = margin - 2
        textRect.setTop(textRect.top() + margin)

        painter.translate(textRect.topLeft())
        painter.setClipRect(textRect.translated(-textRect.topLeft()))
        self.doc.documentLayout().draw(painter, ctx)

        painter.restore()

    def sizeHint(self, option, index):
        return QtCore.QSize(
            int(self.doc.idealWidth()),
            int(self.doc.size().height() - 2),
        )


class LabelTreeWidgetItem(QStandardItem):
    def __init__(self, text=None, shape=None, is_category=False):
        super(LabelTreeWidgetItem, self).__init__()
        self.setText(text or "")
        self.setShape(shape)
        self.is_category = is_category

        # 如果不是分类项，设置可选中
        if not is_category and shape:
            self.setCheckable(True)
            self.setCheckState(
                Qt.Checked if shape.isVisible() else Qt.Unchecked)

        self.setEditable(False)
        self.setSelectable(True)

        # 如果是分类项，设置粗体
        if is_category:
            font = self.font()
            font.setBold(True)
            self.setFont(font)

            # 为分类项设置更现代的字体大小
            font.setPointSize(10)
            self.setFont(font)

            # 为分类项设置更暗的颜色
            self.setForeground(QtGui.QColor(51, 51, 51))

    def clone(self):
        return LabelTreeWidgetItem(self.text(), self.shape(), self.is_category)

    def setShape(self, shape):
        self.setData(shape, Qt.UserRole)

    def shape(self):
        return self.data(Qt.UserRole)

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return '{}("{}")'.format(self.__class__.__name__, self.text())


class LabelTreeWidget(QTreeView):
    itemDoubleClicked = QtCore.pyqtSignal(LabelTreeWidgetItem)
    itemSelectionChanged = QtCore.pyqtSignal(list, list)

    def __init__(self):
        super(LabelTreeWidget, self).__init__()
        self._selectedItems = []
        self.categories = {}  # 存储所有分类

        self.setWindowFlags(Qt.Window)
        self.model = QStandardItemModel()
        self.setModel(self.model)
        self.setItemDelegate(HTMLDelegate())
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setHeaderHidden(True)

        # 设置样式
        self.setStyleSheet("""
            QTreeView {
                background-color: transparent;
                outline: none;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 8px;
            }
            QTreeView::item {
                padding: 12px 10px;
                border-radius: 6px;
                margin: 4px 2px;
                color: #333333;
                min-height: 24px;
            }
            QTreeView::item:hover {
                background-color: #f5f5f5;
            }
            QTreeView::item:selected {
                background-color: #e8f0fe;
                color: #1967d2;
            }
            QTreeView::branch {
                background-color: transparent;
                padding-left: 8px;
            }
            QTreeView::branch:has-children:!has-siblings:closed,
            QTreeView::branch:closed:has-children:has-siblings {
                image: url(:/right-arrow.png);
                padding-left: 10px;
            }
            QTreeView::branch:open:has-children:!has-siblings,
            QTreeView::branch:open:has-children:has-siblings {
                image: url(:/down-arrow.png);
                padding-left: 10px;
            }
            QTreeView::indicator {
                width: 22px;
                height: 22px;
                border: 2px solid #bdbdbd;
                border-radius: 4px;
                background-color: #ffffff;
                margin-right: 8px;
            }
            QTreeView::indicator:checked {
                background-color: #4285f4;
                border: 2px solid #4285f4;
                image: url(:/check-white.png);
            }
            QTreeView::indicator:unchecked:hover {
                border: 2px solid #4285f4;
            }
            
            /* 分类项样式 */
            QStandardItem[is_category="true"] {
                font-weight: bold;
                background-color: #f8f9fa;
                border-radius: 6px;
                padding: 4px;
                margin-top: 5px;
                margin-bottom: 2px;
            }
        """)

        # 设置动画
        self.setAnimated(True)
        self.setIndentation(20)  # 设置缩进

        # 设置图标大小
        self.setIconSize(QtCore.QSize(20, 20))

        self.expandAll()

        # 启用拖放功能
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)

        # 连接信号
        self.doubleClicked.connect(self.itemDoubleClickedEvent)
        self.selectionModel().selectionChanged.connect(self.itemSelectionChangedEvent)

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
            if item and not getattr(item, 'is_category', False):
                selected_items.append(item)

        deselected_items = []
        for index in deselected.indexes():
            item = self.model.itemFromIndex(index)
            if item and not getattr(item, 'is_category', False):
                deselected_items.append(item)

        self.itemSelectionChanged.emit(selected_items, deselected_items)

    def itemDoubleClickedEvent(self, index):
        item = self.model.itemFromIndex(index)
        if item and not getattr(item, 'is_category', False):
            self.itemDoubleClicked.emit(item)

    def selectedItems(self):
        return [self.model.itemFromIndex(i) for i in self.selectedIndexes()
                if not getattr(self.model.itemFromIndex(i), 'is_category', False)]

    def getCategoryItem(self, category_name):
        """获取或创建分类项"""
        if category_name in self.categories:
            return self.categories[category_name]

        # 创建新分类
        category_item = LabelTreeWidgetItem(category_name, None, True)
        self.model.appendRow(category_item)
        self.categories[category_name] = category_item
        self.expandAll()  # 展开所有项
        return category_item

    def addItem(self, item):
        """添加项到对应的分类中"""
        if not isinstance(item, LabelTreeWidgetItem):
            raise TypeError("item must be LabelTreeWidgetItem")

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
        self.expandAll()  # 确保展开显示

        # 更新分类数量
        self.updateCategoryCount(category_name)

    def removeItem(self, item):
        """从树中移除项"""
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
        index = self.model.indexFromItem(item)
        if index.isValid():
            self.selectionModel().select(index, QtCore.QItemSelectionModel.Select)

    def scrollToItem(self, item):
        """滚动到指定项"""
        if item:  # 确保item不是None
            index = self.model.indexFromItem(item)
            if index.isValid():
                self.scrollTo(index)

    def findItemByShape(self, shape):
        """根据形状查找项"""
        for i in range(self.model.rowCount()):
            category_item = self.model.item(i, 0)
            for j in range(category_item.rowCount()):
                item = category_item.child(j, 0)
                if item and item.shape() == shape:
                    return item
        return None  # 找不到形状时返回None，而不是抛出异常

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
        super(LabelTreeWidget, self).clearSelection()
