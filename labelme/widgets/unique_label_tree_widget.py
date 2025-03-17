import html

from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QStandardItem, QStandardItemModel
from PyQt5.QtWidgets import QStyle, QTreeView


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


class UniqueLabelTreeWidgetItem(QStandardItem):
    def __init__(self, text=None, label=None, is_category=False, shape_type=None):
        super(UniqueLabelTreeWidgetItem, self).__init__()
        self.setText(text or "")
        self.setLabel(label)
        self.is_category = is_category
        self.shape_type = shape_type

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

    def setLabel(self, label):
        self.setData(label, Qt.UserRole)

    def label(self):
        return self.data(Qt.UserRole)

    def __hash__(self):
        return id(self)


class UniqueLabelTreeWidget(QTreeView):
    itemDoubleClicked = QtCore.pyqtSignal(UniqueLabelTreeWidgetItem)

    def __init__(self):
        super(UniqueLabelTreeWidget, self).__init__()
        self.categories = {}  # 存储所有分类
        self.labels_by_category = {}  # 按类别存储标签

        self.setWindowFlags(Qt.Window)
        self.model = QStandardItemModel()
        self.setModel(self.model)
        self.setItemDelegate(HTMLDelegate())
        self.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.setHeaderHidden(True)

        # 设置动画
        self.setAnimated(True)
        self.setIndentation(20)  # 设置缩进

        # 设置图标大小
        self.setIconSize(QtCore.QSize(20, 20))

        self.expandAll()

        # 连接信号
        self.doubleClicked.connect(self.itemDoubleClickedEvent)

    def itemDoubleClickedEvent(self, index):
        item = self.model.itemFromIndex(index)
        if item and not getattr(item, 'is_category', False):
            self.itemDoubleClicked.emit(item)

    def count(self):
        # 计算所有标签项的数量（不包括分类项）
        count = 0
        for i in range(self.model.rowCount()):
            category_item = self.model.item(i, 0)
            count += category_item.rowCount()
        return count

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
        category_item = UniqueLabelTreeWidgetItem(category_name, None, True)
        self.model.appendRow(category_item)
        self.categories[category_name] = category_item
        if category_name not in self.labels_by_category:
            self.labels_by_category[category_name] = []
        self.expandAll()  # 展开所有项
        return category_item

    def createItemFromLabel(self, label, shape_type=None):
        """从标签创建项"""
        if self.findItemByLabel(label):
            raise ValueError(f"Item for label '{label}' already exists")

        item = UniqueLabelTreeWidgetItem(label=label, shape_type=shape_type)
        return item

    def addItem(self, item):
        """添加项到对应的分类中"""
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

        self.expandAll()  # 确保展开显示
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
        return [self.model.itemFromIndex(i) for i in self.selectedIndexes()
                if not getattr(self.model.itemFromIndex(i), 'is_category', False)]

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
