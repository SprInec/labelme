"""
标签系统适配器模块 - 连接新的标签管理系统和现有应用

这个模块提供了适配器类，使新的标签管理系统能够无缝集成到现有应用中。
"""

import html
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

from .label_window import LabelWindow
from .label_model import LabelModel
from .label_panel import LabelPanelController, LabelListPanel, LabelUniqueListPanel


class LabelTreeWidgetAdapter(QtWidgets.QWidget):
    """
    LabelTreeWidget适配器类，保持与原LabelTreeWidget接口兼容
    """

    # 与原接口兼容的信号
    itemDoubleClicked = QtCore.pyqtSignal(object)
    itemSelectionChanged = QtCore.pyqtSignal(list, list)

    def __init__(self):
        """初始化适配器"""
        super(LabelTreeWidgetAdapter, self).__init__()

        # 初始化属性
        self.model = None
        self.panel = None
        self.controller = None

        # 布局 - 确保是实例变量而不是方法引用
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # 项目映射缓存，用于兼容旧接口
        self._item_cache = {}  # 项目数据 -> 适配器项目
        self._selectedItems = []

    def _create_adapter_item(self, item_data):
        """创建与旧接口兼容的适配器项目"""
        if not item_data:
            return None

        # 如果是字典，使用对象的id作为键
        cache_key = id(item_data) if isinstance(item_data, dict) else item_data

        if cache_key in self._item_cache:
            return self._item_cache[cache_key]

        # 创建适配器项目
        adapter_item = LabelTreeItemAdapter(item_data)

        # 确保信号连接正确
        if hasattr(adapter_item, '_model') and hasattr(adapter_item._model, 'itemChanged'):
            try:
                # 防止重复连接 - 先尝试查找连接的槽函数数量
                connections = adapter_item._model.receivers(
                    adapter_item._model.itemChanged)
                if connections == 0:
                    # 如果没有连接，添加一个空的信号槽
                    adapter_item._model.itemChanged.connect(lambda: None)
            except:
                # 如果出现异常，只添加连接而不尝试断开
                adapter_item._model.itemChanged.connect(lambda: None)

        # 保存到缓存
        self._item_cache[cache_key] = adapter_item
        return adapter_item

    def _on_item_double_clicked(self, item_data):
        """内部双击处理"""
        adapter_item = self._create_adapter_item(item_data)
        if adapter_item:
            self.itemDoubleClicked.emit(adapter_item)

    def _on_item_selected(self, item_data):
        """内部选中处理"""
        adapter_item = self._create_adapter_item(item_data)
        if adapter_item and adapter_item not in self._selectedItems:
            self._selectedItems.append(adapter_item)
            self.itemSelectionChanged.emit([adapter_item], [])

    def _on_item_deselected(self, item_data):
        """内部取消选中处理"""
        adapter_item = self._create_adapter_item(item_data)
        if adapter_item and adapter_item in self._selectedItems:
            self._selectedItems.remove(adapter_item)
            self.itemSelectionChanged.emit([], [adapter_item])

    def count(self):
        """获取项目数量，兼容旧接口"""
        if self.model:
            return len(self.model._item_map)
        return 0

    def item(self, index):
        """获取指定索引的项目，兼容旧接口"""
        if not self.model or index < 0 or index >= len(self.model._item_map):
            return None

        # 将字典转换为列表以支持索引访问
        item_keys = list(self.model._item_map.keys())
        if index < len(item_keys):
            item_id = item_keys[index]
            item_data = self.model._item_map[item_id]
            return self._create_adapter_item(item_data)

    def __iter__(self):
        """遍历所有项目"""
        if hasattr(self, 'model') and self.model and hasattr(self.model, '_item_map'):
            for item_id, item_data in self.model._item_map.items():
                # 安全地创建并返回适配器项目
                adapter_item = self._create_adapter_item(item_data)
                if adapter_item:
                    yield adapter_item

    def findItemByShape(self, shape):
        """根据形状查找项目"""
        for item_id, item_data in self.model._item_map.items():
            if 'id' in item_data and item_data.get('shape') == shape:
                return self._create_adapter_item(item_data)
        return None

    def addItem(self, item):
        """添加项目"""
        # 从传入的旧项目中提取出形状对象
        shape = None

        # 处理不同类型的项目
        if hasattr(item, 'shape') and callable(item.shape):
            # 如果是适配器项目对象
            shape = item.shape()
        elif hasattr(item, '_shape'):
            # 如果是原始LabelTreeWidgetItem对象
            shape = item._shape

        if not shape:
            return

        # 提取标签等信息
        label = shape.label if hasattr(shape, 'label') else ""
        shape_type = shape.shape_type if hasattr(
            shape, 'shape_type') else "polygon"

        # 获取颜色信息
        fill_color = None
        line_color = None
        if hasattr(shape, 'fill_color'):
            fill_color = shape.fill_color.getRgb()[:3]
        if hasattr(shape, 'line_color'):
            line_color = shape.line_color.getRgb()[:3]

        # 创建新项目
        item_id = id(shape)  # 使用形状对象的id作为项目ID

        # 准备额外的数据
        extra_data = {
            'id': item_id,
            'shape': shape,  # 存储原始形状对象
            'group_id': getattr(shape, 'group_id', None),
            'shape_type': shape_type,
            'fill_color': fill_color,
            'line_color': line_color,
            'selected': False,
            'visible': getattr(shape, 'visible', True)
        }

        # 添加到模型
        self.model.add_item(item_id, label, shape_type, extra_data)

        # 确保标签也被添加到唯一标签列表
        if label and label not in self.model.get_labels():
            self.model.add_label(label, fill_color)

        # 更新所有分类的计数
        self.updateAllCategoryCounts()

    def removeItem(self, item):
        """移除项目"""
        if isinstance(item, LabelTreeItemAdapter):
            item_data = item.item_data
            if item_data and 'id' in item_data:
                self.model.remove_item(item_data['id'])

                # 从缓存中移除，使用同样的键生成逻辑
                cache_key = id(item_data) if isinstance(
                    item_data, dict) else item_data
                if cache_key in self._item_cache:
                    del self._item_cache[cache_key]

        # 更新所有分类的计数
        self.updateAllCategoryCounts()

    def selectItem(self, item):
        """选中项目"""
        if isinstance(item, LabelTreeItemAdapter):
            item_data = item.item_data
            if item_data and 'id' in item_data:
                self.panel.label_tree.select_item(item_data['id'])

    def selectedItems(self):
        """获取所有选中的项目"""
        return self._selectedItems

    def clear(self):
        """清空所有项目"""
        self.model.clear()
        self._item_cache.clear()
        self._selectedItems.clear()

    def updateCategoryCount(self, category_name):
        """更新分类计数（保持兼容接口）"""
        category = self.model.get_category(category_name)
        if category:
            # 模型会自动更新分类计数，这里不需要额外操作
            pass

    def updateAllCategoryCounts(self):
        """更新所有分类计数（保持兼容接口）"""
        # 新系统自动维护计数，这里仅触发刷新
        self.model.categoriesChanged.emit()

    def __len__(self):
        """获取项目数量"""
        return self.count()

    def clearSelection(self):
        """清除所有选择，兼容旧接口"""
        # 记录当前选中的项目
        old_selection = []
        if hasattr(self, '_selectedItems') and self._selectedItems:
            old_selection = self._selectedItems.copy()
            self._selectedItems = []

        # 重置所有项目的选中状态
        if hasattr(self, 'model') and self.model:
            for item_id, item_data in self.model._item_map.items():
                if 'selected' in item_data:
                    item_data['selected'] = False

            # 通知UI更新 - 使用正确的信号
            if hasattr(self.model, 'itemsChanged'):
                self.model.itemsChanged.emit()

        # 如果有之前选中的项目，发送选择变更信号
        if old_selection:
            self.itemSelectionChanged.emit([], old_selection)

    def setModel(self, model):
        """设置数据模型，由LabelPanelController调用"""
        self.model = model

        # 只有在没有面板的情况下才创建新的控制器和面板
        if not self.panel:
            # 创建控制器和面板
            self.controller = LabelPanelController(self.model)
            self.panel = self.controller.get_label_list_panel()

            # 添加到布局
            self.layout.addWidget(self.panel)

            # 连接内部信号到适配器信号
            self.panel.itemDoubleClicked.connect(self._on_item_double_clicked)
            self.panel.itemSelected.connect(self._on_item_selected)
            self.panel.itemDeselected.connect(self._on_item_deselected)
        # 否则只更新现有面板的模型
        elif hasattr(self.panel, 'set_model'):
            self.panel.set_model(model)


class LabelTreeItemAdapter:
    """标签树项目适配器，兼容旧接口"""

    def __init__(self, item_data):
        """初始化适配器项目"""
        self.item_data = item_data
        self.is_category = False
        self._model = QtGui.QStandardItemModel()  # 添加模型对象
        self._model_item = QtGui.QStandardItem()  # 添加标准项目对象
        self._model.appendRow(self._model_item)  # 将项目添加到模型中

    def model(self):
        """获取项目模型"""
        return self._model  # 返回模型对象 - 解决NoneType错误

    def shape(self):
        """获取形状对象"""
        return self.item_data.get('shape')

    def text(self):
        """获取文本"""
        shape = self.shape()
        if shape:
            label = shape.label
            group_id = shape.group_id

            if group_id is None:
                text = label
            else:
                text = "{} ({})".format(label, group_id)

            # 添加颜色标记
            if hasattr(shape, 'fill_color'):
                r, g, b = shape.fill_color.getRgb()[:3]
                colored_text = '{} <font color="#{:02x}{:02x}{:02x}">●</font>'.format(
                    html.escape(text), r, g, b
                )
                return colored_text

            return text
        return ""

    def setText(self, text):
        """设置文本"""
        # 这个适配器中不直接修改文本，而是通过模型更新
        pass

    def checkState(self):
        """获取选中状态"""
        shape = self.shape()
        state = Qt.Checked
        if shape and hasattr(shape, 'isVisible'):
            state = Qt.Checked if shape.isVisible() else Qt.Unchecked
        # 将状态同步到模型项目
        self._model_item.setCheckState(state)
        return state

    def setCheckState(self, state):
        """设置选中状态"""
        # 更新模型项目的状态
        self._model_item.setCheckState(state)
        # 更新关联的形状对象
        shape = self.shape()
        if shape and hasattr(shape, 'setVisible'):
            shape.setVisible(state == Qt.Checked)

    def __eq__(self, other):
        """比较两个适配器项目是否相等"""
        if isinstance(other, LabelTreeItemAdapter):
            return self.item_data == other.item_data
        return False


class UniqueLabelTreeWidgetAdapter(QtWidgets.QWidget):
    """
    UniqueLabelTreeWidget适配器类，保持与原UniqueLabelTreeWidget接口兼容
    """

    # 与原接口兼容的信号
    itemDoubleClicked = QtCore.pyqtSignal(object)
    itemSelectionChanged = QtCore.pyqtSignal(list, list)

    def __init__(self):
        """初始化适配器"""
        super(UniqueLabelTreeWidgetAdapter, self).__init__()

        # 创建内部组件
        self.model = None
        self.panel = None

        # 选中的项目
        self._selectedItem = None

        # 布局 - 确保是实例变量而不是方法引用
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # 项目映射缓存
        self._item_cache = {}  # 标签 -> 适配器项目

    def setModel(self, model):
        """设置数据模型，由LabelPanelController调用"""
        self.model = model

        # 只有在没有面板的情况下才创建新的控制器和面板
        if not self.panel:
            # 创建控制器和面板
            self.controller = LabelPanelController(self.model)
            self.panel = self.controller.get_unique_label_panel()

            # 添加到布局
            self.layout.addWidget(self.panel)

            # 连接信号
            self.panel.labelSelected.connect(self._on_label_selected)
        # 否则只更新现有面板的模型
        elif hasattr(self.panel, 'set_model'):
            self.panel.set_model(model)

    def _create_adapter_item(self, label, shape_type=None):
        """创建与旧接口兼容的适配器项目"""
        if not label:
            return None

        if label in self._item_cache:
            return self._item_cache[label]

        # 创建适配器项目
        adapter_item = UniqueLabelTreeItemAdapter(label, shape_type)

        # 确保信号连接正确
        if hasattr(adapter_item, '_model') and hasattr(adapter_item._model, 'itemChanged'):
            try:
                # 防止重复连接
                connections = adapter_item._model.receivers(
                    adapter_item._model.itemChanged)
                if connections == 0:
                    # 如果没有连接，添加一个空的信号槽
                    adapter_item._model.itemChanged.connect(lambda: None)
            except:
                # 如果出现异常，只添加连接
                adapter_item._model.itemChanged.connect(lambda: None)

        self._item_cache[label] = adapter_item
        return adapter_item

    def _on_label_selected(self, label):
        """内部标签选中处理"""
        if not label or not isinstance(label, str):
            return

        adapter_item = self._create_adapter_item(label)
        if adapter_item:
            self.itemDoubleClicked.emit(adapter_item)

    def findItemByLabel(self, label):
        """根据标签查找项目"""
        if not label or not self.model:
            return None

        # 检查标签是否存在于模型中
        if label in self.model.get_labels():
            return self._create_adapter_item(label)
        return None

    def createItemFromLabel(self, label, shape_type=None):
        """从标签创建项目"""
        return self._create_adapter_item(label, shape_type)

    def addItem(self, item):
        """添加项目"""
        if isinstance(item, UniqueLabelTreeItemAdapter):
            label = item.label()
            shape_type = item.shape_type

            # 检查模型中是否有使用该标签的项目
            items = self.model.get_items_by_label(label)
            if not items and self.model:
                # 如果没有项目使用该标签，添加一个虚拟项目
                item_id = id(label)  # 使用标签字符串的id作为项目ID
                self.model.add_item(item_id, label, shape_type or "unknown")

            # 更新所有分类的计数
            if self.model:
                self.model.labelsChanged.emit()

    def setItemLabel(self, item, label, rgb=None):
        """设置项目标签和颜色"""
        # 在此适配器中不需要额外操作，模型会自动处理
        pass

    def clear(self):
        """清空所有项目"""
        if self.model:
            self.model.clear()
        self._item_cache.clear()

    def updateCategoryCount(self, category_name):
        """更新分类计数（保持兼容接口）"""
        if self.model:
            category = self.model.get_category(category_name)
            if category:
                # 模型会自动更新分类计数，这里不需要额外操作
                pass

    def updateAllCategoryCounts(self):
        """更新所有分类计数（保持兼容接口）"""
        # 新系统自动维护计数，这里仅触发刷新
        if self.model:
            self.model.categoriesChanged.emit()

    def count(self):
        """获取项目数量，兼容旧接口"""
        if self.model:
            return len(self.model.get_labels())
        return 0

    def item(self, index):
        """获取指定索引的项目，兼容旧接口"""
        if not self.model or index < 0 or index >= self.count():
            return None

        # 获取标签列表并访问指定索引
        labels = list(self.model.get_labels())
        if index < len(labels):
            return self._create_adapter_item(labels[index])

    def __len__(self):
        """获取项目数量"""
        return self.count()

    def clearSelection(self):
        """清除所有选择，兼容旧接口"""
        # 如果有选中的项目，发送选择变更信号
        if hasattr(self, '_selectedItem') and self._selectedItem:
            old_selection = self._selectedItem
            self._selectedItem = None
            if hasattr(self, 'itemSelectionChanged'):
                self.itemSelectionChanged.emit([], [old_selection])

        # 重置所有高亮状态
        if hasattr(self, 'model') and self.model:
            # 通知UI更新 - 使用正确的信号
            if hasattr(self.model, 'labelsChanged'):
                self.model.labelsChanged.emit()


class UniqueLabelTreeItemAdapter:
    """唯一标签树项目适配器，兼容旧接口"""

    def __init__(self, label, shape_type=None):
        """初始化适配器项目"""
        self._label = label
        self.shape_type = shape_type
        self.is_category = False
        self._model = QtGui.QStandardItemModel()  # 添加模型对象
        self._model_item = QtGui.QStandardItem()  # 添加标准项目对象
        self._model.appendRow(self._model_item)  # 将项目添加到模型中

    def model(self):
        """获取项目模型"""
        return self._model  # 返回模型对象

    def label(self):
        """获取标签"""
        return self._label

    def setLabel(self, label):
        """设置标签"""
        self._label = label

    def text(self):
        """获取文本"""
        return self._label or ""

    def setText(self, text):
        """设置文本"""
        # 这个适配器中不直接修改文本
        pass

    def __eq__(self, other):
        """比较两个适配器项目是否相等"""
        if isinstance(other, UniqueLabelTreeItemAdapter):
            return self._label == other._label
        return False


def create_label_widgets():
    """创建标签管理组件，共享同一个数据模型"""
    try:
        # 创建共享数据模型
        model = LabelModel()

        # 创建标签树适配器 - 多边形标签
        label_tree = LabelTreeWidgetAdapter()
        label_tree.model = model

        # 创建唯一标签树适配器 - 标签列表
        unique_label_tree = UniqueLabelTreeWidgetAdapter()
        unique_label_tree.setModel(model)

        # 创建控制器并获取UI面板
        panel_controller = LabelPanelController(model)
        label_tree.panel = panel_controller.get_label_list_panel()
        unique_label_tree.panel = panel_controller.get_unique_label_panel()

        # 重新添加面板到布局
        if hasattr(label_tree, 'layout') and label_tree.layout:
            for i in reversed(range(label_tree.layout.count())):
                widget = label_tree.layout.itemAt(i).widget()
                if widget:
                    widget.setParent(None)
            label_tree.layout.addWidget(label_tree.panel)

        if hasattr(unique_label_tree, 'layout') and unique_label_tree.layout:
            for i in reversed(range(unique_label_tree.layout.count())):
                widget = unique_label_tree.layout.itemAt(i).widget()
                if widget:
                    widget.setParent(None)
            unique_label_tree.layout.addWidget(unique_label_tree.panel)

        return label_tree, unique_label_tree
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e
