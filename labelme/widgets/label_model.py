"""
标签数据模型模块 - 高效管理标签数据

该模块实现了高效的标签数据管理模型，使用现代设计模式和优化算法
提高标签统计和查询的性能。
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Union, Tuple, Callable
from PyQt5.QtCore import QObject, pyqtSignal


class LabelCategory:
    """标签分类模型"""

    def __init__(self, name: str, display_name: str = None):
        """
        初始化标签分类

        Args:
            name: 分类名称
            display_name: 分类显示名称，若为None则使用name
        """
        self.name = name
        self.display_name = display_name or name
        self.labels: Set[str] = set()
        self.items: List[int] = []  # 存储项目ID

    def add_label(self, label: str) -> bool:
        """添加标签到分类"""
        if label in self.labels:
            return False
        self.labels.add(label)
        return True

    def remove_label(self, label: str) -> bool:
        """从分类中移除标签"""
        if label not in self.labels:
            return False
        self.labels.remove(label)
        return True

    def add_item(self, item_id: int) -> None:
        """添加项目ID到分类"""
        if item_id not in self.items:
            self.items.append(item_id)

    def remove_item(self, item_id: int) -> None:
        """从分类中移除项目ID"""
        if item_id in self.items:
            self.items.remove(item_id)

    @property
    def count(self) -> int:
        """获取该分类下的项目数量"""
        return len(self.items)

    def __str__(self) -> str:
        return f"{self.display_name} ({self.count})"


class LabelModel(QObject):
    """
    标签数据模型 - 高效管理与组织所有标签数据

    使用快速索引和缓存优化标签查询性能
    """

    # 模型变更信号
    categoriesChanged = pyqtSignal()
    labelsChanged = pyqtSignal()
    itemsChanged = pyqtSignal()

    def __init__(self):
        """初始化标签数据模型"""
        super().__init__()

        # 数据存储
        self._categories: Dict[str, LabelCategory] = {}  # 类别名称 -> 类别对象
        self._labels_by_category: Dict[str, Set[str]] = defaultdict(
            set)  # 类别 -> 标签集合
        self._item_map: Dict[int, dict] = {}  # 项目ID -> 项目数据
        self._label_to_items: Dict[str, Set[int]
                                   ] = defaultdict(set)  # 标签 -> 项目ID集合
        self._category_rules: Dict[str, Callable] = {}  # 自动分类规则

        # 创建默认分类规则
        self._setup_default_category_rules()

    def _setup_default_category_rules(self):
        """设置默认分类规则"""
        self._category_rules = {
            "polygon": lambda item: "多边形",
            "rectangle": lambda item: "矩形",
            "circle": lambda item: "圆形",
            "line": lambda item: "线段",
            "point": lambda item: "点",
            "linestrip": lambda item: "折线",
            "points": lambda item: "多点",
            "mask": lambda item: "蒙版"
        }

        # 确保所有默认分类已创建
        for shape_type, rule in self._category_rules.items():
            category_name = rule({"shape_type": shape_type})
            if category_name not in self._categories:
                self.add_category(category_name)

    def add_category(self, name: str, display_name: str = None) -> LabelCategory:
        """
        添加或获取分类

        如果分类已存在则返回现有分类，否则创建新分类

        Args:
            name: 分类名称
            display_name: 显示名称

        Returns:
            LabelCategory: 分类对象
        """
        if name in self._categories:
            return self._categories[name]

        category = LabelCategory(name, display_name)
        self._categories[name] = category
        self.categoriesChanged.emit()
        return category

    def remove_category(self, name: str) -> bool:
        """
        移除分类

        Args:
            name: 分类名称

        Returns:
            bool: 是否成功移除
        """
        if name not in self._categories:
            return False

        # 移除分类中的标签引用
        labels = self._labels_by_category.pop(name, set())

        # 移除分类
        self._categories.pop(name)
        self.categoriesChanged.emit()
        return True

    def get_category(self, name: str) -> Optional[LabelCategory]:
        """获取指定名称的分类"""
        return self._categories.get(name)

    def get_category_names(self) -> List[str]:
        """获取所有分类名称"""
        return list(self._categories.keys())

    def get_categories(self) -> List[LabelCategory]:
        """获取所有分类对象"""
        return list(self._categories.values())

    def add_item(self, item_id: Union[int, str], label: str,
                 shape_type: str = "polygon", extra_data: dict = None) -> bool:
        """
        添加项目到模型

        Args:
            item_id: 项目唯一ID
            label: 标签名称
            shape_type: 形状类型
            extra_data: 额外数据

        Returns:
            bool: 是否添加成功
        """
        if not label or item_id in self._item_map:
            return False

        # 标准化标签名称
        label = label.strip()

        # 准备项目数据
        item_data = {
            'id': item_id,
            'label': label,
            'shape_type': shape_type,
            'selected': False,
            'visible': True,  # 默认可见
        }

        # 合并额外数据
        if extra_data:
            for key, value in extra_data.items():
                item_data[key] = value

        # 添加到项目映射
        self._item_map[item_id] = item_data

        # 更新标签到项目的映射
        self._label_to_items[label].add(item_id)

        # 使用规则确定分类
        category_name = self._determine_category(item_data)

        # 将标签添加到分类
        category = self.get_category(category_name)
        if not category:
            category = self.add_category(category_name)

        if category:
            category.add_label(label)
            category.add_item(item_id)
            self._labels_by_category[category_name].add(label)

        # 发送信号
        self.itemsChanged.emit()

        return True

    def remove_item(self, item_id: int) -> bool:
        """
        从模型中移除项目

        Args:
            item_id: 项目ID

        Returns:
            bool: 是否成功移除
        """
        if item_id not in self._item_map:
            return False

        # 获取项目数据
        item = self._item_map[item_id]
        label = item["label"]

        # 从标签到项目的映射中移除
        if label in self._label_to_items:
            self._label_to_items[label].discard(item_id)
            # 如果没有项目使用此标签，考虑是否需要清理
            if not self._label_to_items[label]:
                del self._label_to_items[label]

        # 从分类中移除
        category_name = self._determine_category(item)
        if category_name in self._categories:
            self._categories[category_name].remove_item(item_id)

        # 移除项目
        del self._item_map[item_id]

        self.itemsChanged.emit()
        return True

    def get_item(self, item_id: int) -> Optional[dict]:
        """获取项目数据"""
        return self._item_map.get(item_id)

    def get_items_by_label(self, label: str) -> List[int]:
        """获取指定标签的所有项目ID"""
        return list(self._label_to_items.get(label, set()))

    def get_items_by_category(self, category_name: str) -> List[int]:
        """获取指定分类的所有项目ID"""
        category = self._categories.get(category_name)
        if not category:
            return []
        return category.items

    def get_labels(self) -> Set[str]:
        """获取所有标签"""
        return set(self._label_to_items.keys())

    def get_labels_by_category(self, category_name: str) -> Set[str]:
        """获取指定分类的所有标签"""
        return self._labels_by_category.get(category_name, set())

    def update_item(self,
                    item_id: int,
                    label: str = None,
                    shape_type: str = None,
                    data: dict = None) -> bool:
        """
        更新项目数据

        Args:
            item_id: 项目ID
            label: 新标签（如需更新）
            shape_type: 新形状类型（如需更新）
            data: 新数据（如需更新）

        Returns:
            bool: 是否成功更新
        """
        if item_id not in self._item_map:
            return False

        # 获取当前数据
        old_data = self._item_map[item_id]
        old_label = old_data["label"]
        old_shape_type = old_data["shape_type"]
        old_category = self._determine_category(old_data)

        # 准备更新数据
        new_data = old_data.copy()
        if label is not None:
            new_data["label"] = label
        if shape_type is not None:
            new_data["shape_type"] = shape_type
        if data:
            new_data.update(data)

        # 确定新分类
        new_category = self._determine_category(new_data)

        # 如果标签发生变化，更新标签到项目的映射
        if label is not None and label != old_label:
            self._label_to_items[old_label].discard(item_id)
            if not self._label_to_items[old_label]:
                del self._label_to_items[old_label]
            self._label_to_items[label].add(item_id)

        # 如果分类发生变化，更新分类
        if old_category != new_category:
            # 从旧分类中移除
            if old_category in self._categories:
                self._categories[old_category].remove_item(item_id)
                if old_label is not None and label is not None and old_label != label:
                    self._labels_by_category[old_category].discard(old_label)

            # 添加到新分类
            category = self.add_category(new_category)
            category.add_item(item_id)
            if label is not None:
                category.add_label(label)
                self._labels_by_category[new_category].add(label)

        # 更新存储的项目数据
        self._item_map[item_id] = new_data

        self.itemsChanged.emit()
        return True

    def _determine_category(self, item_data: dict) -> str:
        """
        根据项目数据确定分类

        使用已注册的分类规则确定项目应归属的分类

        Args:
            item_data: 项目数据

        Returns:
            str: 分类名称
        """
        shape_type = item_data.get("shape_type")

        # 应用分类规则
        if shape_type in self._category_rules:
            return self._category_rules[shape_type](item_data)

        # 默认分类
        return "未分类"

    def clear(self):
        """清空所有数据"""
        self._categories.clear()
        self._labels_by_category.clear()
        self._item_map.clear()
        self._label_to_items.clear()

        # 重新设置默认分类
        self._setup_default_category_rules()

        self.categoriesChanged.emit()
        self.labelsChanged.emit()
        self.itemsChanged.emit()
