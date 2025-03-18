#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
标签管理系统的演示程序

运行此文件可以测试标签管理系统的功能，包括标签模型、视图组件和控制器。
"""

import sys
import random

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout

from labelme.widgets.label_window import LabelWindow


class DemoApp(QMainWindow):
    """标签管理系统演示程序"""

    def __init__(self):
        """初始化演示程序"""
        super(DemoApp, self).__init__()

        # 窗口设置
        self.setWindowTitle("标签管理系统演示")
        self.resize(1200, 800)

        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QVBoxLayout(central_widget)

        # 创建操作按钮
        buttons_widget = QWidget()
        buttons_layout = QHBoxLayout(buttons_widget)

        # 创建标签窗口
        self.label_window = LabelWindow()

        # 测试按钮
        self.btn_add_random = QPushButton("添加随机标签")
        self.btn_clear = QPushButton("清空所有")
        self.btn_add_polygon = QPushButton("添加多边形")
        self.btn_add_rectangle = QPushButton("添加矩形")
        self.btn_add_circle = QPushButton("添加圆形")

        # 添加按钮到布局
        buttons_layout.addWidget(self.btn_add_random)
        buttons_layout.addWidget(self.btn_add_polygon)
        buttons_layout.addWidget(self.btn_add_rectangle)
        buttons_layout.addWidget(self.btn_add_circle)
        buttons_layout.addWidget(self.btn_clear)

        # 添加组件到主布局
        main_layout.addWidget(buttons_widget)
        main_layout.addWidget(self.label_window)

        # 连接按钮信号
        self.btn_add_random.clicked.connect(self.add_random_labels)
        self.btn_clear.clicked.connect(self.label_window.clear)
        self.btn_add_polygon.clicked.connect(lambda: self.add_shape("polygon"))
        self.btn_add_rectangle.clicked.connect(
            lambda: self.add_shape("rectangle"))
        self.btn_add_circle.clicked.connect(lambda: self.add_shape("circle"))

        # 连接标签窗口信号
        self.label_window.itemSelected.connect(self.on_item_selected)
        self.label_window.itemDeselected.connect(self.on_item_deselected)
        self.label_window.itemDoubleClicked.connect(
            self.on_item_double_clicked)

        # 添加一些示例数据
        self.next_id = 1
        self.add_example_data()

    def add_example_data(self):
        """添加示例数据"""
        # 添加多边形
        for i in range(5):
            self.label_window.add_item(
                self.next_id,
                f"人物_{i+1}",
                "polygon",
                {"color": "#FF0000"}
            )
            self.next_id += 1

        # 添加矩形
        for i in range(3):
            self.label_window.add_item(
                self.next_id,
                f"车辆_{i+1}",
                "rectangle",
                {"color": "#00FF00"}
            )
            self.next_id += 1

        # 添加圆形
        for i in range(2):
            self.label_window.add_item(
                self.next_id,
                f"标志_{i+1}",
                "circle",
                {"color": "#0000FF"}
            )
            self.next_id += 1

    def add_random_labels(self):
        """添加随机标签"""
        # 形状类型列表
        shape_types = ["polygon", "rectangle",
                       "circle", "line", "point", "linestrip"]

        # 标签前缀列表
        label_prefixes = ["人物", "车辆", "动物", "植物", "建筑", "标志", "背景", "道路"]

        # 随机生成10个标签
        for i in range(10):
            # 随机选择形状类型和标签前缀
            shape_type = random.choice(shape_types)
            label_prefix = random.choice(label_prefixes)

            # 随机生成颜色
            color = f"#{random.randint(0, 255):02X}{random.randint(0, 255):02X}{random.randint(0, 255):02X}"

            # 添加到标签窗口
            self.label_window.add_item(
                self.next_id,
                f"{label_prefix}_{self.next_id}",
                shape_type,
                {"color": color}
            )

            # 增加ID
            self.next_id += 1

    def add_shape(self, shape_type):
        """添加指定形状类型的标签"""
        # 根据形状类型选择标签前缀
        if shape_type == "polygon":
            prefix = "多边形"
        elif shape_type == "rectangle":
            prefix = "矩形"
        elif shape_type == "circle":
            prefix = "圆形"
        elif shape_type == "line":
            prefix = "线段"
        elif shape_type == "point":
            prefix = "点"
        else:
            prefix = "标签"

        # 添加到标签窗口
        self.label_window.add_item(
            self.next_id,
            f"{prefix}_{self.next_id}",
            shape_type,
            {"added_time": QtCore.QDateTime.currentDateTime().toString()}
        )

        # 增加ID
        self.next_id += 1

    def on_item_selected(self, item_data):
        """处理项目选中事件"""
        print(f"选中项目: {item_data.get('label')} (ID: {item_data.get('id')})")

    def on_item_deselected(self, item_data):
        """处理项目取消选中事件"""
        print(f"取消选中项目: {item_data.get('label')} (ID: {item_data.get('id')})")

    def on_item_double_clicked(self, item_data):
        """处理项目双击事件"""
        print(f"双击项目: {item_data.get('label')} (ID: {item_data.get('id')})")
        # 在这里可以执行特定操作，如定位到项目、编辑项目等


def main():
    """程序入口"""
    # 创建应用程序
    app = QApplication(sys.argv)

    # 设置高DPI缩放
    app.setAttribute(Qt.AA_EnableHighDpiScaling)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps)

    # 创建并显示主窗口
    demo = DemoApp()
    demo.show()

    # 运行应用程序
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
