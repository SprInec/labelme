"""
工具类，用于将QColor转换为HTML颜色代码格式
"""
from PyQt5.QtGui import QColor


class ShapeToColor:
    @staticmethod
    def hex_color(color):
        """将QColor转换为16进制HTML颜色代码"""
        if not color:
            return "#000000"

        if isinstance(color, QColor):
            return '#{:02x}{:02x}{:02x}'.format(*color.getRgb()[:3])
        elif isinstance(color, tuple) and len(color) >= 3:
            return '#{:02x}{:02x}{:02x}'.format(*color[:3])
        else:
            return "#000000"


# 创建一个实例供直接导入使用
shape_to_color = ShapeToColor()
