"""
主题样式定义模块，提供明亮和暗黑两套主题样式。
"""
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import Qt

# 明亮主题样式表
LIGHT_STYLE = """
QMainWindow, QDialog {
    background-color: #f5f5f5;
}

QMenuBar {
    background-color: #fafafa;
    color: #212121;
}

QMenuBar::item:selected {
    background-color: #e0e0e0;
}

QMenu {
    background-color: #fafafa;
    color: #212121;
    border: 1px solid #bdbdbd;
}

QMenu::item:selected {
    background-color: #e0e0e0;
}

QToolBar {
    background-color: #fafafa;
    border-bottom: 1px solid #e0e0e0;
    spacing: 2px;
}

QStatusBar {
    background-color: #f5f5f5;
    color: #616161;
}

QListWidget, QTreeView {
    background-color: #ffffff;
    alternate-background-color: #f5f5f5;
    border: 1px solid #e0e0e0;
}

QListWidget::item:hover, QTreeView::item:hover {
    background-color: #f5f5f5;
}

QListWidget::item:selected, QTreeView::item:selected {
    background-color: #e3f2fd;
    color: #2196f3;
}

QDockWidget {
    border: 1px solid #e0e0e0;
    titlebar-close-icon: url(:/close.png);
}

QDockWidget::title {
    background-color: #fafafa;
    padding-left: 5px;
    padding-top: 2px;
}

QPushButton {
    background-color: #f5f5f5;
    border: 1px solid #e0e0e0;
    border-radius: 3px;
    padding: 4px 12px;
    color: #212121;
}

QPushButton:hover {
    background-color: #e0e0e0;
}

QPushButton:pressed {
    background-color: #bdbdbd;
}

QLabel {
    color: #212121;
}

QLineEdit, QComboBox {
    background-color: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 3px;
    padding: 3px;
    color: #212121;
}

QLineEdit:focus, QComboBox:focus {
    border: 1px solid #2196f3;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QScrollBar:vertical {
    background-color: #f5f5f5;
    width: 12px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background-color: #bdbdbd;
    min-height: 20px;
    border-radius: 3px;
}

QScrollBar::handle:vertical:hover {
    background-color: #9e9e9e;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    background-color: #f5f5f5;
    height: 12px;
    margin: 0px;
}

QScrollBar::handle:horizontal {
    background-color: #bdbdbd;
    min-width: 20px;
    border-radius: 3px;
}

QScrollBar::handle:horizontal:hover {
    background-color: #9e9e9e;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

QTabWidget::pane {
    border: 1px solid #e0e0e0;
}

QTabBar::tab {
    background-color: #f5f5f5;
    border: 1px solid #e0e0e0;
    padding: 5px 10px;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background-color: #ffffff;
    border-bottom-color: #ffffff;
}

QHeaderView::section {
    background-color: #f5f5f5;
    padding: 3px;
    border: 1px solid #e0e0e0;
}

QToolTip {
    background-color: #fafafa;
    color: #212121;
    border: 1px solid #e0e0e0;
}
"""

# 暗黑主题样式表
DARK_STYLE = """
QMainWindow, QDialog {
    background-color: #212121;
}

QMenuBar {
    background-color: #292929;
    color: #e0e0e0;
}

QMenuBar::item:selected {
    background-color: #3d3d3d;
}

QMenu {
    background-color: #292929;
    color: #e0e0e0;
    border: 1px solid #3d3d3d;
}

QMenu::item:selected {
    background-color: #3d3d3d;
}

QToolBar {
    background-color: #292929;
    border-bottom: 1px solid #3d3d3d;
    spacing: 2px;
}

QStatusBar {
    background-color: #212121;
    color: #9e9e9e;
}

QListWidget, QTreeView {
    background-color: #292929;
    alternate-background-color: #252525;
    border: 1px solid #3d3d3d;
    color: #e0e0e0;
}

QListWidget::item:hover, QTreeView::item:hover {
    background-color: #333333;
}

QListWidget::item:selected, QTreeView::item:selected {
    background-color: #3d3d3d;
    color: #2196f3;
}

QDockWidget {
    border: 1px solid #3d3d3d;
    titlebar-close-icon: url(:/close.png);
    color: #e0e0e0;
}

QDockWidget::title {
    background-color: #292929;
    padding-left: 5px;
    padding-top: 2px;
}

QPushButton {
    background-color: #3d3d3d;
    border: 1px solid #5d5d5d;
    border-radius: 3px;
    padding: 4px 12px;
    color: #e0e0e0;
}

QPushButton:hover {
    background-color: #4d4d4d;
}

QPushButton:pressed {
    background-color: #5d5d5d;
}

QLabel {
    color: #e0e0e0;
}

QLineEdit, QComboBox {
    background-color: #3d3d3d;
    border: 1px solid #5d5d5d;
    border-radius: 3px;
    padding: 3px;
    color: #e0e0e0;
}

QLineEdit:focus, QComboBox:focus {
    border: 1px solid #2196f3;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QScrollBar:vertical {
    background-color: #292929;
    width: 12px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background-color: #5d5d5d;
    min-height: 20px;
    border-radius: 3px;
}

QScrollBar::handle:vertical:hover {
    background-color: #6d6d6d;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    background-color: #292929;
    height: 12px;
    margin: 0px;
}

QScrollBar::handle:horizontal {
    background-color: #5d5d5d;
    min-width: 20px;
    border-radius: 3px;
}

QScrollBar::handle:horizontal:hover {
    background-color: #6d6d6d;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

QTabWidget::pane {
    border: 1px solid #3d3d3d;
}

QTabBar::tab {
    background-color: #292929;
    border: 1px solid #3d3d3d;
    padding: 5px 10px;
    margin-right: 2px;
    color: #e0e0e0;
}

QTabBar::tab:selected {
    background-color: #3d3d3d;
    border-bottom-color: #3d3d3d;
}

QHeaderView::section {
    background-color: #292929;
    padding: 3px;
    border: 1px solid #3d3d3d;
    color: #e0e0e0;
}

QToolTip {
    background-color: #292929;
    color: #e0e0e0;
    border: 1px solid #3d3d3d;
}

QWidget {
    color: #e0e0e0;
}
"""

# 明亮主题调色板


def get_light_palette():
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(245, 245, 245))
    palette.setColor(QPalette.WindowText, QColor(33, 33, 33))
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ToolTipText, QColor(33, 33, 33))
    palette.setColor(QPalette.Text, QColor(33, 33, 33))
    palette.setColor(QPalette.Button, QColor(245, 245, 245))
    palette.setColor(QPalette.ButtonText, QColor(33, 33, 33))
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(33, 150, 243))
    palette.setColor(QPalette.Highlight, QColor(33, 150, 243))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    return palette

# 暗黑主题调色板


def get_dark_palette():
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(33, 33, 33))
    palette.setColor(QPalette.WindowText, QColor(224, 224, 224))
    palette.setColor(QPalette.Base, QColor(41, 41, 41))
    palette.setColor(QPalette.AlternateBase, QColor(45, 45, 45))
    palette.setColor(QPalette.ToolTipBase, QColor(41, 41, 41))
    palette.setColor(QPalette.ToolTipText, QColor(224, 224, 224))
    palette.setColor(QPalette.Text, QColor(224, 224, 224))
    palette.setColor(QPalette.Button, QColor(61, 61, 61))
    palette.setColor(QPalette.ButtonText, QColor(224, 224, 224))
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(33, 150, 243))
    palette.setColor(QPalette.Highlight, QColor(33, 150, 243))
    palette.setColor(QPalette.HighlightedText, QColor(41, 41, 41))
    return palette
