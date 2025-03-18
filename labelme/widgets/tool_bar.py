from PyQt5 import QtCore
from PyQt5 import QtWidgets


class ToolBar(QtWidgets.QToolBar):
    def __init__(self, title):
        super(ToolBar, self).__init__(title)
        layout = self.layout()
        m = (0, 0, 0, 0)
        layout.setSpacing(0)
        layout.setContentsMargins(*m)
        self.setContentsMargins(*m)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.FramelessWindowHint)
        # 设置可调整大小
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred,
                           QtWidgets.QSizePolicy.Expanding)
        
        # 允许工具栏在所有区域停靠
        self.setAllowedAreas(
            QtCore.Qt.LeftToolBarArea |
            QtCore.Qt.RightToolBarArea |
            QtCore.Qt.TopToolBarArea |
            QtCore.Qt.BottomToolBarArea
        )
        # 设置工具栏可移动
        self.setMovable(True)
        # 设置浮动
        self.setFloatable(True)

    def addAction(self, action):
        if isinstance(action, QtWidgets.QWidgetAction):
            return super(ToolBar, self).addAction(action)
        btn = QtWidgets.QToolButton()
        btn.setDefaultAction(action)
        btn.setToolButtonStyle(self.toolButtonStyle())
        self.addWidget(btn)

        # center align
        for i in range(self.layout().count()):
            if isinstance(self.layout().itemAt(i).widget(), QtWidgets.QToolButton):
                self.layout().itemAt(i).setAlignment(QtCore.Qt.AlignCenter)
