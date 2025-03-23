# -*- coding: utf-8 -*-

import functools
import html
import math
import os
import os.path as osp
import re
import webbrowser
import colorsys
import random
import yaml
import PIL.Image
import json
import logging

import imgviz
import natsort
import numpy as np
from loguru import logger
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

from labelme import __appname__
from labelme import PY2
from labelme._automation import bbox_from_text
from labelme._automation import object_detection
from labelme._automation import pose_estimation
from labelme._automation.config_loader import ConfigLoader
from labelme.config import get_config
from labelme.label_file import LabelFile
from labelme.label_file import LabelFileError
from labelme.shape import Shape
from labelme.widgets import AiPromptWidget
from labelme.widgets import AISettingsDialog
from labelme.widgets import BrightnessContrastDialog
from labelme.widgets import Canvas
from labelme.widgets import FileDialogPreview
from labelme.widgets import LabelDialog
from labelme.widgets import LabelTreeWidget
from labelme.widgets import LabelTreeWidgetItem
from labelme.widgets import ToolBar
from labelme.widgets import UniqueLabelTreeWidget
from labelme.widgets import ZoomWidget
from labelme.widgets.ai_settings_dialog import AISettingsDialog
from labelme.widgets.shortcuts_dialog import ShortcutsDialog
from labelme.widgets.unique_label_tree_widget import UniqueLabelTreeWidget
from labelme.widgets.label_tree_widget import LabelTreeWidgetItem

from . import utils
import labelme.styles  # 导入样式模块

# FIXME
# - [medium] Set max zoom value to something big enough for FitWidth/Window

# TODO(unknown):
# - Zoom is too "steppy".


LABEL_COLORMAP = imgviz.label_colormap()


class MainWindow(QtWidgets.QMainWindow):
    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = 0, 1, 2

    def __init__(
        self,
        config=None,
        filename=None,
        output=None,
        output_file=None,
        output_dir=None,
    ):
        if output is not None:
            logger.warning(
                "argument output is deprecated, use output_file instead"
            )
            if output_file is None:
                output_file = output

        # see labelme/config/default_config.yaml for valid configuration
        if config is None:
            config = get_config()
        self._config = config

        # 确保自动保存默认开启，同时保存图像数据默认关闭
        self._config["auto_save"] = True
        self._config["store_data"] = False

        # 初始化输出目录
        if output_dir is not None:
            self.output_dir = output_dir
            self._config["output_dir"] = output_dir
        elif self._config.get("output_dir") and osp.exists(self._config["output_dir"]):
            self.output_dir = self._config["output_dir"]
        else:
            self.output_dir = None

        # set default shape colors
        Shape.line_color = QtGui.QColor(*self._config["shape"]["line_color"])
        Shape.fill_color = QtGui.QColor(*self._config["shape"]["fill_color"])
        Shape.select_line_color = QtGui.QColor(
            *self._config["shape"]["select_line_color"]
        )
        Shape.select_fill_color = QtGui.QColor(
            *self._config["shape"]["select_fill_color"]
        )
        Shape.vertex_fill_color = QtGui.QColor(
            *self._config["shape"]["vertex_fill_color"]
        )
        Shape.hvertex_fill_color = QtGui.QColor(
            *self._config["shape"]["hvertex_fill_color"]
        )

        # 初始化显示标签名称设置为False
        self._showLabelNames = False
        Shape.show_label_names = False

        # 初始化标签内容显示选项
        self._showLabelText = True
        self._showLabelGID = True
        self._showLabelDesc = False
        Shape.show_label_text = True
        Shape.show_label_gid = True
        Shape.show_label_desc = False

        # 添加用于记住上一次标签的变量
        self._previous_label_text = None

        # Set point size from config file
        Shape.point_size = self._config["shape"]["point_size"]

        # 初始化主题设置
        self.currentTheme = self._config.get("theme", "light")

        super(MainWindow, self).__init__()
        self.setWindowTitle(__appname__)

        # Whether we need to save or not.
        self.dirty = False

        self._noSelectionSlot = False

        self._copied_shapes = None

        # Main widgets and related state.
        self.labelDialog = LabelDialog(
            text=self.tr("Enter object label"),
            parent=self,
            labels=self._config["labels"],
            sort_labels=self._config["sort_labels"],
            show_text_field=self._config["show_label_text_field"],
            completion=self._config["label_completion"],
            fit_to_content={"row": True, "column": True},
            flags=self._config["label_flags"],
            app=self,  # 添加app参数传递self引用
        )

        # 将LabelListWidget替换为LabelTreeWidget
        self.labelList = LabelTreeWidget(is_dark=(self.currentTheme == "dark"))
        self.lastOpenDir = None

        self.flag_dock = self.flag_widget = None
        self.flag_dock = QtWidgets.QDockWidget(self.tr("标记"), self)
        self.flag_dock.setObjectName("Flags")
        self.flag_widget = QtWidgets.QListWidget()
        if config["flags"]:
            self.loadFlags({k: False for k in config["flags"]})
        self.flag_dock.setWidget(self.flag_widget)
        self.flag_widget.itemChanged.connect(self.setDirty)

        self.labelList.itemSelectionChanged.connect(self.labelSelectionChanged)
        self.labelList.itemDoubleClicked.connect(self._edit_label)
        # self.labelList.itemChanged.connect(self.labelItemChanged)  # LabelTreeWidget没有这个信号
        # self.labelList.itemDropped.connect(self.labelOrderChanged)  # LabelTreeWidget没有这个信号
        self.shape_dock = QtWidgets.QDockWidget(
            self.tr("多边形标签"), self)
        self.shape_dock.setObjectName("Labels")
        self.shape_dock.setWidget(self.labelList)

        # 使用树形结构的标签列表替代原来的列表
        self.uniqLabelList = UniqueLabelTreeWidget(
            is_dark=(self.currentTheme == "dark"))
        self.uniqLabelList.setToolTip(
            self.tr(
                "Select label to start annotating for it. "
                "Press 'Esc' to deselect."
            )
        )
        if self._config["labels"]:
            for label in self._config["labels"]:
                item = self.uniqLabelList.createItemFromLabel(label)
                self.uniqLabelList.addItem(item)
                rgb = self._get_rgb_by_label(label)
                self.uniqLabelList.setItemLabel(item, label, rgb)
        # 连接标签选择变化信号
        self.uniqLabelList.itemSelectionChanged.connect(
            self.labelItemSelectedForDrawing)
        self.label_dock = QtWidgets.QDockWidget(self.tr("标签列表"), self)
        self.label_dock.setObjectName("Label List")
        self.label_dock.setWidget(self.uniqLabelList)

        self.fileSearch = QtWidgets.QLineEdit()
        self.fileSearch.setPlaceholderText(self.tr("Search Filename"))
        self.fileSearch.textChanged.connect(self.fileSearchChanged)
        self.fileListWidget = QtWidgets.QListWidget()
        self.fileListWidget.itemSelectionChanged.connect(
            self.fileSelectionChanged)
        fileListLayout = QtWidgets.QVBoxLayout()
        fileListLayout.setContentsMargins(0, 0, 0, 0)
        fileListLayout.setSpacing(0)
        fileListLayout.addWidget(self.fileSearch)
        fileListLayout.addWidget(self.fileListWidget)
        self.file_dock = QtWidgets.QDockWidget(self.tr("文件列表"), self)
        self.file_dock.setObjectName("Files")
        fileListWidget = QtWidgets.QWidget()
        fileListWidget.setLayout(fileListLayout)
        self.file_dock.setWidget(fileListWidget)

        self.zoomWidget = ZoomWidget()
        self.setAcceptDrops(True)

        self.canvas = self.labelList.canvas = Canvas(
            epsilon=self._config["epsilon"],
            double_click=self._config["canvas"]["double_click"],
            num_backups=self._config["canvas"]["num_backups"],
            crosshair=self._config["canvas"]["crosshair"],
        )
        self.canvas.zoomRequest.connect(self.zoomRequest)
        self.canvas.mouseMoved.connect(
            lambda pos: self.status(f"Mouse is at: x={pos.x()}, y={pos.y()}")
        )

        scrollArea = QtWidgets.QScrollArea()
        scrollArea.setWidget(self.canvas)
        scrollArea.setWidgetResizable(True)
        self.scrollBars = {
            Qt.Vertical: scrollArea.verticalScrollBar(),
            Qt.Horizontal: scrollArea.horizontalScrollBar(),
        }
        self.canvas.scrollRequest.connect(self.scrollRequest)

        self.canvas.newShape.connect(self.newShape)
        self.canvas.shapeMoved.connect(self.setDirty)
        self.canvas.selectionChanged.connect(self.shapeSelectionChanged)
        self.canvas.drawingPolygon.connect(self.toggleDrawingSensitive)
        self.canvas.toggleVisibilityRequest.connect(
            self.toggleShapesVisibility)  # 连接新的信号
        self.canvas.editLabelRequest.connect(self._edit_label)  # 连接双击编辑标签信号

        self.setCentralWidget(scrollArea)

        features = QtWidgets.QDockWidget.DockWidgetFeatures()
        for dock in ["flag_dock", "label_dock", "shape_dock", "file_dock"]:
            if self._config[dock]["closable"]:
                features = features | QtWidgets.QDockWidget.DockWidgetClosable
            if self._config[dock]["floatable"]:
                features = features | QtWidgets.QDockWidget.DockWidgetFloatable
            if self._config[dock]["movable"]:
                features = features | QtWidgets.QDockWidget.DockWidgetMovable
            getattr(self, dock).setFeatures(features)
            if self._config[dock]["show"] is False:
                getattr(self, dock).setVisible(False)

        self.addDockWidget(Qt.RightDockWidgetArea, self.flag_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.label_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.shape_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.file_dock)

        # Actions
        action = functools.partial(utils.newAction, self)
        shortcuts = self._config["shortcuts"]
        quit = action(
            self.tr("&Quit"),
            self.close,
            shortcuts["quit"],
            "quit",
            self.tr("Quit application"),
        )
        open_ = action(
            self.tr("&Open\n"),
            self.openFile,
            shortcuts["open"],
            "icons8-image-64",
            self.tr("Open image or label file"),
        )
        opendir = action(
            self.tr("Open Dir"),
            self.openDirDialog,
            shortcuts["open_dir"],
            "icons8-folder-64",
            self.tr("Open Dir"),
        )
        openNextImg = action(
            self.tr("&Next Image"),
            self.openNextImg,
            shortcuts["open_next"],
            "next",
            self.tr("Open next (hold Ctl+Shift to copy labels)"),
            enabled=False,
        )
        openPrevImg = action(
            self.tr("&Prev Image"),
            self.openPrevImg,
            shortcuts["open_prev"],
            "prev",
            self.tr("Open prev (hold Ctl+Shift to copy labels)"),
            enabled=False,
        )
        save = action(
            self.tr("&Save\n"),
            self.saveFile,
            shortcuts["save"],
            "icons8-save-60",
            self.tr("Save labels to file"),
            enabled=False,
        )
        saveAs = action(
            self.tr("&Save As"),
            self.saveFileAs,
            shortcuts["save_as"],
            "save-as",
            self.tr("Save labels to a different file"),
            enabled=False,
        )

        deleteFile = action(
            self.tr("&Delete File"),
            self.deleteFile,
            shortcuts["delete_file"],
            "icons8-delete-48",
            self.tr("Delete current label file"),
            enabled=False,
        )

        changeOutputDir = action(
            self.tr("输出路径"),
            slot=self.changeOutputDirDialog,
            shortcut=shortcuts["save_to"],
            icon="icons8-file-64",
            tip=self.tr("Change where annotations are loaded/saved"),
        )

        saveAuto = action(
            text=self.tr("自动保存"),
            slot=self.toggleAutoSave,
            tip=self.tr("自动保存标注文件"),
            checkable=True,
            enabled=True,
        )
        # 默认开启自动保存
        saveAuto.setChecked(True)

        saveWithImageData = action(
            text=self.tr("同时保存图像数据"),
            slot=self.enableSaveImageWithData,
            tip=self.tr("在标注文件中保存图像数据"),
            checkable=True,
            checked=False,  # 默认关闭同时保存图像数据
        )

        close = action(
            self.tr("&Close"),
            self.closeFile,
            shortcuts["close"],
            "close",
            self.tr("Close current file"),
        )

        toggle_keep_prev_mode = action(
            self.tr("Keep Previous Annotation"),
            self.toggleKeepPrevMode,
            shortcuts["toggle_keep_prev_mode"],
            None,
            self.tr('Toggle "keep previous annotation" mode'),
            checkable=True,
        )
        toggle_keep_prev_mode.setChecked(self._config["keep_prev"])

        createMode = action(
            self.tr("Create Polygons"),
            lambda: self.toggleDrawMode(False, createMode="polygon"),
            shortcuts["create_polygon"],
            "icons8-polygon-100",
            self.tr("Start drawing polygons"),
            enabled=False,
        )
        createRectangleMode = action(
            self.tr("Create Rectangle"),
            lambda: self.toggleDrawMode(False, createMode="rectangle"),
            shortcuts["create_rectangle"],
            "icons8-rectangular-90",
            self.tr("Start drawing rectangles"),
            enabled=False,
        )
        createCircleMode = action(
            self.tr("Create Circle"),
            lambda: self.toggleDrawMode(False, createMode="circle"),
            shortcuts["create_circle"],
            "icons8-circle-50",
            self.tr("Start drawing circles"),
            enabled=False,
        )
        createLineMode = action(
            self.tr("Create Line"),
            lambda: self.toggleDrawMode(False, createMode="line"),
            shortcuts["create_line"],
            "icons8-line-50",
            self.tr("Start drawing lines"),
            enabled=False,
        )
        createPointMode = action(
            self.tr("Create Point"),
            lambda: self.toggleDrawMode(False, createMode="point"),
            shortcuts["create_point"],
            "icons8-point-100",
            self.tr("Start drawing points"),
            enabled=False,
        )
        createLineStripMode = action(
            self.tr("Create LineStrip"),
            lambda: self.toggleDrawMode(False, createMode="linestrip"),
            shortcuts["create_linestrip"],
            "icons8-polyline-100",
            self.tr("Start drawing linestrip. Ctrl+LeftClick ends creation."),
            enabled=False,
        )
        createAiPolygonMode = action(
            self.tr("Create AI-Polygon"),
            lambda: self.toggleDrawMode(False, createMode="ai_polygon"),
            None,
            "icons8-radar-plot-50",
            self.tr("Start drawing ai_polygon. Ctrl+LeftClick ends creation."),
            enabled=False,
        )
        createAiPolygonMode.changed.connect(
            lambda: self.canvas.initializeAiModel(
                model_name=self._config["ai"].get(
                    "default", "sam:latest")
            )
            if self.canvas.createMode == "ai_polygon"
            else None
        )
        createAiMaskMode = action(
            self.tr("Create AI-Mask"),
            lambda: self.toggleDrawMode(False, createMode="ai_mask"),
            None,
            "icons8-layer-mask-50",
            self.tr("Start drawing ai_mask. Ctrl+LeftClick ends creation."),
            enabled=False,
        )
        createAiMaskMode.changed.connect(
            lambda: self.canvas.initializeAiModel(
                model_name=self._config["ai"].get(
                    "default", "sam:latest")
            )
            if self.canvas.createMode == "ai_mask"
            else None
        )
        editMode = action(
            self.tr("Edit Polygons"),
            self.setEditMode,
            shortcuts["edit_polygon"],
            "icons8-compose-100",
            self.tr("Move and edit the selected polygons"),
            enabled=False,
        )

        delete = action(
            self.tr("Delete Polygons"),
            self.deleteSelectedShape,
            shortcuts["delete_polygon"],
            "icons8-delete-48",
            self.tr("Delete the selected polygons"),
            enabled=False,
        )
        duplicate = action(
            self.tr("Duplicate Polygons"),
            self.duplicateSelectedShape,
            shortcuts["duplicate_polygon"],
            "copy",
            self.tr("Create a duplicate of the selected polygons"),
            enabled=False,
        )
        copy = action(
            self.tr("Copy Polygons"),
            self.copySelectedShape,
            shortcuts["copy_polygon"],
            "icons8-copy-32",
            self.tr("Copy selected polygons to clipboard"),
            enabled=False,
        )
        paste = action(
            self.tr("Paste Polygons"),
            self.pasteSelectedShape,
            shortcuts["paste_polygon"],
            "paste",
            self.tr("Paste copied polygons"),
            enabled=False,
        )
        undoLastPoint = action(
            self.tr("Undo last point"),
            self.canvas.undoLastPoint,
            shortcuts["undo_last_point"],
            "icons8-undo-60",
            self.tr("Undo last drawn point"),
            enabled=False,
        )
        removePoint = action(
            text=self.tr("Remove Selected Point"),
            slot=self.removeSelectedPoint,
            shortcut=shortcuts["remove_selected_point"],
            icon="edit",
            tip=self.tr("Remove selected point from polygon"),
            enabled=False,
        )

        undo = action(
            self.tr("Undo\n"),
            self.undoShapeEdit,
            shortcuts["undo"],
            "icons8-undo-60",
            self.tr("Undo last add and edit of shape"),
            enabled=False,
        )

        hideAll = action(
            self.tr("&Hide\nPolygons"),
            functools.partial(self.togglePolygons, False),
            shortcuts["hide_all_polygons"],
            icon="icons8-eye-64",
            tip=self.tr("Hide all polygons"),
            enabled=False,
        )
        showAll = action(
            self.tr("&Show\nPolygons"),
            functools.partial(self.togglePolygons, True),
            shortcuts["show_all_polygons"],
            icon="icons8-eye-64",
            tip=self.tr("Show all polygons"),
            enabled=False,
        )
        toggleAll = action(
            self.tr("&Toggle\nPolygons"),
            functools.partial(self.togglePolygons, None),
            shortcuts["toggle_all_polygons"],
            icon="icons8-eye-64",
            tip=self.tr("Toggle all polygons"),
            enabled=False,
        )

        help = action(
            self.tr("&Tutorial"),
            self.tutorial,
            icon="help",
            tip=self.tr("Show tutorial page"),
        )

        zoom = QtWidgets.QWidgetAction(self)
        zoomBoxLayout = QtWidgets.QVBoxLayout()
        zoomLabel = QtWidgets.QLabel(self.tr("Zoom"))
        zoomLabel.setAlignment(Qt.AlignCenter)
        zoomBoxLayout.addWidget(zoomLabel)
        zoomBoxLayout.addWidget(self.zoomWidget)
        zoom.setDefaultWidget(QtWidgets.QWidget())
        zoom.defaultWidget().setLayout(zoomBoxLayout)
        self.zoomWidget.setWhatsThis(
            str(
                self.tr(
                    "Zoom in or out of the image. Also accessible with "
                    "{} and {} from the canvas."
                )
            ).format(
                utils.fmtShortcut(
                    "{},{}".format(shortcuts["zoom_in"], shortcuts["zoom_out"])
                ),
                utils.fmtShortcut(self.tr("Ctrl+Wheel")),
            )
        )
        self.zoomWidget.setEnabled(False)

        zoomIn = action(
            self.tr("Zoom &In"),
            functools.partial(self.addZoom, 1.1),
            shortcuts["zoom_in"],
            "zoom-in",
            self.tr("Increase zoom level"),
            enabled=False,
        )
        zoomOut = action(
            self.tr("&Zoom Out"),
            functools.partial(self.addZoom, 0.9),
            shortcuts["zoom_out"],
            "zoom-out",
            self.tr("Decrease zoom level"),
            enabled=False,
        )
        zoomOrg = action(
            self.tr("&Original size"),
            functools.partial(self.setZoom, 100),
            shortcuts["zoom_to_original"],
            "zoom",
            self.tr("Zoom to original size"),
            enabled=False,
        )
        keepPrevScale = action(
            self.tr("&Keep Previous Scale"),
            self.enableKeepPrevScale,
            tip=self.tr("Keep previous zoom scale"),
            checkable=True,
            checked=self._config["keep_prev_scale"],
            enabled=True,
        )
        fitWindow = action(
            self.tr("&Fit Window"),
            self.setFitWindow,
            shortcuts["fit_window"],
            "fit-window",
            self.tr("Zoom follows window size"),
            checkable=True,
            enabled=False,
        )
        fitWidth = action(
            self.tr("Fit &Width"),
            self.setFitWidth,
            shortcuts["fit_width"],
            "fit-width",
            self.tr("Zoom follows window width"),
            checkable=True,
            enabled=False,
        )
        brightnessContrast = action(
            self.tr("&Brightness Contrast"),
            self.brightnessContrast,
            None,
            "color",
            self.tr("Adjust brightness and contrast"),
            enabled=False,
        )

        # 主题相关动作
        lightTheme = action(
            self.tr("明亮主题"),
            self.setLightTheme,
            None,
            "color",
            self.tr("切换至明亮主题"),
            checkable=True,
            checked=self.currentTheme == "light",
        )

        darkTheme = action(
            self.tr("暗黑主题"),
            self.setDarkTheme,
            None,
            "color-fill",
            self.tr("切换至暗黑主题"),
            checkable=True,
            checked=self.currentTheme == "dark",
        )

        defaultTheme = action(
            self.tr("原始主题"),
            self.setDefaultTheme,
            None,
            "color-fill",
            self.tr("恢复原始主题"),
            checkable=True,
            checked=self.currentTheme == "default",
        )

        # 创建主题切换动作组，确保只有一个主题被选中
        themeActionGroup = QtWidgets.QActionGroup(self)
        themeActionGroup.setExclusive(True)
        themeActionGroup.addAction(lightTheme)
        themeActionGroup.addAction(darkTheme)
        themeActionGroup.addAction(defaultTheme)

        # AI设置
        ai_settings = action(
            self.tr("半自动标注配置"),
            self.openAISettings,
            None,
            "settings",
            self.tr("配置半自动标注功能"),
            enabled=True,
        )

        runObjectDetection = action(
            self.tr("目标检测"),
            self.runObjectDetection,
            None,
            "icons8-facial-recognition-100",  # 使用facial-recognition图标
            self.tr("使用AI检测图像中的对象"),
            enabled=False,
        )

        runPoseEstimation = action(
            self.tr("姿态估计"),
            self.runPoseEstimation,
            None,
            "icons8-natural-user-interface-1-100",  # 使用natural-user-interface图标
            self.tr("检测图像中的人体姿态"),
            enabled=False,
        )

        submitAiPrompt = action(
            self.tr("提交AI提示"),
            lambda: self._submit_ai_prompt(None),
            None,
            "icons8-done-64",
            self.tr("使用AI提示检测对象"),
            enabled=False,
        )

        # Group zoom controls into a list for easier toggling.
        zoomActions = (
            self.zoomWidget,
            zoomIn,
            zoomOut,
            zoomOrg,
            fitWindow,
            fitWidth,
        )
        self.zoomMode = self.FIT_WINDOW
        fitWindow.setChecked(Qt.Checked)
        self.scalers = {
            self.FIT_WINDOW: self.scaleFitWindow,
            self.FIT_WIDTH: self.scaleFitWidth,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }

        edit = action(
            self.tr("&Edit Label"),
            self._edit_label,
            shortcuts["edit_label"],
            "icons8-label-50",
            self.tr("Modify the label of the selected polygon"),
            enabled=False,
        )

        fill_drawing = action(
            self.tr("Fill Drawing Polygon"),
            self.canvas.setFillDrawing,
            None,
            "color",
            self.tr("Fill polygon while drawing"),
            checkable=True,
            enabled=True,
        )
        if self._config["canvas"]["fill_drawing"]:
            fill_drawing.trigger()

        # 添加显示标签名称相关选项
        # 主选项 - 显示标签名称
        showLabelNames = self.createDockLikeAction(
            self.tr("显示标签名称"),
            self.toggleShowLabelNames,
            False  # 默认未选中
        )

        # 子选项 - 显示标签文本
        showLabelText = self.createDockLikeAction(
            self.tr("　显示标签文本"),  # 前面加空格表示层级
            self.toggleShowLabelText,
            True  # 默认选中
        )
        showLabelText.setEnabled(False)  # 初始禁用，因为父选项未选中

        # 子选项 - 显示GID
        showLabelGID = self.createDockLikeAction(
            self.tr("　显示GID"),  # 前面加空格表示层级
            self.toggleShowLabelGID,
            True  # 默认选中
        )
        showLabelGID.setEnabled(False)  # 初始禁用

        # 子选项 - 显示描述
        showLabelDesc = self.createDockLikeAction(
            self.tr("　显示描述"),  # 前面加空格表示层级
            self.toggleShowLabelDesc,
            False  # 默认不选中
        )
        showLabelDesc.setEnabled(False)  # 初始禁用

        # 保存到实例变量中便于访问
        self.showLabelNames = showLabelNames
        self.showLabelText = showLabelText
        self.showLabelGID = showLabelGID
        self.showLabelDesc = showLabelDesc
        self.labelNameOptions = [showLabelText, showLabelGID, showLabelDesc]

        # Label list context menu.
        labelMenu = QtWidgets.QMenu()
        utils.addActions(labelMenu, (edit, delete))
        self.labelList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.labelList.customContextMenuRequested.connect(
            self.popLabelListMenu)

        # Store actions for further handling.
        self.actions = utils.struct(
            saveAuto=saveAuto,
            saveWithImageData=saveWithImageData,
            changeOutputDir=changeOutputDir,
            save=save,
            saveAs=saveAs,
            open=open_,
            close=close,
            deleteFile=deleteFile,
            toggleKeepPrevMode=toggle_keep_prev_mode,
            delete=delete,
            edit=edit,
            duplicate=duplicate,
            copy=copy,
            paste=paste,
            undoLastPoint=undoLastPoint,
            undo=undo,
            removePoint=removePoint,
            createMode=createMode,
            editMode=editMode,
            createRectangleMode=createRectangleMode,
            createCircleMode=createCircleMode,
            createLineMode=createLineMode,
            createPointMode=createPointMode,
            createLineStripMode=createLineStripMode,
            createAiPolygonMode=createAiPolygonMode,
            createAiMaskMode=createAiMaskMode,
            zoom=zoom,
            zoomIn=zoomIn,
            zoomOut=zoomOut,
            zoomOrg=zoomOrg,
            keepPrevScale=keepPrevScale,
            fitWindow=fitWindow,
            fitWidth=fitWidth,
            brightnessContrast=brightnessContrast,
            zoomActions=zoomActions,
            openNextImg=openNextImg,
            openPrevImg=openPrevImg,
            fileMenuActions=(open_, opendir, save, saveAs, close, quit),
            aiMenuActions=(ai_settings, None, createAiPolygonMode, createAiMaskMode,
                           None, runObjectDetection, runPoseEstimation, submitAiPrompt),
            # showLabelNames line removed to fix error
            lightTheme=lightTheme,  # 添加明亮主题动作
            darkTheme=darkTheme,    # 添加暗黑主题动作
            defaultTheme=defaultTheme,  # 添加原始主题动作
            themeActions=(lightTheme, darkTheme, defaultTheme),  # 添加主题动作组
            tool=(),
            # XXX: need to add some actions here to activate the shortcut
            editMenu=(
                edit,
                duplicate,
                copy,
                paste,
                delete,
                None,
                undo,
                undoLastPoint,
                None,
                removePoint,
                None,
                toggle_keep_prev_mode,
            ),
            # menu shown at right click
            menu=(
                createMode,
                createRectangleMode,
                createCircleMode,
                createLineMode,
                createPointMode,
                createLineStripMode,
                createAiPolygonMode,
                createAiMaskMode,
                editMode,
                edit,
                duplicate,
                copy,
                paste,
                delete,
                undo,
                undoLastPoint,
                removePoint,
            ),
            onLoadActive=(
                close,
                createMode,
                createRectangleMode,
                createCircleMode,
                createLineMode,
                createPointMode,
                createLineStripMode,
                createAiPolygonMode,
                createAiMaskMode,
                editMode,
                brightnessContrast,
                runObjectDetection,  # 添加运行目标检测
                runPoseEstimation,   # 添加运行人体姿态估计
            ),
            onShapesPresent=(saveAs, hideAll, showAll, toggleAll),
        )

        self.canvas.vertexSelected.connect(self.actions.removePoint.setEnabled)

        # 创建菜单
        self.menus = utils.struct(
            file=self.menu(self.tr("&文件")),
            edit=self.menu(self.tr("&编辑")),
            view=self.menu(self.tr("&视图")),
            ai=self.menu(self.tr("&半自动标注")),
            shortcuts=self.menu(self.tr("&快捷键")),
            help=self.menu(self.tr("&帮助")),
            theme=self.menu(self.tr("&主题")),  # 添加主题菜单
            recentFiles=QtWidgets.QMenu(self.tr("打开最近文件")),
            labelList=labelMenu,
        )

        # 设置视图菜单在点击后不关闭，仅在鼠标离开后关闭
        self.menus.view.installEventFilter(self)
        self.menus.view.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)

        # 应用上次保存的主题设置
        if self.currentTheme == "dark":
            self.setDarkTheme(update_actions=False)  # 添加参数，表示不更新动作选中状态
        elif self.currentTheme == "default":
            self.setDefaultTheme(update_actions=False)  # 添加参数，表示不更新动作选中状态
        else:
            self.setLightTheme(update_actions=False)  # 添加参数，表示不更新动作选中状态

        utils.addActions(
            self.menus.file,
            (
                open_,
                openNextImg,
                openPrevImg,
                opendir,
                self.menus.recentFiles,
                save,
                saveAs,
                saveAuto,
                changeOutputDir,
                saveWithImageData,
                close,
                deleteFile,
                None,
                quit,
            ),
        )
        utils.addActions(self.menus.help, (help,))
        utils.addActions(
            self.menus.view,
            (
                self.flag_dock.toggleViewAction(),
                self.label_dock.toggleViewAction(),
                self.shape_dock.toggleViewAction(),
                self.file_dock.toggleViewAction(),
                None,
                fill_drawing,
                showLabelNames,  # 显示标签名称
                showLabelText,   # 显示标签文本
                showLabelGID,    # 显示GID
                showLabelDesc,   # 显示描述
            ),
        )

        # 创建标签云流式布局动作
        self.cloud_layout_action = self.createDockLikeAction(
            self.tr("标签云流式布局"),
            self.toggleLabelCloudLayout,
            self._config.get("label_cloud_layout", False)
        )

        # 将标签云流式布局动作添加到视图菜单
        self.menus.view.addAction(self.cloud_layout_action)

        # 添加其他视图菜单选项
        utils.addActions(
            self.menus.view,
            (
                None,
                hideAll,
                showAll,
                toggleAll,
                None,
                zoomIn,
                zoomOut,
                zoomOrg,
                keepPrevScale,
                None,
                fitWindow,
                fitWidth,
                None,
                brightnessContrast,
            ),
        )

        # 添加AI菜单动作
        utils.addActions(self.menus.ai, self.actions.aiMenuActions)

        # 添加主题相关菜单项
        utils.addActions(
            self.menus.theme,
            self.actions.themeActions
        )

        # 添加快捷键菜单
        shortcuts_menu = action(
            self.tr("快捷键设置"),
            self.openShortcutsDialog,
            None,
            "settings",
            self.tr("自定义快捷键设置"),
        )
        utils.addActions(self.menus.shortcuts, (shortcuts_menu,))

        self.menus.file.aboutToShow.connect(self.updateFileMenu)

        # 自定义菜单栏顺序
        menubar = self.menuBar()
        menubar.clear()
        menubar.addMenu(self.menus.file)
        menubar.addMenu(self.menus.edit)
        menubar.addMenu(self.menus.view)
        menubar.addMenu(self.menus.ai)
        menubar.addMenu(self.menus.theme)  # 添加主题菜单到菜单栏
        menubar.addMenu(self.menus.shortcuts)
        menubar.addMenu(self.menus.help)

        # Custom context menu for the canvas widget:
        utils.addActions(self.canvas.menus[0], self.actions.menu)
        utils.addActions(
            self.canvas.menus[1],
            (
                action("&Copy here", self.copyShape),
                action("&Move here", self.moveShape),
            ),
        )

        # 创建AI提示部件，但不添加到工具栏
        self._ai_prompt_widget: QtWidgets.QWidget = AiPromptWidget(
            on_submit=self._submit_ai_prompt, parent=self
        )
        ai_prompt_action = QtWidgets.QWidgetAction(self)
        ai_prompt_action.setDefaultWidget(self._ai_prompt_widget)

        self.tools = self.toolbar("Tools")
        self.actions.tool = (
            open_,
            opendir,
            changeOutputDir,  # 添加输出路径按钮
            openPrevImg,
            openNextImg,
            save,
            deleteFile,
            None,
            createMode,
            createRectangleMode,
            createPointMode,
            createLineStripMode,
            editMode,
            duplicate,
            delete,
            undo,
            brightnessContrast,
            None,
            runObjectDetection,  # 添加运行目标检测按钮
            runPoseEstimation,   # 添加运行人体姿态估计按钮
            None,
            fitWindow,
            zoom,
        )

        self.statusBar().setStyleSheet(
            "QStatusBar::item {border: none;}")  # 移除状态栏项的边框

        # 创建状态栏进度条
        self.statusProgress = QtWidgets.QProgressBar()
        self.statusProgress.setFixedHeight(16)  # 调整高度使其更现代
        self.statusProgress.setFixedWidth(300)  # 加宽进度条
        self.statusProgress.setTextVisible(False)
        self.statusProgress.hide()  # 默认隐藏进度条

        # 创建状态栏标签用于显示当前模式
        self.modeLabel = QtWidgets.QLabel("编辑模式")
        self.modeLabel.setStyleSheet("padding-right: 10px;")
        self.statusBar().addPermanentWidget(self.modeLabel)
        self.statusBar().addPermanentWidget(self.statusProgress)

        # 连接画布的模式改变信号
        self.canvas.modeChanged.connect(self.updateModeLabel)

        # 添加到状态栏
        self.statusBar().addPermanentWidget(self.statusProgress)

        self.statusBar().showMessage(str(self.tr("%s started.")) % __appname__)
        self.statusBar().show()

        # 设置窗口最小尺寸，避免缩放太小
        self.setMinimumSize(1200, 800)

        # 设置窗口默认最大化
        self.showMaximized()

        if output_file is not None and self._config["auto_save"]:
            logger.warning(
                "If `auto_save` argument is True, `output_file` argument "
                "is ignored and output filename is automatically "
                "set as IMAGE_BASENAME.json."
            )
        self.output_file = output_file
        self.output_dir = output_dir

        # Application state.
        self.image = QtGui.QImage()
        self.imagePath = None
        self.recentFiles = []
        self.maxRecent = 7
        self.otherData = None
        self.zoom_level = 100
        self.fit_window = False
        self.zoom_values = {}  # key=filename, value=(zoom_mode, zoom_value)
        self.brightnessContrast_values = {}
        self.scroll_values = {
            Qt.Horizontal: {},
            Qt.Vertical: {},
        }  # key=filename, value=scroll_value

        if filename is not None and osp.isdir(filename):
            self.importDirImages(filename, load=False)
        else:
            self.filename = filename

        if config["file_search"]:
            self.fileSearch.setText(config["file_search"])
            self.fileSearchChanged()

        # XXX: Could be completely declarative.
        # Restore application settings.
        self.settings = QtCore.QSettings("labelme", "labelme")
        self.recentFiles = self.settings.value("recentFiles", []) or []
        size = self.settings.value("window/size", QtCore.QSize(600, 500))
        position = self.settings.value("window/position", QtCore.QPoint(0, 0))
        state = self.settings.value("window/state", QtCore.QByteArray())
        self.resize(size)
        self.move(position)
        # or simply:
        # self.restoreGeometry(settings['window/geometry']
        self.restoreState(state)

        # Populate the File menu dynamically.
        self.updateFileMenu()
        # Since loading the file may take some time,
        # make sure it runs in the background.
        if self.filename is not None:
            self.queueEvent(functools.partial(self.loadFile, self.filename))

        # Callbacks:
        self.zoomWidget.valueChanged.connect(self.paintCanvas)

        self.populateModeActions()

        # self.firstStart = True
        # if self.firstStart:
        #    QWhatsThis.enterWhatsThisMode()

        # 在UI初始化完成后，加载上次的目录
        if self._config.get("last_dir") and osp.exists(self._config["last_dir"]):
            self.lastOpenDir = self._config["last_dir"]
            self.importDirImages(self._config["last_dir"], load=True)

            # 如果有输出目录，在加载完图像后应用它
            if self._config.get("output_dir") and osp.exists(self._config["output_dir"]):
                self.output_dir = self._config["output_dir"]
                # 确保应用输出目录
                self.statusBar().showMessage(
                    self.tr("输出目录已设置为: %s") % self.output_dir, 5000
                )

                # 重新加载当前文件列表，以显示输出目录中的标注文件
                if self.lastOpenDir and osp.exists(self.lastOpenDir):
                    # 保存当前文件名
                    current_filename = self.filename
                    # 重新加载目录
                    self.importDirImages(self.lastOpenDir, load=False)

                    # 如果存在上次编辑的图片索引，优先跳转到该图片
                    if self._config.get("last_image_index") is not None and self._config["last_image_index"] < len(self.imageList):
                        last_idx = self._config["last_image_index"]
                        if last_idx >= 0 and last_idx < len(self.imageList):
                            self.fileListWidget.setCurrentRow(last_idx)
                            self.fileListWidget.repaint()
                            # 加载该索引的图片
                            self.loadFile(self.imageList[last_idx])
                            # 设置默认适应窗口
                            self.setFitWindow(True)
                            self.actions.fitWindow.setChecked(True)
                            self.adjustScale(initial=True)
                    # 如果没有上次编辑索引或者索引无效，但有当前文件名，则跳转到当前文件
                    elif current_filename and current_filename in self.imageList:
                        self.fileListWidget.setCurrentRow(
                            self.imageList.index(current_filename))
                        self.fileListWidget.repaint()
                        # 重新加载当前文件
                        self.loadFile(current_filename)
                        # 设置默认适应窗口
                        self.setFitWindow(True)
                        self.actions.fitWindow.setChecked(True)
                        self.adjustScale(initial=True)

#         # 添加显示标签名称的动作
#         showLabelNames = action(
#             self.tr("显示标签名称"),
#             self.toggleShowLabelNames,
#             'Ctrl+L',
#             'tag',
#             self.tr("在标注上显示标签名称"),
#             checkable=True,
#         )
#         showLabelNames.setChecked(self._showLabelNames)

    def menu(self, title, actions=None):
        menu = self.menuBar().addMenu(title)
        if actions:
            utils.addActions(menu, actions)
        return menu

    def toolbar(self, title, actions=None):
        toolbar = ToolBar(title)
        toolbar.setObjectName("%sToolBar" % title)
        # toolbar.setOrientation(Qt.Vertical)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        if actions:
            utils.addActions(toolbar, actions)
        self.addToolBar(Qt.TopToolBarArea, toolbar)
        return toolbar

    # Support Functions

    def noShapes(self):
        return not len(self.labelList)

    def populateModeActions(self):
        tool, menu = self.actions.tool, self.actions.menu
        self.tools.clear()
        utils.addActions(self.tools, tool)
        self.canvas.menus[0].clear()
        utils.addActions(self.canvas.menus[0], menu)
        self.menus.edit.clear()
        actions = (
            self.actions.createMode,
            self.actions.createRectangleMode,
            self.actions.createCircleMode,
            self.actions.createLineMode,
            self.actions.createPointMode,
            self.actions.createLineStripMode,
            self.actions.editMode,
        )
        utils.addActions(self.menus.edit, actions + self.actions.editMenu)

    def setDirty(self):
        # Even if we autosave the file, we keep the ability to undo
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)

        if self._config["auto_save"] or self.actions.saveAuto.isChecked():
            label_file = osp.splitext(self.imagePath)[0] + ".json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)
            self.saveLabels(label_file)
            return
        self.dirty = True
        self.actions.save.setEnabled(True)
        title = __appname__
        if self.filename is not None:
            title = "{} - {}*".format(title, self.filename)
        self.setWindowTitle(title)

        # 更新dock标题显示
        self.updateDockTitles()

    def setClean(self):
        self.dirty = False
        self.actions.save.setEnabled(False)
        self.actions.createMode.setEnabled(True)
        self.actions.createRectangleMode.setEnabled(True)
        self.actions.createCircleMode.setEnabled(True)
        self.actions.createLineMode.setEnabled(True)
        self.actions.createPointMode.setEnabled(True)
        self.actions.createLineStripMode.setEnabled(True)
        self.actions.createAiPolygonMode.setEnabled(True)
        self.actions.createAiMaskMode.setEnabled(True)
        title = __appname__
        if self.filename is not None:
            title = "{} - {}".format(title, self.filename)
        self.setWindowTitle(title)

        if self.hasLabelFile():
            self.actions.deleteFile.setEnabled(True)
        else:
            self.actions.deleteFile.setEnabled(False)

    def toggleActions(self, value=True):
        """Enable/Disable widgets which depend on an opened image."""
        for z in self.actions.zoomActions:
            z.setEnabled(value)
        for action in self.actions.onLoadActive:
            action.setEnabled(value)

        # 启用AI相关动作
        if hasattr(self.actions, "aiMenuActions"):
            for action in self.actions.aiMenuActions:
                # 跳过AI设置和分隔符
                if action is not None and action != self.actions.aiMenuActions[0]:
                    action.setEnabled(value)

    def queueEvent(self, function):
        QtCore.QTimer.singleShot(0, function)

    def status(self, message, delay=5000):
        self.statusBar().showMessage(message, delay)

    def _submit_ai_prompt(self, _) -> None:
        # 从配置中获取AI Prompt设置
        ai_config = self._config.get("ai", {})
        prompt_config = ai_config.get("prompt", {})

        # 使用配置中的文本提示，如果为空则使用工具栏中的文本提示
        text_prompt = prompt_config.get("text", "")
        if not text_prompt and hasattr(self, "_ai_prompt_widget"):
            text_prompt = self._ai_prompt_widget.get_text_prompt()

        texts = text_prompt.split(",")
        if not texts or texts[0] == "":
            self.errorMessage(
                self.tr("错误"),
                self.tr("请先设置AI提示文本"),
            )
            return

        # 使用配置中的Score阈值
        score_threshold = prompt_config.get("score_threshold", 0.1)
        if hasattr(self, "_ai_prompt_widget"):
            score_threshold = self._ai_prompt_widget.get_score_threshold()

        # 使用配置中的IoU阈值
        iou_threshold = prompt_config.get("iou_threshold", 0.5)
        if hasattr(self, "_ai_prompt_widget"):
            iou_threshold = self._ai_prompt_widget.get_iou_threshold()

        boxes, scores, labels = bbox_from_text.get_bboxes_from_texts(
            model="yoloworld",
            image=utils.img_qt_to_arr(self.image)[:, :, :3],
            texts=texts,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
        )

        for shape in self.canvas.shapes:
            if shape.shape_type != "rectangle" or shape.label not in texts:
                continue
            box = np.array(
                [
                    shape.points[0].x(),
                    shape.points[0].y(),
                    shape.points[1].x(),
                    shape.points[1].y(),
                ],
                dtype=np.float32,
            )
            boxes = np.r_[boxes, [box]]
            scores = np.r_[scores, [1.01]]
            labels = np.r_[labels, [texts.index(shape.label)]]

        boxes, scores, labels = bbox_from_text.nms_bboxes(
            boxes=boxes,
            scores=scores,
            labels=labels,
            iou_threshold=self._ai_prompt_widget.get_iou_threshold(),
            score_threshold=self._ai_prompt_widget.get_score_threshold(),
            max_num_detections=100,
        )

        keep = scores != 1.01
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        shape_dicts: list[dict] = bbox_from_text.get_shapes_from_bboxes(
            boxes=boxes,
            scores=scores,
            labels=labels,
            texts=texts,
        )

        shapes: list[Shape] = []
        for shape_dict in shape_dicts:
            shape = Shape(
                label=shape_dict["label"],
                shape_type=shape_dict["shape_type"],
                description=shape_dict["description"],
            )
            for point in shape_dict["points"]:
                shape.addPoint(QtCore.QPointF(*point))
            shapes.append(shape)

        self.canvas.storeShapes()
        self.loadShapes(shapes, replace=False)
        self.setDirty()

    def resetState(self):
        self.labelList.clear()
        self.filename = None
        self.imagePath = None
        self.imageData = None
        self.labelFile = None
        self.otherData = {}
        self.canvas.resetState()
        # 更新状态栏的模式标签
        self.updateModeLabel("编辑模式")

    def currentItem(self):
        items = self.labelList.selectedItems()
        if items:
            return items[0]
        return None

    def addRecentFile(self, filename):
        if filename in self.recentFiles:
            self.recentFiles.remove(filename)
        elif len(self.recentFiles) >= self.maxRecent:
            self.recentFiles.pop()
        self.recentFiles.insert(0, filename)

    # Callbacks

    def undoShapeEdit(self):
        self.canvas.restoreShape()
        self.labelList.clear()
        self.loadShapes(self.canvas.shapes)
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)

    def tutorial(self):
        url = "https://github.com/labelmeai/labelme/tree/main/examples/tutorial"  # NOQA
        webbrowser.open(url)

    def toggleDrawingSensitive(self, drawing=True):
        """Toggle drawing sensitive.

        In the middle of drawing, toggling between modes should be disabled.
        """
        self.actions.editMode.setEnabled(not drawing)
        self.actions.undoLastPoint.setEnabled(drawing)
        self.actions.undo.setEnabled(not drawing)
        self.actions.delete.setEnabled(not drawing)

    def toggleDrawMode(self, edit=True, createMode="polygon"):
        draw_actions = {
            "polygon": self.actions.createMode,
            "rectangle": self.actions.createRectangleMode,
            "circle": self.actions.createCircleMode,
            "point": self.actions.createPointMode,
            "line": self.actions.createLineMode,
            "linestrip": self.actions.createLineStripMode,
            "ai_polygon": self.actions.createAiPolygonMode,
            "ai_mask": self.actions.createAiMaskMode,
        }

        # 如果是AI工具，且已经处于该模式，点击相同的AI工具按钮应该退出该模式
        if not edit and createMode in ["ai_polygon", "ai_mask"] and self.canvas.createMode == createMode:
            # 切换到编辑模式
            edit = True
            # 更新UI显示状态
            for draw_action in draw_actions.values():
                draw_action.setEnabled(True)
            self.actions.editMode.setChecked(True)
            self.actions.editMode.setEnabled(False)
            # 设置画布模式
            self.canvas.setEditing(True)
            # 更新状态栏提示
            self.status(self.tr("已退出AI标注模式，切换到编辑模式"))
            return

        # 切换绘制模式时清除标签列表的选中状态
        self.labelList.clearSelection()

        self.canvas.setEditing(edit)
        self.canvas.createMode = createMode
        if edit:
            for draw_action in draw_actions.values():
                draw_action.setEnabled(True)
            # 更新状态栏提示
            self.status(self.tr("已切换到编辑模式"))
        else:
            for draw_mode, draw_action in draw_actions.items():
                draw_action.setEnabled(createMode != draw_mode)
            # 更新状态栏提示
            if createMode in ["ai_polygon", "ai_mask"]:
                tool_name = "AI多边形" if createMode == "ai_polygon" else "AI蒙版"
                self.status(self.tr(f"已切换到{tool_name}标注模式，再次点击可退出"))
        self.actions.editMode.setEnabled(not edit)

    def setEditMode(self):
        self.toggleDrawMode(True)

    def updateFileMenu(self):
        current = self.filename

        def exists(filename):
            return osp.exists(str(filename))

        menu = self.menus.recentFiles
        menu.clear()
        files = [f for f in self.recentFiles if f != current and exists(f)]
        for i, f in enumerate(files):
            icon = utils.newIcon("icons8-label-48")
            action = QtWidgets.QAction(
                icon, "&%d %s" % (i + 1, QtCore.QFileInfo(f).fileName()), self
            )
            action.triggered.connect(functools.partial(self.loadRecent, f))
            menu.addAction(action)

    def popLabelListMenu(self, point):
        self.menus.labelList.exec_(self.labelList.mapToGlobal(point))

    def validateLabel(self, label):
        # no validation
        if self._config["validate_label"] is None:
            return True

        for i in range(self.uniqLabelList.count()):
            label_i = self.uniqLabelList.item(i).data(Qt.UserRole)
            if self._config["validate_label"] in ["exact"]:
                if label_i == label:
                    return True
        return False

    def _edit_label(self, value=None):
        """编辑当前选中形状的标签

        Args:
            value: 可能是双击的形状项或鼠标位置（QtCore.QPoint）
        """
        # 处理从鼠标双击传递过来的位置信息
        mouse_pos = None
        if isinstance(value, QtCore.QPoint):
            mouse_pos = value
            value = None

        shapes = [s for s in self.canvas.selectedShapes if s.selected]
        if not shapes or len(shapes) != 1:
            return

        shape = shapes[0]
        old_label = shape.label
        old_flags = shape.flags
        old_group_id = shape.group_id
        old_description = shape.description

        # 获取形状的颜色
        shape_color = shape.fill_color

        result = self.labelDialog.popUp(
            old_label,
            flags=old_flags,
            group_id=old_group_id,
            description=old_description,
            color=shape_color,  # 传递形状的颜色
            mouse_pos=mouse_pos,  # 传递鼠标位置
        )
        if result is None:
            return

        text, flags, group_id, description, color = result
        if not self.validateLabel(text):
            self.errorMessage(
                self.tr("Invalid label"),
                self.tr("Invalid label '{}' with validation type '{}'").format(
                    text, self._config["validate_label"]
                ),
            )
            return

        color_changed = color is not None and color != shape.fill_color

        # 更新当前形状
        shape.label = text
        shape.flags = flags
        shape.group_id = group_id
        shape.description = description

        # 如果颜色已更改，更新当前形状颜色
        if color_changed:
            self._apply_shape_color(shape, color)

            # 如果标签名称没有变化，更新所有同类同名的标签颜色
            if old_label == text:
                self._update_same_label_colors(text, color)
        else:
            # 如果标签名称发生变化，更新颜色
            if old_label != text:
                self._update_shape_color(shape)

        # 无论颜色是否变化，都更新标签项显示
        item = self.labelList.findItemByShape(shape)
        if item:
            # 更新标签列表中的文本
            display_text = text
            if group_id is not None:
                display_text = "{} ({})".format(text, group_id)

            r, g, b = shape.fill_color.getRgb()[:3]
            colored_text = '{} <font color="#{:02x}{:02x}{:02x}">●</font>'.format(
                html.escape(display_text), r, g, b
            )
            item.setText(colored_text)

        # 更新UI
        self.setDirty()
        self.canvas.update()
        self.labelDialog.edit.setText(text)

    def _apply_shape_color(self, shape, color):
        """应用颜色到形状"""
        shape.line_color = color

        # 根据形状类型设置不同的填充透明度
        if shape.shape_type == "point":
            # 点标签使用更高的填充透明度
            fill_alpha = 120
            # 点标签的选中效果使用白色边框和原色填充
            select_line_color = QtGui.QColor(255, 255, 255)
            select_fill_alpha = 180
        elif shape.shape_type == "rectangle":
            # 矩形标签使用较低的填充透明度
            fill_alpha = 20
            # 矩形选中效果使用原色边框，保持与悬停效果一致
            select_line_color = color.lighter(120)
            select_fill_alpha = 15  # 大幅降低选中时的透明度
        else:
            fill_alpha = 30
            select_line_color = QtGui.QColor(255, 255, 255)
            select_fill_alpha = 80

        shape.fill_color = QtGui.QColor(
            color.red(), color.green(), color.blue(), fill_alpha)

        shape.select_line_color = select_line_color
        shape.select_fill_color = QtGui.QColor(
            color.red(), color.green(), color.blue(), select_fill_alpha)

        # 设置顶点颜色
        shape.vertex_fill_color = color
        # 高亮顶点使用白色
        shape.hvertex_fill_color = QtGui.QColor(255, 255, 255)

    def _update_same_label_colors(self, label, color):
        """更新所有同类同名标签的颜色"""
        # 更新所有已加载的形状
        for shape in self.canvas.shapes:
            if shape.label == label:
                self._apply_shape_color(shape, color)

                # 更新标签列表显示
                try:
                    item = self.labelList.findItemByShape(shape)
                    if item:
                        display_text = shape.label
                        if shape.group_id is not None:
                            display_text = "{} ({})".format(
                                shape.label, shape.group_id)

                        # 使用新颜色更新显示
                        r, g, b = color.getRgb()[:3]
                        colored_text = '{} <font color="#{:02x}{:02x}{:02x}">●</font>'.format(
                            html.escape(display_text), r, g, b
                        )
                        item.setText(colored_text)
                except ValueError:
                    # 形状可能尚未添加到labelList中，忽略这个错误
                    pass

        # 更新标签列表项
        item = self.uniqLabelList.findItemByLabel(label)
        if item:
            r, g, b = color.getRgb()[:3]
            rgb = (r, g, b)
            self.uniqLabelList.setItemLabel(item, label, rgb)

        # 保存到标签颜色映射中，以便后续使用
        if self._config["label_colors"] is None:
            self._config["label_colors"] = {}
        self._config["label_colors"][label] = (
            color.red(), color.green(), color.blue())

    def _get_rgb_by_label(self, label):
        """获取标签的RGB颜色"""
        # 检查label_colors配置
        if (
            self._config["shape_color"] == "manual"
            and self._config["label_colors"]
            and label in self._config["label_colors"]
        ):
            color_hex = self._config["label_colors"][label]
            # 将十六进制转为RGB
            r = int(color_hex[1:3], 16)
            g = int(color_hex[3:5], 16)
            b = int(color_hex[5:7], 16)
            return (r, g, b)

        # 查找已存在的标签项
        item = self.uniqLabelList.findItemByLabel(label)
        if item:
            # 尝试从标签项文本中提取颜色
            text = item.text()
            if "●" in text:
                try:
                    color_str = text.split('color="')[1].split('">')[0]
                    r = int(color_str[1:3], 16)
                    g = int(color_str[3:5], 16)
                    b = int(color_str[5:7], 16)
                    return (r, g, b)
                except (IndexError, ValueError):
                    pass

        # 如果是auto模式，生成唯一颜色
        if self._config["shape_color"] == "auto":
            # 创建新的标签项
            if not item:
                item = self.uniqLabelList.createItemFromLabel(label)
                self.uniqLabelList.addItem(item)

            # 使用黄金比例生成唯一颜色
            hash_value = sum(ord(c) for c in label) % 100
            hue = (hash_value * 0.618033988749895) % 1.0
            r, g, b = [int(x * 255)
                       for x in colorsys.hsv_to_rgb(hue, 0.8, 0.95)]
            return (r, g, b)

        # 使用默认颜色
        elif self._config["default_shape_color"]:
            return self._config["default_shape_color"]

        # 最后的默认值
        return (0, 255, 0)  # 默认绿色

    def fileSearchChanged(self):
        self.importDirImages(
            self.lastOpenDir,
            pattern=self.fileSearch.text(),
            load=False,
        )

    def fileSelectionChanged(self):
        items = self.fileListWidget.selectedItems()
        if not items:
            return
        item = items[0]

        if not self.mayContinue():
            return

        # 获取当前项存储的完整文件路径
        filepath = item.data(Qt.UserRole)
        if filepath and osp.exists(filepath):
            self.loadFile(filepath)

    # React to canvas signals.
    def shapeSelectionChanged(self, selected_shapes):
        # 先切换到编辑模式
        if not self.canvas.editing():
            self.toggleDrawMode(True)  # 切换到编辑模式

        self._noSelectionSlot = True
        for shape in self.canvas.selectedShapes:
            shape.selected = False
        # 清除当前标注列表的选择
        self.labelList.clearSelection()
        # 清除标签列表dock的选择
        self.uniqLabelList.clearSelection()

        self.canvas.selectedShapes = selected_shapes
        for shape in self.canvas.selectedShapes:
            shape.selected = True
            item = self.labelList.findItemByShape(shape)
            if item:  # 确保item不是None
                self.labelList.selectItem(item)
                self.labelList.scrollToItem(item)
        self._noSelectionSlot = False
        n_selected = len(selected_shapes)
        self.actions.delete.setEnabled(n_selected)
        self.actions.duplicate.setEnabled(n_selected)
        self.actions.copy.setEnabled(n_selected)
        self.actions.edit.setEnabled(n_selected)

    def addLabel(self, shape):
        if shape.group_id is None:
            text = shape.label
        else:
            text = "{} ({})".format(shape.label, shape.group_id)

        # 创建标签项
        r, g, b = shape.fill_color.getRgb()[:3]
        colored_text = '{} <font color="#{:02x}{:02x}{:02x}">●</font>'.format(
            html.escape(text), r, g, b
        )

        # 使用带颜色的文本
        label_list_item = LabelTreeWidgetItem(colored_text, shape)
        self.labelList.addItem(label_list_item)

        # 对标签列表窗口的处理
        if self.uniqLabelList.findItemByLabel(shape.label) is None:
            # 添加shape_type参数
            item = self.uniqLabelList.createItemFromLabel(
                shape.label, shape_type=shape.shape_type)
            self.uniqLabelList.addItem(item)
            rgb = self._get_rgb_by_label(shape.label)
            self.uniqLabelList.setItemLabel(item, shape.label, rgb)

        self.labelDialog.addLabelHistory(shape.label)
        for action in self.actions.onShapesPresent:
            action.setEnabled(True)

        self._update_shape_color(shape)

        # 更新分类数量
        self.labelList.updateAllCategoryCounts()

        # 更新dock标题
        self.updateDockTitles()

        # 连接复选框状态变化信号
        self.connectItemCheckState(label_list_item)

        # 更新文件列表中当前文件的复选框状态
        self.updateFileItemCheckState()

        # 更新未使用标签的高亮状态
        self.uniqLabelList.highlightUnusedLabels(self.labelList)

    def connectItemCheckState(self, item):
        """连接标签项的复选框状态变化信号"""
        item.model().itemChanged.connect(self.labelItemCheckStateChanged)

    def labelItemCheckStateChanged(self, item):
        """标签项复选框状态变化处理"""
        shape = item.shape()
        if shape is None:
            return

        # 根据复选框状态设置可见性
        visible = item.checkState() == Qt.Checked
        self.canvas.setShapeVisible(shape, visible)

        # 不再自动选择形状，只更新可见性
        # 更新未使用标签的高亮状态
        self.uniqLabelList.highlightUnusedLabels(self.labelList)

    def _update_shape_color(self, shape):
        r, g, b = self._get_rgb_by_label(shape.label)
        base_color = QtGui.QColor(r, g, b)
        shape.line_color = base_color
        shape.vertex_fill_color = base_color
        shape.hvertex_fill_color = QtGui.QColor(255, 255, 255)

        # 根据形状类型设置不同的填充透明度
        if shape.shape_type == "point":
            # 点标签使用更高的填充透明度
            fill_alpha = 120
            select_fill_alpha = 180
            # 使用白色边框
            select_line_color = QtGui.QColor(255, 255, 255)
        elif shape.shape_type == "rectangle":
            # 矩形标签使用较低的填充透明度
            fill_alpha = 20
            select_fill_alpha = 15
            # 矩形选中使用原色边框，但略微增亮
            select_line_color = base_color.lighter(120)
        else:
            fill_alpha = 30
            select_fill_alpha = 80
            select_line_color = QtGui.QColor(255, 255, 255)

        shape.fill_color = QtGui.QColor(r, g, b, fill_alpha)
        shape.select_line_color = select_line_color
        shape.select_fill_color = QtGui.QColor(r, g, b, select_fill_alpha)

    def save_label_order(self, labels, temporary=False):
        """保存标签的排序顺序

        Args:
            labels (list): 排序后的标签列表
            temporary (bool, optional): 如果为True，则只在内存中更新顺序，不保存到配置文件
        """
        if not self._config:
            self._config = {}

        # 更新标签顺序
        self._config['label_order'] = labels

        # 只有当temporary为False时，才将更新后的配置保存到文件
        if not temporary:
            try:
                from labelme.config import save_config
                save_config(self._config)
            except Exception as e:
                logger.exception("保存标签顺序时出错: %s", e)

        # 更新UI中的标签顺序
        self._update_label_menu_from_config()

    def _update_label_menu_from_config(self):
        """根据配置更新标签菜单顺序"""
        if not self._config or not hasattr(self, 'labelMenu'):
            return

        # 如果存在label_order配置项
        if 'label_order' in self._config:
            # 清空当前标签菜单
            self.labelMenu.clear()

            # 按照保存的顺序重新添加标签
            for label in self._config['label_order']:
                if label:
                    # 检查颜色
                    color = None
                    if self._config['label_colors'] and label in self._config['label_colors']:
                        color = self._config['label_colors'][label]

                    # 创建动作
                    action = QtWidgets.QAction(label, self)
                    if color:
                        rgb = (int(color[1:3], 16), int(
                            color[3:5], 16), int(color[5:7], 16))
                        action.setIcon(labelme.utils.newIcon('circle', rgb))
                    self.labelMenu.addAction(action)

    def remLabels(self, shapes):
        for shape in shapes:
            item = self.labelList.findItemByShape(shape)
            if item:
                self.labelList.removeItem(item)

        # 更新文件列表中当前文件的复选框状态
        self.updateFileItemCheckState()

        # 更新dock标题
        self.updateDockTitles()

        # 更新未使用标签的高亮状态
        self.uniqLabelList.highlightUnusedLabels(self.labelList)

    def loadShapes(self, shapes, replace=True):
        self._noSelectionSlot = True
        for shape in shapes:
            self.addLabel(shape)
        self.labelList.updateAllCategoryCounts()  # 更新所有分类的数量
        self.labelList.clearSelection()

        # 修改这里，如果replace=False，保留原有的形状
        if replace:
            self.canvas.loadShapes(shapes, replace=True)
            self.setClean()
        else:
            # 将新形状添加到已有形状列表中
            existing_shapes = self.canvas.shapes
            existing_shapes.extend(shapes)
            self.canvas.loadShapes(existing_shapes, replace=True)

        self.canvas.setEnabled(True)
        # 确保没有选中任何形状
        self.canvas.deSelectShape()
        self._noSelectionSlot = False
        self.updateDockTitles()

        # 修复所有标签项的颜色显示
        self.fixAllLabelColors()

        # 更新未使用标签的高亮状态
        self.uniqLabelList.highlightUnusedLabels(self.labelList)

    def fixAllLabelColors(self):
        """修复所有标签项的颜色显示"""
        for item in self.labelList:
            shape = item.shape()
            if shape:
                # 获取标签文本（去掉颜色标记部分）
                text = item.text()
                if "<font" in text:
                    text = text.split("<font")[0].strip()
                else:
                    if shape.group_id is None:
                        text = shape.label
                    else:
                        text = "{} ({})".format(shape.label, shape.group_id)

                # 使用形状的实际颜色
                r, g, b = shape.fill_color.getRgb()[:3]
                colored_text = '{} <font color="#{:02x}{:02x}{:02x}">●</font>'.format(
                    html.escape(text), r, g, b
                )
                item.setText(colored_text)

    def loadLabels(self, shapes):
        s = []
        for shape in shapes:
            label = shape["label"]
            points = shape["points"]
            shape_type = shape["shape_type"]
            flags = shape["flags"]
            group_id = shape["group_id"]
            description = shape["description"]
            other_data = shape["other_data"]

            if not points:
                # skip point-empty shape
                continue

            shape = Shape(
                label=label,
                shape_type=shape_type,
                group_id=group_id,
                description=description,
            )
            for x, y in points:
                shape.addPoint(QtCore.QPointF(x, y))
            shape.close()

            default_flags = {}
            if self._config["label_flags"]:
                for pattern, keys in self._config["label_flags"].items():
                    if re.match(pattern, label):
                        for key in keys:
                            default_flags[key] = False
            shape.flags = default_flags
            shape.flags.update(flags)
            shape.other_data = other_data

            s.append(shape)
        self.loadShapes(s)

    def loadFlags(self, flags):
        self.flag_widget.clear()
        for key, flag in flags.items():
            item = QtWidgets.QListWidgetItem(key)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if flag else Qt.Unchecked)
            self.flag_widget.addItem(item)

        # 更新dock标题显示
        self.updateDockTitles()

    def saveLabels(self, filename):
        lf = LabelFile()

        def format_shape(s):
            data = s.other_data.copy()
            data.update(
                dict(
                    label=s.label.encode("utf-8") if PY2 else s.label,
                    points=[(p.x(), p.y()) for p in s.points],
                    group_id=s.group_id,
                    description=s.description,
                    shape_type=s.shape_type,
                    flags=s.flags,
                    # 删除对不存在方法的调用
                )
            )
            return data

        shapes = [format_shape(item.shape()) for item in self.labelList]
        # 获取flags
        flags = {}
        if hasattr(self, 'flag_widget'):
            for i in range(self.flag_widget.count()):
                item = self.flag_widget.item(i)
                key = item.text()
                flag = item.checkState() == Qt.Checked
                flags[key] = flag
        try:
            imagePath = osp.relpath(self.imagePath, osp.dirname(filename))
            imageData = self.imageData if self._config["store_data"] else None
            if osp.dirname(filename) and not osp.exists(osp.dirname(filename)):
                os.makedirs(osp.dirname(filename))
            lf.save(
                filename=filename,
                shapes=shapes,
                imagePath=imagePath,
                imageData=imageData,
                imageHeight=self.image.height(),
                imageWidth=self.image.width(),
                otherData=self.otherData,
                flags=flags,
            )
            self.labelFile = lf
            items = self.fileListWidget.findItems(
                "   " + self.imagePath, Qt.MatchEndsWith
            )
            if len(items) > 0:
                if len(items) != 1:
                    raise RuntimeError("There are duplicate files.")
                items[0].setCheckState(Qt.Checked)

            # 确保文件项的复选框状态与标注列表同步
            self.updateFileItemCheckState()

            # disable allows next and previous image to proceed
            # self.filename = filename
            return True
        except LabelFileError as e:
            self.errorMessage(
                self.tr("Error saving label data"), self.tr("<b>%s</b>") % e
            )
            return False

    def duplicateSelectedShape(self):
        self.copySelectedShape()
        self.pasteSelectedShape()

    def pasteSelectedShape(self):
        self.loadShapes(self._copied_shapes, replace=False)
        self.setDirty()

    def copySelectedShape(self):
        self._copied_shapes = [s.copy() for s in self.canvas.selectedShapes]
        self.actions.paste.setEnabled(len(self._copied_shapes) > 0)

    def labelSelectionChanged(self):
        if self._noSelectionSlot:
            return

        # 先切换到编辑模式
        if not self.canvas.editing():
            self.toggleDrawMode(True)  # 切换到编辑模式

        # 清除标签列表dock的选择，避免混淆
        self.uniqLabelList.clearSelection()

        if self.canvas.editing():
            selected_shapes = []
            for item in self.labelList.selectedItems():
                selected_shapes.append(item.shape())
            if selected_shapes:
                self.canvas.selectShapes(selected_shapes)
            else:
                self.canvas.deSelectShape()

    def labelItemChanged(self, item):
        shape = item.shape()
        self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)

    def labelOrderChanged(self):
        self.setDirty()
        self.canvas.loadShapes([item.shape() for item in self.labelList])

    # Callback functions:

    def newShape(self):
        """调用此方法创建一个新形状"""
        # 使用上一次的标签名，如果没有则使用默认提示
        text = self._previous_label_text if self._previous_label_text else self.tr(
            "请输入对象标签")
        flags = {}
        group_id = None
        description = ""

        # 生成随机颜色作为默认颜色
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        default_color = QtGui.QColor(r, g, b)

        # 获取鼠标当前位置
        # 将鼠标在屏幕上的位置与窗口位置相关联，以便更准确地定位标签对话框
        mouse_pos = QtGui.QCursor.pos()  # 获取全局位置

        # 保存当前位置，每次添加新形状后重新弹出对话框，对话框位置应该一致
        saved_mouse_pos = mouse_pos

        while True:
            result = self.labelDialog.popUp(
                text=text,
                flags=flags,
                group_id=group_id,
                description=description,
                color=default_color,
                mouse_pos=saved_mouse_pos,  # 使用保存的鼠标位置
            )
            if result is None:
                # 用户取消了标签对话框，删除当前绘制的形状
                if len(self.canvas.shapes) > 0:
                    # 删除最后一个形状（即刚刚创建的）
                    last_shape = self.canvas.shapes.pop()
                    self.canvas.storeShapes()
                    self.canvas.update()
                return

            text, flags, group_id, description, color = result

            if text is not None and self.validateLabel(text):
                # 保存当前使用的标签名称
                self._previous_label_text = text

                # 检查是否已存在同名标签
                item = self.uniqLabelList.findItemByLabel(text)
                if item and color == default_color:  # 用户没有修改颜色，使用已有标签的颜色
                    # 从现有标签项中获取颜色
                    try:
                        item_text = item.text()
                        if "●" in item_text:
                            color_str = item_text.split(
                                'color="')[1].split('">')[0]
                            r = int(color_str[1:3], 16)
                            g = int(color_str[3:5], 16)
                            b = int(color_str[5:7], 16)
                            color = QtGui.QColor(r, g, b)
                    except (IndexError, ValueError):
                        # 如果无法从文本中提取颜色，尝试从配置或默认值获取
                        rgb = self._get_rgb_by_label(text)
                        if rgb:
                            color = QtGui.QColor(*rgb)
                break

            # 标签未通过验证，显示错误消息
            self.errorMessage(
                self.tr("Invalid label"),
                self.tr("Invalid label '{}' with validation type '{}'").format(
                    text, self._config["validate_label"]
                ),
            )

        shape = self.canvas.setLastLabel(text, flags, group_id, description)

        # 如果是新标签或者用户选择了自定义颜色，应用颜色
        self._apply_shape_color(shape, color)

        # 添加到形状列表
        self.addLabel(shape)

        # 如果是已有标签，确保所有同名标签颜色一致
        if self.uniqLabelList.findItemByLabel(text):
            self._update_same_label_colors(text, color)

        # 如果是第一个形状，启用相关动作
        if len(self.canvas.shapes) == 1:
            self.actions.delete.setEnabled(True)

        # 设置为修改状态
        self.setDirty()

    def scrollRequest(self, delta, orientation):
        units = -delta * 0.1  # natural scroll
        bar = self.scrollBars[orientation]
        value = bar.value() + bar.singleStep() * units
        self.setScroll(orientation, value)

    def setScroll(self, orientation, value):
        self.scrollBars[orientation].setValue(int(value))
        self.scroll_values[orientation][self.filename] = value

    def setZoom(self, value):
        self.actions.fitWidth.setChecked(False)
        self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.MANUAL_ZOOM
        self.zoomWidget.setValue(value)
        self.zoom_values[self.filename] = (self.zoomMode, value)

    def addZoom(self, increment=1.1):
        zoom_value = self.zoomWidget.value() * increment
        if increment > 1:
            zoom_value = math.ceil(zoom_value)
        else:
            zoom_value = math.floor(zoom_value)
        self.setZoom(zoom_value)

    def zoomRequest(self, delta, pos):
        canvas_width_old = self.canvas.width()
        units = 1.1
        if delta < 0:
            units = 0.9
        self.addZoom(units)

        canvas_width_new = self.canvas.width()
        if canvas_width_old != canvas_width_new:
            canvas_scale_factor = canvas_width_new / canvas_width_old

            x_shift = round(pos.x() * canvas_scale_factor) - pos.x()
            y_shift = round(pos.y() * canvas_scale_factor) - pos.y()

            self.setScroll(
                Qt.Horizontal,
                self.scrollBars[Qt.Horizontal].value() + x_shift,
            )
            self.setScroll(
                Qt.Vertical,
                self.scrollBars[Qt.Vertical].value() + y_shift,
            )

    def setFitWindow(self, value=True):
        if value:
            self.actions.fitWidth.setChecked(False)
        self.zoomMode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self.adjustScale()

    def setFitWidth(self, value=True):
        if value:
            self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        self.adjustScale()

    def enableKeepPrevScale(self, enabled):
        self._config["keep_prev_scale"] = enabled
        self.actions.keepPrevScale.setChecked(enabled)

    def onNewBrightnessContrast(self, qimage):
        self.canvas.loadPixmap(
            QtGui.QPixmap.fromImage(qimage), clear_shapes=False)

    def brightnessContrast(self, value):
        dialog = BrightnessContrastDialog(
            utils.img_data_to_pil(self.imageData),
            self.onNewBrightnessContrast,
            parent=self,
        )
        brightness, contrast = self.brightnessContrast_values.get(
            self.filename, (None, None)
        )
        if brightness is not None:
            dialog.slider_brightness.setValue(brightness)
        if contrast is not None:
            dialog.slider_contrast.setValue(contrast)
        dialog.exec_()

        brightness = dialog.slider_brightness.value()
        contrast = dialog.slider_contrast.value()
        self.brightnessContrast_values[self.filename] = (brightness, contrast)

    def togglePolygons(self, value):
        flag = value
        for item in self.labelList:
            if value is None:
                flag = item.checkState() == Qt.Unchecked
            item.setCheckState(Qt.Checked if flag else Qt.Unchecked)

    def loadFile(self, filename=None):
        """Load the specified file, or the last opened file if None."""
        # changing fileListWidget loads file
        if filename in self.imageList and (
            self.fileListWidget.currentRow() != self.imageList.index(filename)
        ):
            self.fileListWidget.setCurrentRow(self.imageList.index(filename))
            self.fileListWidget.repaint()
            return

        self.resetState()
        self.canvas.setEnabled(False)
        if filename is None:
            filename = self.settings.value("filename", "")
        filename = str(filename)

        # 记录当前图片索引
        if filename in self.imageList:
            current_index = self.imageList.index(filename)
            self._config["last_image_index"] = current_index

        if not QtCore.QFile.exists(filename):
            self.errorMessage(
                self.tr("Error opening file"),
                self.tr("No such file: <b>%s</b>") % filename,
            )
            return False

        # assumes same name, but json extension
        self.status(
            str(self.tr("Loading %s...")) % osp.basename(str(filename))
        )
        label_file = osp.splitext(filename)[0] + ".json"
        if self.output_dir:
            label_file_without_path = osp.basename(label_file)
            label_file = osp.join(self.output_dir, label_file_without_path)
        if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(
            label_file
        ):
            try:
                self.labelFile = LabelFile(label_file)
            except LabelFileError as e:
                self.errorMessage(
                    self.tr("Error opening file"),
                    self.tr(
                        "<p><b>%s</b></p>"
                        "<p>Make sure <i>%s</i> is a valid label file."
                    )
                    % (e, label_file),
                )
                self.status(self.tr("Error reading %s") % label_file)
                return False
            self.imageData = self.labelFile.imageData
            self.imagePath = osp.join(
                osp.dirname(label_file),
                self.labelFile.imagePath,
            )
            if len(self.labelFile.shapes) == 0:
                # 如果存在标签文件但没有标注，则确保文件列表中的复选框状态为未选中
                self.updateFileItemCheckState()

            self.otherData = self.labelFile.otherData
        else:
            self.imageData = LabelFile.load_image_file(filename)
            if self.imageData:
                self.imagePath = filename
            self.labelFile = None

            # 确保文件列表中的复选框状态为未选中
            self.updateFileItemCheckState()

        image = QtGui.QImage.fromData(self.imageData)

        if image.isNull():
            formats = [
                "*.{}".format(fmt.data().decode())
                for fmt in QtGui.QImageReader.supportedImageFormats()
            ]
            self.errorMessage(
                self.tr("Error opening file"),
                self.tr(
                    "<p>Make sure <i>{0}</i> is a valid image file.<br/>"
                    "Supported image formats: {1}</p>"
                ).format(filename, ",".join(formats)),
            )
            self.status(self.tr("Error reading %s") % filename)
            return False
        self.image = image
        self.filename = filename
        if self._config["keep_prev"]:
            prev_shapes = self.canvas.shapes
        self.canvas.loadPixmap(QtGui.QPixmap.fromImage(image))
        flags = {k: False for k in self._config["flags"] or []}
        if self.labelFile:
            self.loadLabels(self.labelFile.shapes)
            if self.labelFile.flags is not None:
                flags.update(self.labelFile.flags)
        self.loadFlags(flags)
        if self._config["keep_prev"] and self.noShapes():
            self.loadShapes(prev_shapes, replace=False)
            self.setDirty()
        else:
            self.setClean()
        self.canvas.setEnabled(True)
        # set zoom values
        is_initial_load = not self.zoom_values
        if self.filename in self.zoom_values:
            self.zoomMode = self.zoom_values[self.filename][0]
            self.setZoom(self.zoom_values[self.filename][1])
        elif is_initial_load or not self._config["keep_prev_scale"]:
            self.adjustScale(initial=True)
        # set scroll values
        for orientation in self.scroll_values:
            if self.filename in self.scroll_values[orientation]:
                self.setScroll(
                    orientation, self.scroll_values[orientation][self.filename]
                )
        # set brightness contrast values
        dialog = BrightnessContrastDialog(
            utils.img_data_to_pil(self.imageData),
            self.onNewBrightnessContrast,
            parent=self,
        )
        brightness, contrast = self.brightnessContrast_values.get(
            self.filename, (None, None)
        )
        if self._config["keep_prev_brightness"] and self.recentFiles:
            brightness, _ = self.brightnessContrast_values.get(
                self.recentFiles[0], (None, None)
            )
        if self._config["keep_prev_contrast"] and self.recentFiles:
            _, contrast = self.brightnessContrast_values.get(
                self.recentFiles[0], (None, None)
            )
        if brightness is not None:
            dialog.slider_brightness.setValue(brightness)
        if contrast is not None:
            dialog.slider_contrast.setValue(contrast)
        self.brightnessContrast_values[self.filename] = (brightness, contrast)

        if brightness is not None or contrast is not None:
            dialog.onValueChanged()

        self.paintCanvas()
        self.addRecentFile(self.filename)
        self.toggleActions(True)
        self.canvas.setFocus()

        # 保存图像路径到配置
        self.settings.setValue("filename", filename)

        # 保存当前文件目录
        self.settings.setValue("last_open_dir", osp.dirname(filename))

        self.status(str(self.tr("Loaded %s")) % osp.basename(str(filename)))
        # 更新dock标题
        self.updateDockTitles()
        # 设置适应窗口
        self.setFitWindow(True)
        self.actions.fitWindow.setChecked(True)
        self.adjustScale(initial=True)

        # 更新未使用标签的高亮状态
        self.uniqLabelList.highlightUnusedLabels(self.labelList)

        return True

    def resizeEvent(self, event):
        if self.canvas and hasattr(self, 'image') and self.image is not None and not self.image.isNull():
            self.adjustScale()
        super(MainWindow, self).resizeEvent(event)

    def paintCanvas(self):
        assert not self.image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.adjustSize()
        self.canvas.update()

    def adjustScale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoomMode]()
        value = int(100 * value)
        self.zoomWidget.setValue(value)
        self.zoom_values[self.filename] = (self.zoomMode, value)

    def scaleFitWindow(self):
        """Figure out the size of the pixmap to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.centralWidget().width() - e
        h1 = self.centralWidget().height() - e
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def scaleFitWidth(self):
        # The epsilon does not seem to work too well here.
        w = self.centralWidget().width() - 2.0
        return w / self.canvas.pixmap.width()

    def enableSaveImageWithData(self, enabled):
        self._config["store_data"] = enabled
        self.actions.saveWithImageData.setChecked(enabled)

    def closeEvent(self, event):
        # 保存当前目录到配置
        if hasattr(self, "currentPath") and self.currentPath():
            self._config["last_dir"] = self.currentPath()
        if hasattr(self, "output_dir") and self.output_dir:
            self._config["output_dir"] = self.output_dir
            logger.info("Saving output directory: {}".format(self.output_dir))

        # 保存当前图片索引
        if hasattr(self, "filename") and self.filename and hasattr(self, "imageList") and self.filename in self.imageList:
            current_index = self.imageList.index(self.filename)
            self._config["last_image_index"] = current_index
            logger.info("Saving current image index: {}".format(current_index))

        # 保存当前主题设置
        if hasattr(self, "currentTheme"):
            self._config["theme"] = self.currentTheme
            logger.info("Saving theme setting: {}".format(self.currentTheme))

        if not self.mayContinue():
            event.ignore()
            return

        # 保存窗口状态
        self.settings.setValue(
            "filename", self.filename if self.filename else "")
        self.settings.setValue("window/size", self.size())
        self.settings.setValue("window/position", self.pos())
        self.settings.setValue("window/state", self.saveState())
        self.settings.setValue("recentFiles", self.recentFiles)

        # 保存配置到文件
        from labelme.config import save_config
        save_config(self._config)

        event.accept()

    def dragEnterEvent(self, event):
        extensions = [
            ".%s" % fmt.data().decode().lower()
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]
        if event.mimeData().hasUrls():
            items = [i.toLocalFile() for i in event.mimeData().urls()]
            if any([i.lower().endswith(tuple(extensions)) for i in items]):
                event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if not self.mayContinue():
            event.ignore()
            return
        items = [i.toLocalFile() for i in event.mimeData().urls()]
        self.importDroppedImageFiles(items)

    # User Dialogs #

    def loadRecent(self, filename):
        if self.mayContinue():
            self.loadFile(filename)

    def openPrevImg(self, _value=False):
        keep_prev = self._config["keep_prev"]
        if QtWidgets.QApplication.keyboardModifiers() == (
            Qt.ControlModifier | Qt.ShiftModifier
        ):
            self._config["keep_prev"] = True

        if not self.mayContinue():
            return

        if len(self.imageList) <= 0:
            return

        if self.filename is None:
            return

        currIndex = self.imageList.index(self.filename)
        if currIndex - 1 >= 0:
            filename = self.imageList[currIndex - 1]
            if filename:
                self.loadFile(filename)

        self._config["keep_prev"] = keep_prev

    def openNextImg(self, _value=False, load=True):
        keep_prev = self._config["keep_prev"]
        if QtWidgets.QApplication.keyboardModifiers() == (
            Qt.ControlModifier | Qt.ShiftModifier
        ):
            self._config["keep_prev"] = True

        if not self.mayContinue():
            return

        if len(self.imageList) <= 0:
            return

        filename = None
        if self.filename is None:
            filename = self.imageList[0]
        else:
            currIndex = self.imageList.index(self.filename)
            if currIndex + 1 < len(self.imageList):
                filename = self.imageList[currIndex + 1]
            else:
                filename = self.imageList[-1]
        self.filename = filename

        if self.filename and load:
            self.loadFile(self.filename)

        self._config["keep_prev"] = keep_prev

    def openFile(self, _value=False):
        if not self.mayContinue():
            return
        path = osp.dirname(str(self.filename)) if self.filename else "."
        formats = [
            "*.{}".format(fmt.data().decode())
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]
        filters = self.tr("Image & Label files (%s)") % " ".join(
            formats + ["*%s" % LabelFile.suffix]
        )
        fileDialog = FileDialogPreview(self)
        fileDialog.setFileMode(FileDialogPreview.ExistingFile)
        fileDialog.setNameFilter(filters)
        fileDialog.setWindowTitle(
            self.tr("%s - Choose Image or Label file") % __appname__,
        )
        fileDialog.setWindowFilePath(path)
        fileDialog.setViewMode(FileDialogPreview.Detail)
        if fileDialog.exec_():
            fileName = fileDialog.selectedFiles()[0]
            if fileName:
                self.loadFile(fileName)

    def changeOutputDirDialog(self, _value=False):
        default_output_dir = self.output_dir
        if default_output_dir is None and self.filename:
            default_output_dir = osp.dirname(self.filename)
        if default_output_dir is None:
            default_output_dir = self.currentPath()

        dirpath = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("%s - 选择输出目录") % __appname__,
            default_output_dir or "",
            QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.DontResolveSymlinks,
        )
        if not dirpath:
            return

        self.output_dir = dirpath
        self._config["output_dir"] = dirpath
        self.statusBar().showMessage(
            self.tr("输出目录已更改为: %s") % self.output_dir, 5000
        )
        logger.info("Output directory changed to: {}".format(self.output_dir))

        # 保存当前文件名，以便在重新加载后恢复选择
        current_filename = self.filename

        # 重新加载当前目录的图像，以更新标注状态
        if self.lastOpenDir and osp.exists(self.lastOpenDir):
            self.importDirImages(self.lastOpenDir, load=False)

            # 如果当前有选中的文件，保持选中状态
            if current_filename and current_filename in self.imageList:
                self.fileListWidget.setCurrentRow(
                    self.imageList.index(current_filename))
                self.fileListWidget.repaint()
                # 重新加载当前文件
                self.loadFile(current_filename)

    def saveFile(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        if self.labelFile:
            # DL20180323 - overwrite when in directory
            self._saveFile(self.labelFile.filename)
        elif self.output_file:
            self._saveFile(self.output_file)
            self.close()
        else:
            self._saveFile(self.saveFileDialog())

    def saveFileAs(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        self._saveFile(self.saveFileDialog())

    def saveFileDialog(self):
        caption = self.tr("%s - Choose File") % __appname__
        filters = self.tr("Label files (*%s)") % LabelFile.suffix
        if self.output_dir:
            dlg = QtWidgets.QFileDialog(
                self, caption, self.output_dir, filters)
        else:
            dlg = QtWidgets.QFileDialog(
                self, caption, self.currentPath(), filters)
        dlg.setDefaultSuffix(LabelFile.suffix[1:])
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dlg.setOption(QtWidgets.QFileDialog.DontConfirmOverwrite, False)
        dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, False)
        basename = osp.basename(osp.splitext(self.filename)[0])
        if self.output_dir:
            default_labelfile_name = osp.join(
                self.output_dir, basename + LabelFile.suffix
            )
        else:
            default_labelfile_name = osp.join(
                self.currentPath(), basename + LabelFile.suffix
            )
        filename = dlg.getSaveFileName(
            self,
            self.tr("Choose File"),
            default_labelfile_name,
            self.tr("Label files (*%s)") % LabelFile.suffix,
        )
        if isinstance(filename, tuple):
            filename, _ = filename
        return filename

    def _saveFile(self, filename):
        if filename and self.saveLabels(filename):
            self.addRecentFile(filename)
            self.setClean()

    def closeFile(self, _value=False):
        if not self.mayContinue():
            return
        self.resetState()
        self.setClean()
        self.toggleActions(False)
        self.canvas.setEnabled(False)
        self.actions.saveAs.setEnabled(False)

    def getLabelFile(self):
        if self.filename.lower().endswith(".json"):
            label_file = self.filename
        else:
            label_file = osp.splitext(self.filename)[0] + ".json"

        return label_file

    def deleteFile(self):
        mb = QtWidgets.QMessageBox
        msg = self.tr(
            "You are about to permanently delete this label file, " "proceed anyway?"
        )
        answer = mb.warning(self, self.tr("Attention"), msg, mb.Yes | mb.No)
        if answer != mb.Yes:
            return

        label_file = self.getLabelFile()
        if osp.exists(label_file):
            os.remove(label_file)
            logger.info("Label file is removed: {}".format(label_file))

            item = self.fileListWidget.currentItem()
            item.setCheckState(Qt.Unchecked)

            self.resetState()

    # Message Dialogs. #
    def hasLabels(self):
        if self.noShapes():
            self.errorMessage(
                "No objects labeled",
                "You must label at least one object to save the file.",
            )
            return False
        return True

    def hasLabelFile(self):
        if self.filename is None:
            return False

        label_file = self.getLabelFile()
        return osp.exists(label_file)

    def mayContinue(self):
        if not self.dirty:
            return True
        mb = QtWidgets.QMessageBox
        msg = self.tr('Save annotations to "{}" before closing?').format(
            self.filename)
        answer = mb.question(
            self,
            self.tr("Save annotations?"),
            msg,
            mb.Save | mb.Discard | mb.Cancel,
            mb.Save,
        )
        if answer == mb.Discard:
            return True
        elif answer == mb.Save:
            self.saveFile()
            return True
        else:  # answer == mb.Cancel
            return False

    def errorMessage(self, title, message):
        return QtWidgets.QMessageBox.critical(
            self, title, "<p><b>%s</b></p>%s" % (title, message)
        )

    def currentPath(self):
        return osp.dirname(str(self.filename)) if self.filename else "."

    def toggleKeepPrevMode(self):
        self._config["keep_prev"] = not self._config["keep_prev"]

    def removeSelectedPoint(self):
        self.canvas.removeSelectedPoint()
        self.canvas.update()
        if not self.canvas.hShape.points:
            self.canvas.deleteShape(self.canvas.hShape)
            self.remLabels([self.canvas.hShape])
            if self.noShapes():
                for action in self.actions.onShapesPresent:
                    action.setEnabled(False)
        self.setDirty()

    def deleteSelectedShape(self):
        if not self.canvas.selectedShapes:
            return

        yes, no = QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No
        msg = self.tr(
            "您即将永久删除 {} 个标注对象，确定继续吗？"
        ).format(len(self.canvas.selectedShapes))
        if yes == QtWidgets.QMessageBox.warning(
            self, self.tr("注意"), msg, yes | no, yes
        ):
            deleted_shapes = self.canvas.deleteSelected()
            self.remLabels(deleted_shapes)
            self.setDirty()
            if self.noShapes():
                for action in self.actions.onShapesPresent:
                    action.setEnabled(False)
            # 更新dock标题
            self.updateDockTitles()

    def copyShape(self):
        self.canvas.endMove(copy=True)
        for shape in self.canvas.selectedShapes:
            self.addLabel(shape)
        self.labelList.clearSelection()
        self.setDirty()

    def moveShape(self):
        self.canvas.endMove(copy=False)
        self.setDirty()

    def openDirDialog(self, _value=False, dirpath=None):
        if not self.mayContinue():
            return

        defaultOpenDirPath = dirpath if dirpath else "."
        if self.lastOpenDir and osp.exists(self.lastOpenDir):
            defaultOpenDirPath = self.lastOpenDir
        else:
            defaultOpenDirPath = osp.dirname(
                self.filename) if self.filename else "."

        targetDirPath = str(
            QtWidgets.QFileDialog.getExistingDirectory(
                self,
                self.tr("%s - Open Directory") % __appname__,
                defaultOpenDirPath,
                QtWidgets.QFileDialog.ShowDirsOnly
                | QtWidgets.QFileDialog.DontResolveSymlinks,
            )
        )
        self.importDirImages(targetDirPath)

    @property
    def imageList(self):
        lst = []
        for i in range(self.fileListWidget.count()):
            item = self.fileListWidget.item(i)
            # 获取存储在Qt.UserRole中的完整路径
            filepath = item.data(Qt.UserRole)
            lst.append(filepath)
        return lst

    def importDroppedImageFiles(self, imageFiles):
        extensions = [
            ".%s" % fmt.data().decode().lower()
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]

        self.filename = None
        for file in imageFiles:
            if file in self.imageList or not file.lower().endswith(tuple(extensions)):
                continue
            label_file = osp.splitext(file)[0] + ".json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)

            # 创建一个QListWidgetItem，但只显示文件名，不显示路径
            basename = osp.basename(file)
            item = QtWidgets.QListWidgetItem()
            # 设置文本格式，在文本前添加空格以增加与复选框的距离
            item.setText("   " + basename)
            # 存储完整路径作为项的数据，用于后续加载文件
            item.setData(Qt.UserRole, file)

            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(label_file):
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)
            self.fileListWidget.addItem(item)

        if len(self.imageList) > 1:
            self.actions.openNextImg.setEnabled(True)
            self.actions.openPrevImg.setEnabled(True)

        self.openNextImg()

    def importDirImages(self, dirpath, pattern=None, load=True):
        self.actions.openNextImg.setEnabled(True)
        self.actions.openPrevImg.setEnabled(True)

        if not self.mayContinue() or not dirpath:
            return

        self.lastOpenDir = dirpath
        self.filename = None
        self.fileListWidget.clear()

        filenames = self.scanAllImages(dirpath)
        if pattern:
            try:
                filenames = [f for f in filenames if re.search(pattern, f)]
            except re.error:
                pass
        for filename in filenames:
            label_file = osp.splitext(filename)[0] + ".json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)

            # 创建一个QListWidgetItem，但只显示文件名，不显示路径
            basename = osp.basename(filename)
            item = QtWidgets.QListWidgetItem()
            # 设置文本格式，在文本前添加空格以增加与复选框的距离
            item.setText("   " + basename)
            # 存储完整路径作为项的数据，用于后续加载文件
            item.setData(Qt.UserRole, filename)

            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(label_file):
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)
            self.fileListWidget.addItem(item)
        # 更新dock标题
        self.updateDockTitles()

        # 如果有上次记录的图片索引，并且是有效的索引，则优先加载该图片
        if load and self._config.get("last_image_index") is not None:
            last_idx = self._config["last_image_index"]
            if last_idx >= 0 and last_idx < len(self.imageList):
                self.fileListWidget.setCurrentRow(last_idx)
                self.fileListWidget.repaint()
                self.loadFile(self.imageList[last_idx])
                return

        # 否则加载下一张图片
        self.openNextImg(load=load)

    def scanAllImages(self, folderPath):
        extensions = [
            ".%s" % fmt.data().decode().lower()
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]

        images = []
        for root, dirs, files in os.walk(folderPath):
            for file in files:
                if file.lower().endswith(tuple(extensions)):
                    relativePath = os.path.normpath(osp.join(root, file))
                    images.append(relativePath)
        images = natsort.os_sorted(images)
        return images

    def openAISettings(self):
        """打开AI设置对话框"""
        dialog = AISettingsDialog(self)

        # 如果对话框被接受，更新配置
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            # 重新加载配置
            self._config = get_config()

            # 如果有AI模型选择框，更新它
            if hasattr(self, "_selectAiModelComboBox"):
                # 获取当前选择的AI模型
                ai_config = self._config.get("ai", {})
                default_model = ai_config.get(
                    "default", "EfficientSam (speed)")

                # 查找并设置模型
                for i in range(self._selectAiModelComboBox.count()):
                    if self._selectAiModelComboBox.itemText(i) == default_model:
                        self._selectAiModelComboBox.setCurrentIndex(i)
                        break

    def runObjectDetection(self):
        """运行目标检测"""
        if self.image is None:
            self.errorMessage(
                self.tr("错误"),
                self.tr("请先加载图像"),
            )
            return

        try:
            # 开始显示进度条
            self.startProgress(self.tr("正在执行目标检测..."))

            # 获取已存在的矩形框数量
            existing_rectangle_count = 0
            if self.canvas.shapes:
                for shape in self.canvas.shapes:
                    if shape.shape_type == "rectangle":
                        existing_rectangle_count += 1

            self.setProgress(10)  # 更新进度

            # 将QImage转换为numpy数组
            image = self.image.convertToFormat(QtGui.QImage.Format_RGB888)
            width = image.width()
            height = image.height()
            ptr = image.bits()
            ptr.setsize(height * width * 3)
            img_array = np.frombuffer(
                ptr, np.uint8).reshape((height, width, 3))

            self.setProgress(20)  # 更新进度

            # 加载配置
            config_loader = ConfigLoader()
            detection_config = config_loader.get_detection_config()

            # 从配置中获取参数
            model_name = detection_config.get("model_name")
            conf_threshold = detection_config.get("conf_threshold")
            device = detection_config.get("device")
            filter_classes = detection_config.get("filter_classes")
            nms_threshold = detection_config.get("nms_threshold")
            max_detections = detection_config.get("max_detections")
            use_gpu_if_available = detection_config.get("use_gpu_if_available")
            advanced_params = detection_config.get("advanced")

            self.setProgress(30)  # 更新进度

            logger.info(
                f"使用模型: {model_name}, 置信度阈值: {conf_threshold}, NMS阈值: {nms_threshold}")
            # 运行目标检测
            self.setProgress(40)  # 更新进度 - 开始模型推理
            shape_dicts = object_detection.detect_objects(
                img_array,
                model_name=model_name,
                conf_threshold=conf_threshold,
                device=device,
                filter_classes=filter_classes,
                nms_threshold=nms_threshold,
                max_detections=max_detections,
                use_gpu_if_available=use_gpu_if_available,
                advanced_params=advanced_params,
                start_group_id=existing_rectangle_count  # 传递起始group_id
            )

            self.setProgress(80)  # 更新进度 - 模型推理完成

            if not shape_dicts:
                self.endProgress(self.tr("未检测到任何对象"))
                self.errorMessage(
                    self.tr("提示"),
                    self.tr("未检测到任何对象"),
                )
                return

            # 将字典列表转换为Shape对象列表
            shapes = []
            for shape_dict in shape_dicts:
                shape = Shape(
                    label=shape_dict["label"],
                    shape_type=shape_dict["shape_type"],
                    group_id=shape_dict.get("group_id"),
                    flags=shape_dict.get("flags", {}),
                    description=shape_dict.get("description", ""),
                )
                for point in shape_dict["points"]:
                    shape.addPoint(QtCore.QPointF(point[0], point[1]))
                shapes.append(shape)

            self.setProgress(90)  # 更新进度 - 开始加载形状

            # 加载检测结果，使用replace=False保留原有形状
            self.loadShapes(shapes, replace=False)
            self.setDirty()

            # 完成并显示结果消息
            result_message = self.tr(f"检测到 {len(shapes)} 个对象")
            self.endProgress(result_message)

        except Exception as e:
            self.endProgress(self.tr("检测失败"))
            self.errorMessage(
                self.tr("目标检测错误"),
                self.tr(f"运行目标检测时出错: {str(e)}"),
            )
            logger.exception("目标检测错误")

    def runPoseEstimation(self):
        """运行人体姿态估计"""
        if self.image is None:
            self.errorMessage(
                self.tr("错误"),
                self.tr("请先加载图像"),
            )
            return

        try:
            # 开始显示进度条
            self.startProgress(self.tr("正在执行人体姿态估计..."))

            # 将QImage转换为numpy数组
            image = self.image.convertToFormat(QtGui.QImage.Format_RGB888)
            width = image.width()
            height = image.height()
            ptr = image.bits()
            ptr.setsize(height * width * 3)
            img_array = np.frombuffer(
                ptr, np.uint8).reshape((height, width, 3))

            self.setProgress(20)  # 更新进度

            # 获取现有的person边界框
            existing_person_boxes = []
            existing_person_boxes_ids = []
            if self.canvas.shapes:
                for shape in self.canvas.shapes:
                    if shape.label.lower() == "person" and shape.shape_type == "rectangle":
                        # 只处理矩形框并且标签为person的形状
                        points = shape.points
                        if len(points) >= 2:  # 矩形应该有两个点 (左上和右下)
                            x1 = min(points[0].x(), points[1].x())
                            y1 = min(points[0].y(), points[1].y())
                            x2 = max(points[0].x(), points[1].x())
                            y2 = max(points[0].y(), points[1].y())
                            existing_person_boxes.append([x1, y1, x2, y2])
                            # 记录框的group_id，用于关联姿态关键点
                            existing_person_boxes_ids.append(shape.group_id)

            self.setProgress(30)  # 更新进度

            # 加载配置
            config_loader = ConfigLoader()
            pose_config = config_loader.get_pose_estimation_config()

            # 获取是否使用已有目标检测结果的设置
            use_detection_results = pose_config.get(
                "use_detection_results", True)

            # 获取是否绘制骨骼的设置
            draw_skeleton = pose_config.get("draw_skeleton", True)

            # 记录日志
            if existing_person_boxes and use_detection_results:
                logger.info(f"找到 {len(existing_person_boxes)} 个已有的person框")
            else:
                logger.info("未找到已有的person框或未启用使用已有框")

            self.setProgress(40)  # 更新进度 - 开始模型推理

            # 运行人体姿态估计，传递已有的person框和group_id
            shape_dicts = pose_estimation.estimate_poses(
                img_array,
                existing_person_boxes=existing_person_boxes,
                existing_person_boxes_ids=existing_person_boxes_ids,
                use_detection_results=use_detection_results,
                draw_skeleton=draw_skeleton
            )

            self.setProgress(80)  # 更新进度 - 模型推理完成

            if not shape_dicts:
                self.endProgress(self.tr("未检测到任何人体姿态"))
                self.errorMessage(
                    self.tr("提示"),
                    self.tr("未检测到任何人体姿态"),
                )
                return

            # 将字典列表转换为Shape对象列表
            shapes = []
            for shape_dict in shape_dicts:
                shape = Shape(
                    label=shape_dict["label"],
                    shape_type=shape_dict["shape_type"],
                    group_id=shape_dict.get("group_id"),
                    flags=shape_dict.get("flags", {}),
                    description=shape_dict.get("description", ""),
                )
                for point in shape_dict["points"]:
                    shape.addPoint(QtCore.QPointF(point[0], point[1]))
                shapes.append(shape)

            self.setProgress(90)  # 更新进度 - 开始加载形状

            # 加载检测结果，使用replace=False保留原有形状
            self.loadShapes(shapes, replace=False)
            self.setDirty()

            # 完成并显示结果消息
            result_message = self.tr(f"检测到 {len(shapes)} 个人体姿态关键点")
            self.endProgress(result_message)

        except Exception as e:
            self.endProgress(self.tr("姿态估计失败"))
            self.errorMessage(
                self.tr("人体姿态估计错误"),
                self.tr(f"运行人体姿态估计时出错: {str(e)}"),
            )
            logger.exception("人体姿态估计错误")

    def toggleAutoSave(self, enabled):
        """启用或禁用自动保存功能"""
        self._config["auto_save"] = enabled
        # 更新菜单项的勾选状态
        self.actions.saveAuto.setChecked(enabled)

    def updateDockTitles(self):
        # 获取各个面板中的项目数量
        file_count = len(self.fileListWidget) if self.fileListWidget else 0
        label_count = self.uniqLabelList.count(
        ) if self.uniqLabelList else 0  # 修改为使用uniqLabelList的count
        # 已经使用正确的labelList计数
        shape_count = len(self.labelList) if self.labelList else 0
        # 获取标记面板的计数
        flag_count = self.flag_widget.count() if self.flag_widget else 0

        # 判断当前主题
        is_dark_theme = self.currentTheme == "dark"

        # 根据当前主题设置徽章样式和标题栏容器样式
        if is_dark_theme:
            # 暗色主题徽章样式 - 使用蓝色背景和白色文字
            badge_style = """
                QLabel { 
                    background-color: #2d5f9e; 
                    color: #ffffff; 
                    border-radius: 8px; 
                    min-width: 26px; 
                    max-width: 60px;
                    height: 14px; 
                    font-size: 25px; 
                    font-weight: 700; 
                    margin-left: 10px;
                    padding: 0px 8px;
                    text-align: center;
                    font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', 'Arial', sans-serif;
                    border: none;
                }
            """
            # 暗色主题标题容器样式 - 深色背景和亮色文字
            title_container_style = """
                QWidget {
                    background-color: #2d2d30;
                    padding: 3px;
                    border-bottom: 1px solid #3e3e42;
                }
                QLabel {
                    color: #e0e0e0;
                    font-weight: 500;
                }
            """
        else:
            # 亮色主题徽章样式 - 使用浅蓝色背景和深蓝色文字
            badge_style = """
                QLabel { 
                    background-color: #dfeaf7; 
                    color: #2d81f7; 
                    border-radius: 8px; 
                    min-width: 26px; 
                    max-width: 60px;
                    height: 14px; 
                    font-size: 25px; 
                    font-weight: 700; 
                    margin-left: 10px;
                    padding: 0px 8px;
                    text-align: center;
                    font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', 'Arial', sans-serif;
                    border: none;
                }
            """
            # 亮色主题标题容器样式 - 浅色背景和深色文字
            title_container_style = """
                QWidget {
                    background-color: #F8F9FA;
                    padding: 3px;
                    border-bottom: 1px solid #e0e0e0;
                }
                QLabel {
                    color: #333333;
                    font-weight: 500;
                }
            """

        # 创建标签面板计数徽章
        label_badge = QtWidgets.QLabel(f"{label_count}")
        label_badge.setAlignment(QtCore.Qt.AlignCenter)
        label_badge.setStyleSheet(badge_style)

        # 创建标签面板标题
        label_title_widget = QtWidgets.QWidget()
        label_title_widget.setStyleSheet(title_container_style)
        label_layout = QtWidgets.QHBoxLayout(label_title_widget)
        label_layout.setContentsMargins(5, 2, 5, 2)
        label_layout.setSpacing(0)

        label_text = QtWidgets.QLabel(self.tr("标签列表"))
        label_layout.addWidget(label_text)
        label_layout.addWidget(label_badge)
        label_layout.addStretch()

        # 为文件列表创建计数徽章
        file_badge = QtWidgets.QLabel(f"{file_count}")
        file_badge.setAlignment(QtCore.Qt.AlignCenter)
        file_badge.setStyleSheet(badge_style)

        # 创建文件列表标题
        file_title_widget = QtWidgets.QWidget()
        file_title_widget.setStyleSheet(title_container_style)
        file_layout = QtWidgets.QHBoxLayout(file_title_widget)
        file_layout.setContentsMargins(5, 2, 5, 2)
        file_layout.setSpacing(0)

        file_text = QtWidgets.QLabel(self.tr("文件列表"))
        file_layout.addWidget(file_text)
        file_layout.addWidget(file_badge)
        file_layout.addStretch()

        # 为多边形标签面板创建计数徽章
        shape_badge = QtWidgets.QLabel(f"{shape_count}")
        shape_badge.setAlignment(QtCore.Qt.AlignCenter)
        shape_badge.setStyleSheet(badge_style)

        # 创建多边形标签面板标题
        shape_title_widget = QtWidgets.QWidget()
        shape_title_widget.setStyleSheet(title_container_style)
        shape_layout = QtWidgets.QHBoxLayout(shape_title_widget)
        shape_layout.setContentsMargins(5, 2, 5, 2)
        shape_layout.setSpacing(0)

        shape_text = QtWidgets.QLabel(self.tr("当前标注"))
        # 移除自定义标题样式
        shape_layout.addWidget(shape_text)
        shape_layout.addWidget(shape_badge)
        shape_layout.addStretch()

        # 为标记面板创建计数徽章
        flag_badge = QtWidgets.QLabel(f"{flag_count}")
        flag_badge.setAlignment(QtCore.Qt.AlignCenter)
        flag_badge.setStyleSheet(badge_style)

        # 创建标记面板标题
        flag_title_widget = QtWidgets.QWidget()
        flag_title_widget.setStyleSheet(title_container_style)
        flag_layout = QtWidgets.QHBoxLayout(flag_title_widget)
        flag_layout.setContentsMargins(5, 2, 5, 2)
        flag_layout.setSpacing(0)

        flag_text = QtWidgets.QLabel(self.tr("标记"))
        flag_layout.addWidget(flag_text)
        flag_layout.addWidget(flag_badge)
        flag_layout.addStretch()

        # 为文件信息创建标题
        info_title_widget = QtWidgets.QWidget()
        info_title_widget.setStyleSheet(title_container_style)
        info_layout = QtWidgets.QHBoxLayout(info_title_widget)
        info_layout.setContentsMargins(5, 2, 5, 2)
        info_layout.setSpacing(0)

        info_text = QtWidgets.QLabel(self.tr("文件信息"))
        # 移除自定义标题样式
        info_layout.addWidget(info_text)
        info_layout.addStretch()

        # 设置各个 dock 窗口的标题栏小部件
        if hasattr(self, "file_dock"):
            self.file_dock.setTitleBarWidget(file_title_widget)

        if hasattr(self, "label_dock"):
            self.label_dock.setTitleBarWidget(label_title_widget)

        if hasattr(self, "shape_dock"):
            self.shape_dock.setTitleBarWidget(shape_title_widget)

        if hasattr(self, "flag_dock"):
            self.flag_dock.setTitleBarWidget(flag_title_widget)

        if hasattr(self, "file_info_dock"):
            self.file_info_dock.setTitleBarWidget(info_title_widget)

    def _darken_color(self, hex_color, percent):
        """使颜色变暗一定百分比"""
        # 从十六进制转换为rgb
        h = hex_color.lstrip('#')
        rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

        # 使颜色变暗
        rgb_darker = tuple(max(0, c - int(c * percent / 100)) for c in rgb)

        # 转回十六进制
        return '#{:02x}{:02x}{:02x}'.format(*rgb_darker)

    def openShortcutsDialog(self):
        """打开快捷键设置对话框"""
        dialog = ShortcutsDialog(self)
        dialog.exec_()

    def keyPressEvent(self, event):
        # 处理Delete键删除操作
        if event.key() == QtCore.Qt.Key_Delete or event.key() == QtCore.Qt.Key_Backspace:
            if self.canvas.selectedShapes:
                self.deleteSelectedShape()
            return

        # F11键切换全屏模式
        if event.key() == QtCore.Qt.Key_F11:
            self.toggleFullScreen()
            return

        # 空格键切换选中形状的显示/隐藏状态
        if event.key() == QtCore.Qt.Key_Space:
            # 在编辑模式下且有选中的形状时，切换显示/隐藏状态
            if self.canvas.editing() and self.canvas.selectedShapes:
                for shape in self.canvas.selectedShapes:
                    # 查找对应的标签项
                    item = self.labelList.findItemByShape(shape)
                    if item:
                        # 切换复选框状态
                        current_state = item.checkState()
                        new_state = QtCore.Qt.Unchecked if current_state == QtCore.Qt.Checked else QtCore.Qt.Checked
                        item.setCheckState(new_state)
                        # 因为状态变化会自动通过信号槽调用labelItemCheckStateChanged方法
                        # 所以这里不需要额外更新画布的可见性
                return

        # 如果没有使用上面的快捷键，则将事件传递给父类处理
        super(MainWindow, self).keyPressEvent(event)

    def _get_default_label_color(self, label):
        """生成标签的默认颜色"""
        # 使用标签名称的哈希值生成稳定的颜色
        hash_value = sum(ord(c) for c in label) % 100
        # 黄金比例共轭用于生成不同的颜色
        hue = (hash_value * 0.618033988749895) % 1.0
        # 转换HSV到RGB
        r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(hue, 0.8, 0.95)]
        return (r, g, b)

    def setLightTheme(self, update_actions=True):
        """设置为亮色主题"""
        # 保存当前主题设置
        self.currentTheme = "light"

        app = QtWidgets.QApplication.instance()
        app.setStyle("Fusion")
        app.setPalette(labelme.styles.get_light_palette())
        app.setStyleSheet(labelme.styles.LIGHT_STYLE)

        # 更新选中状态（如果动作已初始化且需要更新）
        if update_actions and hasattr(self, 'actions') and hasattr(self.actions, 'lightTheme'):
            self.actions.lightTheme.setChecked(True)
            self.actions.darkTheme.setChecked(False)
            self.actions.defaultTheme.setChecked(False)

        # 重置标签组件和形状组件的主题
        if hasattr(self, 'labelList'):
            self.labelList.setDarkMode(False)
        if hasattr(self, 'uniqLabelList'):
            self.uniqLabelList.setDarkMode(False)

        # 重置标签对话框主题
        if hasattr(self, 'labelDialog'):
            self.labelDialog.setThemeStyleSheet(is_dark=False)
            # 更新标签云布局中的所有标签项
            if hasattr(self.labelDialog, 'cloudContainer') and self.labelDialog.cloudContainer:
                for label_item in self.labelDialog.cloudContainer.label_items:
                    label_item.setDarkTheme(False)

        # 更新dock窗口标题栏
        self.updateDockTitles()

        # 更新配置
        self._config["theme"] = "light"
        try:
            from labelme.config import save_config
            save_config(self._config)
        except Exception as e:
            logger.exception("保存主题配置失败: %s", e)

        # 更新文件列表的主题
        if hasattr(self, 'fileListWidget'):
            # 为文件列表设置亮色主题样式
            self.fileListWidget.setStyleSheet("""
                QListWidget {
                    background-color: #fafafa;
                    alternate-background-color: #f5f5f5;
                    border: 1px solid #e0e0e0;
                    border-radius: 8px;
                    padding: 5px;
                }
                QListWidget::item {
                    padding: 8px;
                    border-radius: 4px;
                    margin: 2px 1px;
                }
                QListWidget::item:hover {
                    background-color: #e6f3ff;
                }
                QListWidget::item:selected {
                    background-color: #0078d7;
                    color: white;
                }
            """)

    def setDarkTheme(self, update_actions=True):
        """设置为暗黑主题"""
        # 保存当前主题设置
        self.currentTheme = "dark"

        app = QtWidgets.QApplication.instance()
        app.setStyle("Fusion")
        app.setPalette(labelme.styles.get_dark_palette())
        app.setStyleSheet(labelme.styles.DARK_STYLE)

        # 更新选中状态（如果动作已初始化且需要更新）
        if update_actions and hasattr(self, 'actions') and hasattr(self.actions, 'darkTheme'):
            self.actions.lightTheme.setChecked(False)
            self.actions.darkTheme.setChecked(True)
            self.actions.defaultTheme.setChecked(False)

        # 更新标签组件和形状组件的主题
        if hasattr(self, 'labelList'):
            self.labelList.setDarkMode(True)
        if hasattr(self, 'uniqLabelList'):
            self.uniqLabelList.setDarkMode(True)

        # 更新标签对话框主题
        if hasattr(self, 'labelDialog'):
            self.labelDialog.setThemeStyleSheet(is_dark=True)
            # 更新标签云布局中的所有标签项
            if hasattr(self.labelDialog, 'cloudContainer') and self.labelDialog.cloudContainer:
                for label_item in self.labelDialog.cloudContainer.label_items:
                    label_item.setDarkTheme(True)

        # 更新dock窗口标题栏
        self.updateDockTitles()

        # 更新配置
        self._config["theme"] = "dark"
        try:
            from labelme.config import save_config
            save_config(self._config)
        except Exception as e:
            logger.exception("保存主题配置失败: %s", e)

        # 更新文件列表和标记组件的主题
        if hasattr(self, 'fileListWidget'):
            # 为文件列表设置暗色主题样式
            self.fileListWidget.setStyleSheet("""
                QListWidget {
                    background-color: #252526;
                    alternate-background-color: #2d2d30;
                    border: 1px solid #3e3e42;
                    border-radius: 8px;
                    padding: 5px;
                    color: #cccccc;
                }
                QListWidget::item {
                    padding: 8px;
                    border-radius: 4px;
                    margin: 2px 1px;
                }
                QListWidget::item:hover {
                    background-color: #3e3e42;
                }
                QListWidget::item:selected {
                    background-color: #007acc;
                    color: #ffffff;
                }
            """)

    def setDefaultTheme(self, update_actions=True):
        """恢复原始主题"""
        # 保存当前主题设置
        self.currentTheme = "default"

        app = QtWidgets.QApplication.instance()
        app.setStyle("")  # 使用默认样式
        app.setPalette(QtWidgets.QApplication.style().standardPalette())
        app.setStyleSheet("")  # 清除所有样式表

        # 更新选中状态（如果动作已初始化且需要更新）
        if update_actions and hasattr(self, 'actions') and hasattr(self.actions, 'defaultTheme'):
            self.actions.lightTheme.setChecked(False)
            self.actions.darkTheme.setChecked(False)
            self.actions.defaultTheme.setChecked(True)

        # 重置标签组件和形状组件的主题
        if hasattr(self, 'labelList'):
            self.labelList.setDarkMode(False)
        if hasattr(self, 'uniqLabelList'):
            self.uniqLabelList.setDarkMode(False)

        # 重置标签对话框主题
        if hasattr(self, 'labelDialog'):
            self.labelDialog.setThemeStyleSheet(is_dark=False)
            # 更新标签云布局中的所有标签项
            if hasattr(self.labelDialog, 'cloudContainer') and self.labelDialog.cloudContainer:
                for label_item in self.labelDialog.cloudContainer.label_items:
                    label_item.setDarkTheme(False)

        # 更新dock窗口标题栏
        self.updateDockTitles()

        # 更新配置
        self._config["theme"] = "default"
        try:
            from labelme.config import save_config
            save_config(self._config)
        except Exception as e:
            logger.exception("保存主题配置失败: %s", e)

    def toggleShowLabelNames(self, checked):
        """切换是否在标注上显示标签名称

        Args:
            checked: 从QAction的toggled信号接收到的选中状态
        """
        self._showLabelNames = checked
        Shape.show_label_names = checked

        # 根据主选项状态启用或禁用子选项
        for option in self.labelNameOptions:
            option.setEnabled(checked)

        # 刷新画布
        self.canvas.update()

    def toggleShowLabelText(self, checked):
        """切换是否在标签中显示标签名称"""
        self._showLabelText = checked
        Shape.show_label_text = checked
        self.canvas.update()

    def toggleShowLabelGID(self, checked):
        """切换是否在标签中显示GID"""
        self._showLabelGID = checked
        Shape.show_label_gid = checked
        self.canvas.update()

    def toggleShowLabelDesc(self, checked):
        """切换是否在标签中显示描述"""
        self._showLabelDesc = checked
        Shape.show_label_desc = checked
        self.canvas.update()

    def createDockLikeAction(self, title, slot, checked=False):
        """创建一个类似于QDockWidget.toggleViewAction()返回的QAction"""
        action = QtWidgets.QAction(title, self)
        action.setCheckable(True)
        action.setChecked(checked)
        action.toggled.connect(slot)
        return action

    def startProgress(self, message, max_value=100):
        """显示进度条并设置最大值"""
        self.statusBar().showMessage(message)
        self.statusProgress.show()
        self.statusProgress.setMaximum(max_value)
        self.statusProgress.setValue(0)
        # 确保模式标签仍然可见
        self.modeLabel.show()
        QtWidgets.QApplication.processEvents()

    def setProgress(self, value):
        """更新进度条的值"""
        self.statusProgress.setValue(value)
        QtWidgets.QApplication.processEvents()

    def endProgress(self, message="完成"):
        """隐藏进度条并显示完成消息"""
        self.statusBar().showMessage(message, 5000)  # 显示5秒
        self.statusProgress.hide()
        # 确保模式标签仍然可见
        self.modeLabel.show()
        QtWidgets.QApplication.processEvents()

    def toggleFullScreen(self):
        """切换全屏模式"""
        if self.isFullScreen():
            self.showNormal()  # 先恢复正常窗口大小
            self.showMaximized()  # 然后最大化
        else:
            self.showFullScreen()

    def updateModeLabel(self, mode_text):
        self.modeLabel.setText(f"当前模式: {mode_text}")

    def labelItemSelectedForDrawing(self, label_text, shape_type):
        """当从标签列表中选择标签时，设置为绘制模式并使用该标签进行标注"""
        # 确保有图像加载
        if not self.canvas.pixmap:
            self.status(self.tr("请先打开一个图像"))
            return

        # 确保当前有创建模式的权限
        if not self.actions.createMode.isEnabled():
            self.status(self.tr("当前模式下无法创建新标注"))
            return

        # 清除当前标注列表中的选择
        self.labelList.clearSelection()
        # 清除画布上选中的形状
        self.canvas.deSelectShape()

        # 设置标签对话框中的文本
        self.labelDialog.edit.setText(label_text)

        # 记住这个标签供下次使用
        self._previous_label_text = label_text

        # 根据形状类型选择正确的创建模式
        create_modes = {
            "polygon": self.actions.createMode,
            "rectangle": self.actions.createRectangleMode,
            "circle": self.actions.createCircleMode,
            "line": self.actions.createLineMode,
            "point": self.actions.createPointMode,
            "linestrip": self.actions.createLineStripMode,
            "ai_polygon": self.actions.createAiPolygonMode,
            "ai_mask": self.actions.createAiMaskMode
        }

        # 如果存在对应的创建模式动作，触发它
        if shape_type in create_modes and create_modes[shape_type].isEnabled():
            # 先将所有创建模式取消选中
            for action in create_modes.values():
                if hasattr(action, 'setChecked'):
                    action.setChecked(False)

            # 选中对应的创建模式
            action = create_modes[shape_type]
            if hasattr(action, 'setChecked'):
                action.setChecked(True)

            # 切换到该绘制模式
            self.toggleDrawMode(False, createMode=shape_type)

            # 更新状态栏
            shape_type_names = {
                "polygon": "多边形",
                "rectangle": "矩形",
                "circle": "圆形",
                "line": "线段",
                "point": "点",
                "linestrip": "折线",
                "ai_polygon": "AI多边形",
                "ai_mask": "AI蒙版"
            }

            shape_type_name = shape_type_names.get(shape_type, shape_type)
            self.status(
                self.tr(f"已选择标签 '{label_text}'，使用{shape_type_name}工具绘制"))

    def _get_label_default_color(self, label):
        """生成标签默认颜色"""
        hash_value = sum(ord(c) for c in label) % 100
        hue = (hash_value * 0.618033988749895) % 1.0
        r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(hue, 0.8, 0.95)]
        return QtGui.QColor(r, g, b)

    def get_label_default_color(self, label):
        """获取标签的默认颜色"""
        # 先尝试从配置中获取颜色
        if self._config["shape_color"] != "auto":
            return QtGui.QColor(*self._config["shape_color"])

        # 如果是自动颜色，使用标签对应的颜色
        item = self.uniqLabelList.findItemByLabel(label)
        if item is not None:
            # 尝试从标签项中提取颜色
            text = item.text()
            if "●" in text:
                try:
                    # 尝试从文本中解析颜色
                    color_str = text.split('color="')[1].split('">')[0]
                    r = int(color_str[1:3], 16)
                    g = int(color_str[3:5], 16)
                    b = int(color_str[5:7], 16)
                    return QtGui.QColor(r, g, b)
                except (IndexError, ValueError):
                    pass

        # 否则根据标签计算默认颜色
        return self._get_default_label_color(label)

    def loadLabels(self, shapes):
        s = []
        for shape in shapes:
            label = shape["label"]
            points = shape["points"]
            shape_type = shape["shape_type"]
            flags = shape["flags"]
            group_id = shape["group_id"]
            description = shape["description"]
            other_data = shape["other_data"]

            if not points:
                # skip point-empty shape
                continue

            shape = Shape(
                label=label,
                shape_type=shape_type,
                group_id=group_id,
                description=description,
            )
            for x, y in points:
                shape.addPoint(QtCore.QPointF(x, y))
            shape.close()

            default_flags = {}
            if self._config["label_flags"]:
                for pattern, keys in self._config["label_flags"].items():
                    if re.match(pattern, label):
                        for key in keys:
                            default_flags[key] = False
            shape.flags = default_flags
            shape.flags.update(flags)
            shape.other_data = other_data

            s.append(shape)
        self.loadShapes(s)

    def loadConfig(self, config_file=None, config_from_args=None):
        # 初始化默认配置
        default_config = {
            "auto_save": False,
            "store_data": False,
            "keep_prev": False,
            "keep_prev_scale": False,
            "keep_prev_brightness": False,
            "keep_prev_contrast": False,
            "canvas": {
                "grid_show": False,
                "grid_size": 30,
                "paint_label": False,
                "paint_label_font_size": 9,
                "label_font_family": "Noto Sans Regular",
                "label_font_weight": "medium",
                "paint_label_fill": True,
                "show_texts": True,
                "fill_drawing": False,
                "epsilon": 10.0,
            },
            "label_flags": None,
            "shape": {
                "line_color": [0, 255, 0, 128],
                "fill_color": [180, 235, 180, 60],
                "vertex_fill_color": [0, 255, 0, 255],
                "select_line_color": [255, 255, 255, 128],
                "select_fill_color": [0, 255, 0, 155],
                "hvertex_fill_color": [255, 255, 255, 255],
            },
            "shape_color": "auto",  # 或manual
            "default_shape_color": [0, 255, 0],  # 当shape_color为非auto时使用
            "label_colors": {},  # 存储每个标签对应的颜色
            "label_order": [],  # 存储标签的自定义顺序
            "label_cloud_layout": False,  # 标签云流式布局，默认关闭
            "flag_dock": {
                "show": True,
            },
            "label_dock": {
                "show": True,
            },
            "shape_dock": {
                "show": True,
            },
            "file_dock": {
                "show": True,
            },
        }

        # 如果提供了配置文件和配置参数
        # ...现有代码...

        # 如果有配置文件
        if config_file and os.path.isfile(config_file):
            self._config_file = config_file
            with open(config_file) as f:
                try:
                    user_config = yaml.safe_load(f) or {}
                except yaml.YAMLError as e:
                    logger.error(f"Error parsing config file: {e}")
                    user_config = {}
                # 合并用户配置到默认配置
                self._update_dict(default_config, user_config)

        # 加载完配置后初始化标签菜单
        if self._config.get('label_order'):
            self._update_label_menu_from_config()

        # 设置UI状态
        if hasattr(self, 'toggle_label_cloud_layout_action'):
            self.toggle_label_cloud_layout_action.setChecked(
                self._config["label_cloud_layout"])

        return default_config

    def toggleLabelCloudLayout(self, enabled=None):
        """切换标签云布局

        Args:
            enabled: 如果提供，直接设置为该值；否则切换当前状态
        """
        if enabled is None:
            enabled = not self._config["label_cloud_layout"]
        self._config["label_cloud_layout"] = enabled
        self.cloud_layout_action.setChecked(enabled)

        # 保存配置
        try:
            from labelme.config import save_config
            save_config(self._config)
        except Exception as e:
            logger.exception("保存标签云布局配置失败: %s", e)

    def eventFilter(self, obj, event):
        """事件过滤器，用于处理菜单和其他UI元素的特殊行为
        """
        # 暂时禁用事件过滤，让原生菜单行为正常工作
        # 直接返回False表示不处理事件，让Qt默认处理逻辑接管
        return False

    def updateFileItemCheckState(self):
        """根据当前标注列表更新文件列表中当前文件的复选框状态"""
        if self.filename and self.fileListWidget:
            # 查找当前文件在文件列表中的项
            items = self.fileListWidget.findItems(
                "   " + osp.basename(self.filename), Qt.MatchEndsWith)
            if not items and self.filename in self.imageList:
                # 如果没有找到，尝试通过存储的数据查找
                for i in range(self.fileListWidget.count()):
                    item = self.fileListWidget.item(i)
                    if item.data(Qt.UserRole) == self.filename:
                        items = [item]
                        break

            if items:
                item = items[0]
                # 根据当前标注列表中是否有标注来设置复选框状态
                has_annotations = len(self.labelList) > 0
                item.setCheckState(
                    Qt.Checked if has_annotations else Qt.Unchecked)

    def toggleShapesVisibility(self, shapes):
        """通过空格键切换选中形状的可见性"""
        for shape in shapes:
            # 查找对应的标签项
            item = self.labelList.findItemByShape(shape)
            if item:
                # 切换复选框状态
                current_state = item.checkState()
                new_state = QtCore.Qt.Unchecked if current_state == QtCore.Qt.Checked else QtCore.Qt.Checked
                item.setCheckState(new_state)

    def setupCanvas(self):
        # Create canvas and set its default shape colors/text.
        canvas = self.canvas = Canvas(
            epsilon=self._config["epsilon"],
            double_click=self._config["canvas"]["double_click"],
            num_backups=self._config["canvas"]["num_backups"],
            crosshair=self._config["canvas"]["crosshair"],
        )
        # 连接信号到槽
        canvas.zoomRequest.connect(self.zoomRequest)
        canvas.scrollRequest.connect(self.scrollRequest)
        canvas.newShape.connect(self.newShape)
        canvas.selectionChanged.connect(self.shapeSelectionChanged)
        canvas.shapeMoved.connect(self.setDirty)
        canvas.drawingPolygon.connect(self.toggleDrawingSensitive)
        canvas.vertexSelected.connect(self.actions.edit.setEnabled)
        canvas.mouseMoved.connect(self.updateStatusBarCoordinates)
        canvas.modeChanged.connect(self.updateModeLabel)
        canvas.toggleVisibilityRequest.connect(self.toggleShapesVisibility)
        canvas.editLabelRequest.connect(self._edit_label)  # 连接双击编辑标签信号
