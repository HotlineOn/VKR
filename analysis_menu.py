from PySide6.QtWidgets import (
    QWidget, QComboBox, QCheckBox, QScrollArea, QVBoxLayout, QLabel, QHBoxLayout,
    QSizePolicy, QPushButton, QLineEdit, QMessageBox, QTabWidget,
    QButtonGroup, QSplitter, QGridLayout, QTableWidget, QTableWidgetItem, QSlider, QStackedWidget,
    QHeaderView
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtGui import QIntValidator, QFont, QCloseEvent
from word_doc_matrix import SelectionMatrix, find_dat_files, get_info_from_name_of_dat_file
from research import (
    reduce_data_features_class, make_word_cloud_plot, clusterization,  # make_correlation_matrix_plot,
    classification_report_for_table, classification_result, reduce_data_features_cluster, CLASSIFICATION_METHODS,
    calculate_jaccard_matrix, calculate_weighted_jaccard_matrix
)
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
)
from matplotlib.figure import Figure
# from matplotlib.animation import TimedAnimation
from pandas import DataFrame
from db_sqlite3 import Database, TABLE_FIELDS_TRANSLATED
from db_manager_window import FullArticle
from typing import List
# from time import time, sleep
from cycler import cycler

import matplotlib
matplotlib.use('qt5agg')


def make_selection_scroll(classes_names: List[str]):
    scroll = QScrollArea()
    scroll.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
    scroll_layout = QVBoxLayout()
    for i, class_name in enumerate(classes_names, 1):
        label = QLabel(f'Класс {i}: {class_name}')
        label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        scroll_layout.addWidget(label)
    scroll_widget = QWidget()
    scroll_widget.setLayout(scroll_layout)
    scroll.setWidget(scroll_widget)
    return scroll


def make_default_window(matrix: SelectionMatrix):
    window = QSplitter(Qt.Orientation.Vertical)
    window.setMinimumSize(640, 480)
    win_title = f'{matrix.selection_name}_{matrix.weight_method}_{matrix.class_size}'
    window.setAttribute(Qt.WA_DeleteOnClose, True)
    window.setWindowTitle(win_title)
    window.setWindowFlags(Qt.MSWindowsOwnDC)
    window.setWindowState(Qt.WindowState.WindowMaximized)
    return window


class NT(NavigationToolbar):
    zoom_signal = Signal()

    def __init__(self, canvas, parent):
        self.toolitems = [t for t in NavigationToolbar.toolitems
                          if t[0] in ('Home', 'Back', 'Forward', 'Pan', 'Zoom', 'Save', None)]
        super().__init__(canvas, parent)
        self.release_zoom = self.new_release_zoom
        self.home = self.new_home

    def new_release_zoom(self, *args, **kwargs):
        s = 'release_zoom_event'
        self.canvas.callbacks.process(s, args[0])
        NavigationToolbar.release_zoom(self, *args, **kwargs)
        self.zoom_signal.emit()

    def new_home(self, *args):
        NavigationToolbar.home(self, *args)
        self.zoom_signal.emit()


class QTableNumberItem(QTableWidgetItem):

    def __lt__(self, other):
        try:
            return float(self.text()) < float(other.text())
        except ValueError:
            return super(QTableNumberItem, self).__lt__(other)


class ReducedDataWindow(QSplitter):
    zoom_released = Signal()
    size_changed = Signal(int)

    def __init__(self, reduced_classes: List[DataFrame], classes_names: List[str], db_name: str, centroids=None):
        super(ReducedDataWindow, self).__init__()

        self.dots_window = None
        self.full_article_window = None
        self.warning = None

        points_whole_ax = 5 * 0.8 * 72
        radius = 0.001
        points_radius = 2 * radius / 1.0 * points_whole_ax
        self.const_radius = points_radius

        self.reduced_classes = reduced_classes
        self.db_name = db_name
        self.classes_names = classes_names
        self.n_components = self.reduced_classes[0].shape[1]
        self.centroids = centroids

        self.canvas = FigureCanvas(Figure((5, 3)))
        self.toolbar = NT(self.canvas, self)
        self.toolbar.zoom_signal.connect(self.get_new_lims)
        if self.n_components == 2:
            self.ax = self.canvas.figure.add_subplot()
        if self.n_components == 3:
            self.ax = self.canvas.figure.add_subplot(projection='3d')

        self.colors = list(matplotlib.rcParams['axes.prop_cycle'][0:len(classes_names)])
        self.limits = list()

        figure_layout = QVBoxLayout()
        figure_layout.addWidget(self.toolbar)
        figure_layout.addWidget(self.canvas)
        figure_widget = QWidget()
        figure_widget.setMinimumSize(480, 360)
        figure_widget.setMaximumSize(1920, 1080)
        # figure_widget.setMaximumWidth(1680)
        figure_widget.setLayout(figure_layout)

        classes_names_layout = QVBoxLayout()
        self.button_group = QButtonGroup()
        self.button_group.setExclusive(False)
        for i, (color, class_name) in enumerate(zip(self.colors, classes_names)):
            widget = QCheckBox(class_name)
            # widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            self.button_group.addButton(widget, id=i)
            checked_params = {"background-color": f"{color['color']} ", "border": f"2px solid {color['color']} "}
            unchecked_params = {"border": f"2px solid {color['color']} "}  # "background-color": f"white",
            # print(checked_params)
            widget.setStyleSheet('')
            widget.setStyleSheet("QCheckBox::indicator {border-radius: 6px} " +
                                 "QCheckBox::indicator:checked " +
                                 str(checked_params).replace(',', ';').replace("'", '') +
                                 "QCheckBox::indicator:unchecked " +
                                 str(unchecked_params).replace(',', ';').replace("'", ''))
            widget.setChecked(True)
            widget.stateChanged.connect(self.get_checked_classes_plot)
            classes_names_layout.addWidget(widget)

        # self.ok_button = QPushButton('Обновить график')
        # self.ok_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        # self.ok_button.clicked.connect(self.get_checked_classes_plot)

        classes_names_widget = QWidget()
        classes_names_widget.setLayout(classes_names_layout)
        classes_names_scroll_area = QScrollArea()
        # classes_names_scroll_area.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        classes_names_scroll_area.setWidget(classes_names_widget)
        classes_names_scroll_area.setAlignment(Qt.AlignVCenter)

        self.dots_radius = QSlider(Qt.Orientation.Horizontal)
        self.dots_radius.setRange(6, 38)
        self.dots_radius.setPageStep(8)
        self.dots_radius.setSingleStep(1)
        self.dots_radius.setTickPosition(self.dots_radius.TickPosition.TicksAbove)
        self.dots_radius.setSliderDown(True)

        self.r = self.dots_radius.value() * points_radius
        # self.dots_radius.setValue(5)
        self.dots_radius.actionTriggered.connect(self.slider_action)
        self.dots_radius.sliderReleased.connect(self.slider_released)

        dots_size_layout = QHBoxLayout()
        # dots_size_layout.setSizeConstraint(QHBoxLayout.SizeConstraint.SetMinimumSize)
        dots_size_layout.addWidget(QLabel('Размер точек'))
        dots_size_layout.addWidget(self.dots_radius)
        dots_size_widget = QWidget()
        dots_size_widget.setLayout(dots_size_layout)

        dots_table_button = QPushButton('Таблица точек')
        dots_table_button.clicked.connect(self.make_dots_table)

        right_menu_layout = QVBoxLayout()
        # right_menu_layout.setSizeConstraint(QVBoxLayout.SizeConstraint.SetMinimumSize)
        right_menu_layout.addWidget(classes_names_scroll_area)
        right_menu_layout.addWidget(dots_size_widget)
        right_menu_layout.addWidget(dots_table_button, alignment=Qt.AlignHCenter)
        # right_menu_layout.addWidget(self.ok_button)

        right_menu_widget = QWidget()
        right_menu_widget.setMinimumWidth(240)
        right_menu_widget.setLayout(right_menu_layout)
        right_menu_widget.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)

        figure_widget.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Expanding)

        self.addWidget(figure_widget)
        self.addWidget(right_menu_widget)
        self.initial_scatter(zip(self.colors, self.reduced_classes))

    def get_new_lims(self):
        self.limits.clear()
        self.limits.append(self.ax.get_xlim())
        self.limits.append(self.ax.get_ylim())
        if self.n_components == 3:
            self.limits.append(self.ax.get_zlim())

    def make_scatter(self, reduced_classes_with_colors):
        for color, reduced_class in reduced_classes_with_colors:
            data = (reduced_class.iloc[:, dim] for dim in range(self.n_components))
            self.ax.scatter(*data, s=self.r, c=color['color'], alpha=1, linewidths=1)

        if self.centroids is not None:
            # data = (self.centroids[:, dim] for dim in range(self.n_components))
            for i, dot in enumerate(self.centroids):
                self.ax.scatter(*dot, c=self.colors[i]['color'], label=f'Центроида кластера {i+1}',
                                s=3*self.r, alpha=1, edgecolors='black', linewidths=1.5)
                self.ax.legend()

        self.ax.set_xlabel('X', c='r')
        self.ax.set_ylabel('Y', c='g')
        if self.n_components == 3:
            self.ax.set_zlabel('Z', c='b')

    def initial_scatter(self, reduced_classes_with_colors):
        self.make_scatter(reduced_classes_with_colors)

        self.get_new_lims()
        self.canvas.draw()

    def scatter_plot(self, reduced_classes_with_colors):
        self.ax.clear()
        self.ax.autoscale(False)

        self.make_scatter(reduced_classes_with_colors)

        self.ax.set_xlim(self.limits[0])
        self.ax.set_ylim(self.limits[1])
        if self.n_components == 3:
            self.ax.set_zlim(self.limits[2])

        self.canvas.draw()

    def get_checked_classes_plot(self):
        checked_boxes = list()
        checkboxes = self.findChildren(QCheckBox)
        for checkbox in checkboxes:
            if checkbox.isChecked():
                button_id = self.button_group.id(checkbox)
                checked_boxes.append((self.colors[button_id], self.reduced_classes[button_id]))
        self.scatter_plot(checked_boxes)

    @Slot(int)
    def slider_action(self, action: int):
        if self.dots_radius.sliderPosition() == self.dots_radius.value():
            return
        if action == QSlider.SliderAction.SliderMove:
            return
        self.dots_radius.setValue(self.dots_radius.sliderPosition())
        self.change_dots_size(self.dots_radius.value())
        self.size_changed.emit(self.dots_radius.value())

    def slider_released(self):
        self.dots_radius.setValue(self.dots_radius.sliderPosition())
        self.change_dots_size(self.dots_radius.value())
        self.size_changed.emit(self.dots_radius.value())

    @Slot(int)
    def change_dots_size(self, new_size: int):
        self.r = self.const_radius * new_size
        self.get_checked_classes_plot()

    def make_dots_table(self):
        self.dots_window = QTableWidget(1, len(self.classes_names))
        # self.dots_window.setMinimumSize(640, 480)
        self.dots_window.setAttribute(Qt.WA_DeleteOnClose, True)
        self.dots_window.setWindowTitle('Координаты точек классов')
        self.dots_window.setWindowFlags(Qt.MSWindowsOwnDC)
        self.dots_window.setSizeAdjustPolicy(QTableWidget.SizeAdjustPolicy.AdjustToContents)
        self.dots_window.setShowGrid(False)

        label_size = 49 if self.n_components == 2 else 53
        self.dots_window.setHorizontalHeaderLabels([text[:label_size] for text in self.classes_names])

        self.dots_window.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.dots_window.horizontalHeader().setSectionsMovable(True)
        self.dots_window.verticalHeader().setHidden(True)
        self.dots_window.verticalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)

        for i, reduced_class in enumerate(self.reduced_classes):
            urls = reduced_class.index
            axis = reduced_class.columns

            table = QTableWidget(len(urls), len(axis) + 1)
            table.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
            table.setSizeAdjustPolicy(QTableWidget.SizeAdjustPolicy.AdjustToContents)
            table.setHorizontalHeaderLabels(['url'] + list(axis))
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            table.verticalHeader().setHidden(True)
            table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
            table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
            table.cellDoubleClicked.connect(self.show_full_article)

            for row, url in enumerate(urls):
                table_item = QTableWidgetItem(url)
                table_item.setFlags((Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable))
                table.setItem(row, 0, table_item)

                dot = reduced_class.iloc[row]
                for col, axis_name in enumerate(axis, 1):
                    # table_item = QTableWidgetItem(str(round(axis_val, 5)))
                    table_item = QTableNumberItem(str(round(dot[axis_name], 5)))
                    table_item.setFlags((Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable))
                    table.setItem(row, col, table_item)

            table.setSortingEnabled(True)
            self.dots_window.setCellWidget(0, i, table)
        self.dots_window.show()

    def show_full_article(self):
        fields_for_output = ['url', 'tags_codes', 'tags_names', 'authors', 'title', 'abstract', 'submitted']
        url = self.sender().selectedItems()[0].text()
        article = Database(f'data/{self.db_name}', []).find_article_by_url(url, fields_for_output)
        if article is None:
            error_text = f'В {self.db_name} не найдено статьи с адресом {url}'
            self.warning = QMessageBox(QMessageBox.Icon.Critical, 'Ошибка', error_text)
            self.warning.show()
            return

        self.full_article_window = FullArticle(article, fields_for_output)
        self.full_article_window.setWindowTitle(url)
        self.full_article_window.show()

    # def closeEvent(self, event: QCloseEvent) -> None:
    #     del self.dots_window, self.full_article_window, self.warning, self.const_radius, \
    #         self.reduced_classes, self.db_name, self.classes_names, self.n_components, self.centroids, \
    #         self.colors, self.limits


class ClassificationReport(QSplitter):
    def __init__(self, reports: List[dict], selection_queries: List[str], method: str, kfold_split: int):
        super().__init__()
        self.setOrientation(Qt.Orientation.Vertical)

        tab = QTabWidget()
        tab_labels = ['Лучшая модель на тестовых данных'] if len(reports) == 1 \
            else [f'K={i}' for i in range(1, len(reports) + 1)]

        scroll_top = make_selection_scroll(selection_queries)

        for i, (report, confusion_matrix) in enumerate(reports):
            scroll = QScrollArea()
            scroll.setSizeAdjustPolicy(QScrollArea.AdjustToContents)
            scroll.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
            scroll_layout = QVBoxLayout()
            classification_method_widget = QLabel(f'Метод классификации: {method}')
            classification_method_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            scroll_layout.addWidget(classification_method_widget)
            kfold_split_widget = QLabel(f'Разбиений K-Fold: {kfold_split}')
            kfold_split_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            scroll_layout.addWidget(kfold_split_widget)
            train_accuracy_widget = QLabel(f'Accuracy на валидационной выборке: {round(report["test_accuracy"], 4)}')
            train_accuracy_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            scroll_layout.addWidget(train_accuracy_widget)
            test_accuracy_widget = QLabel(f'Accuracy на тестовой выборке: {round(report["accuracy"], 4)}')
            test_accuracy_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            scroll_layout.addWidget(test_accuracy_widget)

            scroll_widget = QWidget()
            scroll_widget.setLayout(scroll_layout)
            scroll.setWidget(scroll_widget)

            col_count = len(selection_queries)
            table = QTableWidget(4, col_count + 1)
            table.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
            table.setSizeAdjustPolicy(QScrollArea.AdjustToContents)
            row_names = ['precision', 'recall', 'f1-score', 'support']
            table.setHorizontalHeaderLabels([f'Класс {i}' for i in range(1, col_count + 1)] + ['Среднее'])
            table.setVerticalHeaderLabels(row_names)
            bold_font = QFont(table.verticalHeaderItem(0).font())
            bold_font.setBold(True)
            table.horizontalHeaderItem(col_count).setFont(bold_font)
            fields_for_output = [str(j) for j in range(col_count)] + ['macro avg']
            for r, row_name in enumerate(row_names):
                for c, col_name in enumerate(fields_for_output):
                    table_cell = QTableWidgetItem(str(round(report[col_name][row_name], 4)))
                    table_cell.setFlags(Qt.ItemFlag.ItemIsEnabled)
                    table_cell.setTextAlignment(Qt.AlignCenter)
                    if col_name == 'macro avg':
                        table_cell.setFont(bold_font)
                    table.setItem(r, c, table_cell)

            confusion_matrix_table = QTableWidget(col_count, col_count + 1)
            confusion_matrix_table.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
            confusion_matrix_table.setSizeAdjustPolicy(QScrollArea.AdjustToContents)
            confusion_matrix_table.setHorizontalHeaderLabels([f'Класс {i}' for i in range(1, col_count + 1)]
                                                             + ['Всего'])
            confusion_matrix_table.setVerticalHeaderLabels([f'Класс {i}' for i in range(1, col_count + 1)])
            confusion_matrix_table.horizontalHeaderItem(col_count).setFont(bold_font)
            for r in range(col_count):
                for c in range(col_count + 1):
                    if c == col_count:
                        table_cell = QTableWidgetItem(str(sum(confusion_matrix[r])))
                        table_cell.setFont(bold_font)
                    else:
                        table_cell = QTableWidgetItem(str(confusion_matrix[r, c]))
                    table_cell.setFlags(Qt.ItemFlag.ItemIsEnabled)
                    table_cell.setTextAlignment(Qt.AlignCenter)
                    if r == c:
                        table_cell.setFont(bold_font)
                    confusion_matrix_table.setItem(r, c, table_cell)

            bottom_layout = QVBoxLayout()
            bottom_layout.setSizeConstraint(QVBoxLayout.SetMinAndMaxSize)
            bottom_layout.setAlignment((Qt.AlignHCenter | Qt.AlignTop))
            bottom_layout.addWidget(QLabel('Отчёт о классификации'), alignment=Qt.AlignHCenter)
            bottom_layout.addWidget(table)
            bottom_layout.addWidget(QLabel('Матрица ошибок'), alignment=Qt.AlignHCenter)
            bottom_layout.addWidget(confusion_matrix_table)
            bottom_widget = QWidget()
            bottom_widget.setLayout(bottom_layout)
            bottom_scroll = QScrollArea()
            bottom_scroll.setWidget(bottom_widget)
            bottom_scroll.setAlignment((Qt.AlignTop | Qt.AlignHCenter))

            layout = QVBoxLayout()
            layout.setAlignment((Qt.AlignTop | Qt.AlignHCenter))
            layout.addWidget(scroll)
            layout.addWidget(bottom_scroll)
            widget = QWidget()
            widget.setLayout(layout)
            tab.addTab(widget, tab_labels[i])

        self.addWidget(scroll_top)
        self.addWidget(tab)


class MplFigure(FigureCanvas):
    draw_signal = Signal()

    def __init__(self):
        super(MplFigure, self).__init__(figure=Figure((5, 3)))
        self.timer = QTimer()
        self.timer.setInterval(100)
        self.ax = self.figure.add_subplot(4, 1, (1, 3))
        self.ax.grid()
        self.twin_ax = self.ax.twinx()
        self.axes = self.figure.add_axes(self.ax)

    def set_axis_ticks(self, x, words):
        self.ax.set_xticks(range(x + 1), [f'{i}: {w}' for i, w in enumerate(words)], rotation=270)

    def draw_scatter(self, x, dots, words):
        self.ax.scatter([x] * len(dots), dots)
        self.set_axis_ticks(x, words)
        self.draw()

    def draw_best_split_scatter(self, x, dots, words):
        for dot in dots:
            self.ax.scatter(x, dot)
        self.set_axis_ticks(x, words)
        self.draw()

    def clear(self):
        self.figure.clear()
        # self.twin_ax.clear()
        # self.axes.clear()
        self.ax = self.figure.add_subplot(4, 1, (1, 3))
        self.ax.grid()
        self.twin_ax = self.ax.twinx()
        self.axes = self.figure.add_axes(self.ax)


class FrequencyAnalysis(QSplitter):

    def __init__(self, win_title: str):
        super().__init__()
        self.setChildrenCollapsible(False)
        self.setOrientation(Qt.Orientation.Horizontal)
        self.setMinimumSize(640, 480)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setWindowTitle(win_title)
        self.setWindowFlags(Qt.MSWindowsOwnDC)
        self.setWindowState(Qt.WindowState.WindowMaximized)

        self.warning = None
        # self.matrix = matrix
        self.means = list()
        self.counts = list()
        self.plotting_it = None
        self.timer = QTimer()

        figure_layout = QVBoxLayout()
        self.canvas = MplFigure()
        toolbar = NT(self.canvas, self)
        figure_layout.addWidget(toolbar)
        figure_layout.addWidget(self.canvas)
        figure_widget = QWidget()
        figure_widget.setLayout(figure_layout)
        figure_widget.setMinimumSize(640, 480)
        # self.figure_rect = figure_widget.rect()

        analysis_type_layout = QHBoxLayout()
        analysis_type_layout.addWidget(QLabel('Удаление слов:'))
        self.analysis_type = QComboBox()
        self.analysis_type.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.analysis_type.addItems(['Одиночное', 'С накоплением'])
        analysis_type_layout.addWidget(self.analysis_type)

        classification_method_layout = QHBoxLayout()
        classification_method_layout.addWidget(QLabel('Метод:'))
        self.classification_method = QComboBox()
        self.classification_method.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.classification_method.addItems(CLASSIFICATION_METHODS)
        self.classification_method.setCurrentIndex(2)
        classification_method_layout.addWidget(self.classification_method)

        classification_metric_layout = QHBoxLayout()
        classification_metric_layout.setAlignment(Qt.AlignLeft)
        classification_metric_layout.addWidget(QLabel('Метрика:'))
        self.classification_metric = QComboBox()
        self.classification_metric.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.classification_metric.addItems(['precision', 'recall', 'f1-score'])
        self.classification_metric.setCurrentIndex(2)
        classification_metric_layout.addWidget(self.classification_metric)

        self.kfold_split_num = QLineEdit()
        self.kfold_split_num.setFixedWidth(215)
        kfold_validator = QIntValidator(2, 10)
        self.kfold_split_num.setValidator(kfold_validator)
        self.kfold_split_num.setPlaceholderText('Количество разбиений K-Fold')

        self.num_of_words_to_delete = QLineEdit()
        self.num_of_words_to_delete.setFixedWidth(215)
        words_num_validator = QIntValidator(0, 100)
        self.num_of_words_to_delete.setValidator(words_num_validator)
        self.num_of_words_to_delete.setPlaceholderText('Число удаляемых слов (макс. 100)')

        self.best_split_only = QCheckBox('Только лучшая тестовая модель')
        self.best_split_only.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.stacked_buttons = QStackedWidget()
        self.stacked_buttons.setFixedSize(220, 45)

        start_button_layout = QHBoxLayout()
        start_button_layout.setAlignment(Qt.AlignLeft)
        self.start_button = QPushButton('Начать')
        self.start_button.setFixedSize(100, 25)
        # self.start_button.clicked.connect(self.start_plotting)
        start_button_layout.addWidget(self.start_button)
        start_button_widget = QWidget()
        start_button_widget.setLayout(start_button_layout)
        self.stacked_buttons.addWidget(start_button_widget)

        buttons_layout = QHBoxLayout()
        # buttons_layout.setAlignment(Qt.AlignHCenter)
        self.continue_button = QPushButton('Продолжить')
        self.continue_button.setFixedSize(100, 25)
        self.continue_button.clicked.connect(self.continue_plotting)
        stop_button = QPushButton('Закончить')
        stop_button.setFixedSize(100, 25)
        stop_button.clicked.connect(self.end_plotting)
        buttons_layout.addWidget(self.continue_button)
        buttons_layout.addWidget(stop_button)
        buttons_widget = QWidget()
        buttons_widget.setLayout(buttons_layout)
        self.stacked_buttons.addWidget(buttons_widget)
        self.stacked_buttons.setVisible(True)

        settings_menu = QVBoxLayout()
        settings_menu.setAlignment(Qt.AlignCenter)  # (Qt.AlignHCenter | Qt.AlignVCenter)
        settings_menu.addLayout(analysis_type_layout)
        settings_menu.addLayout(classification_method_layout)
        settings_menu.addLayout(classification_metric_layout)
        settings_menu.addWidget(self.kfold_split_num)
        settings_menu.addWidget(self.num_of_words_to_delete)
        settings_menu.addWidget(self.best_split_only)
        settings_menu.addWidget(self.stacked_buttons)
        settings_menu_widget = QWidget()
        settings_menu_widget.setLayout(settings_menu)
        settings_menu_widget.setFixedWidth(240)

        self.addWidget(figure_widget)
        self.addWidget(settings_menu_widget)

    def make_format(self, current, other):
        # current and other are axes
        def format_coord(x, y):
            # x, y are data coordinates
            # convert to display coords
            display_coord = current.transData.transform((x, y))
            inv = other.transData.inverted()
            # convert back to data coords with respect to ax
            ax_coord = inv.transform(display_coord)
            coords = [ax_coord, (x, y)]
            return ('{:}: {:<10}    Количество слов: {:<}'.format(self.classification_metric.currentText(),
                                                                  str(round(coords[0][1], 4)), str(int(coords[1][1]))))

        return format_coord

    def calculate_mean(self, score, best_split_only):
        if best_split_only:
            self.means.append(score[1])
        else:
            m = sum(score) / len(score)
            diff = list(map(lambda x: abs(x - m), score))
            self.means.append(score[diff.index(min(diff))])

    @Slot(object)
    def start_plotting(self, matrix: SelectionMatrix):
        if not self.kfold_split_num.text():
            self.warning = QMessageBox(QMessageBox.Icon.Critical, 'Ошибка', 'Введите количество разбиений')
            self.warning.show()
            return
        if self.kfold_split_num.validator().validate(self.kfold_split_num.text(), 0)[0] != QIntValidator.Acceptable:
            self.warning = QMessageBox(QMessageBox.Icon.Critical, 'Ошибка', 'Некорректное число разбиений (2-20)')
            self.warning.show()
            return

        delete_num = self.num_of_words_to_delete.text()
        if delete_num:
            if self.num_of_words_to_delete.validator().validate(delete_num, 0)[0] != QIntValidator.Acceptable:
                self.warning = QMessageBox(QMessageBox.Icon.Critical, 'Ошибка', 'Некорректное число разбиений (2-20)')
                self.warning.show()
                return

        self.canvas.clear()

        self.continue_button.setEnabled(True)
        self.classification_method.setEnabled(False)
        self.analysis_type.setEnabled(False)
        self.classification_metric.setEnabled(False)
        self.num_of_words_to_delete.setEnabled(False)
        self.kfold_split_num.setEnabled(False)
        self.best_split_only.setEnabled(False)

        self.stacked_buttons.setCurrentIndex(1)

        self.plotting_it = self.plotting_generator(matrix)
        n_classes = len(matrix.classes_names)
        words_count = matrix.words_count
        k_splits = int(self.kfold_split_num.text())
        classification_metric = self.classification_metric.currentText()
        best_split_only = self.best_split_only.checkState()
        analysis_type = self.analysis_type.currentText()

        score = classification_result(matrix.df, n_classes, self.classification_method.currentText(),
                                      classification_metric, k_splits, best_split_only)

        self.calculate_mean(score, best_split_only)
        if best_split_only:
            self.canvas.ax.set_prop_cycle(cycler(color=['#0047ab', '#ed760e']))
            self.canvas.ax.scatter(0, score[0], label=f'{classification_metric}: тестовое')
            self.canvas.ax.scatter(0, score[1], label=f'{classification_metric}: валидационное')
            self.canvas.set_axis_ticks(0, ['Без изменений'])
            self.canvas.draw()

        else:
            self.canvas.draw_scatter(0, score, ['Без изменений'])

        if analysis_type == 'С накоплением':
            self.counts.append(words_count)

        if delete_num:
            for i in range(int(delete_num)):
                self.continue_plotting()

    def plotting_generator(self, matrix: SelectionMatrix):
        research_df = matrix.df.copy()
        n_classes = len(matrix.classes_names)
        words_count = matrix.words_count
        k_splits = int(self.kfold_split_num.text())
        best_split_only = self.best_split_only.checkState()
        analysis_type = self.analysis_type.currentText()
        most_common_words = [w for w in matrix.most_common_words if w[0] not in matrix.deleted_words]
        words = ['Без изменений'] + list(map(lambda x: x[0], most_common_words))

        for i, (word, count) in enumerate(most_common_words, 1):
            score = classification_result(research_df.drop(columns=word), n_classes,
                                          self.classification_method.currentText(),
                                          self.classification_metric.currentText(), k_splits, best_split_only)

            if best_split_only:
                self.canvas.draw_best_split_scatter(i, score, words[:i+1])
            else:
                self.canvas.draw_scatter(i, score, words[:i+1])

            self.calculate_mean(score, best_split_only)

            if analysis_type == 'Одиночное':
                self.counts.append(count)
            if analysis_type == 'С накоплением':
                words_count -= count
                self.counts.append(words_count)
                research_df.drop(columns=word, inplace=True)
            yield i

    def continue_plotting(self):
        try:
            next(self.plotting_it)
        except StopIteration:
            self.continue_button.setEnabled(False)
            return
        else:
            self.startTimer(500)

    def end_plotting(self):
        if self.analysis_type.currentText() == 'Одиночное':
            self.canvas.twin_ax.plot(range(1, len(self.counts)+1), self.counts, c='red', label='Счётчик для слова')
        if self.analysis_type.currentText() == 'С накоплением':
            self.canvas.twin_ax.plot(range(len(self.counts)), self.counts,
                                     c='red', label='Количество слов после удаления')

        self.canvas.ax.plot(range(len(self.means)), self.means, c='green',
                            label=f'{self.classification_metric.currentText()}: среднее валидационное')
        self.canvas.twin_ax.format_coord = self.make_format(self.canvas.twin_ax, self.canvas.ax)
        self.canvas.figure.legend(loc='upper left')
        self.canvas.draw()

        self.classification_method.setEnabled(True)
        self.analysis_type.setEnabled(True)
        self.classification_metric.setEnabled(True)
        self.kfold_split_num.setEnabled(True)
        self.num_of_words_to_delete.setEnabled(True)
        self.best_split_only.setEnabled(True)

        self.stacked_buttons.setCurrentIndex(0)
        self.means = list()
        self.counts = list()


class VisualizationTab(QGridLayout):

    def __init__(self):
        super().__init__()
        self.setColumnMinimumWidth(0, 300)
        self.setColumnMinimumWidth(1, 300)
        # self.setRowMinimumHeight(0, 300)
        # self.setRowMinimumHeight(1, 300)

        self.reduced_data_window = None
        self.wordcloud_window = None
        self.correlation_window = None
        self.warning = None
        self.selection_matrix = None
        self.dataframe_table = None
        self.full_article_window = None

        # Меню облака слов
        set_choice_layout = QHBoxLayout()
        set_choice_layout.setSpacing(0)
        self.set_choice = QComboBox()
        self.set_choice.addItems(['для каждого класса'])  # TODO допилить ещё множества для облака слов
        set_choice_layout.addWidget(QLabel('Множество:'))
        set_choice_layout.addWidget(self.set_choice)

        self.num_of_words = QLineEdit()
        self.num_of_words.setPlaceholderText('Количество слов')
        self.num_of_words.setValidator(QIntValidator(10, 300))
        self.num_of_words.setFixedWidth(206)
        self.num_of_words.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.wordcloud_button = QPushButton('Показать')
        self.wordcloud_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        wordcloud_widget_b = QWidget()
        wordcloud_layout_b = QVBoxLayout()
        wordcloud_layout_b.setAlignment(Qt.AlignHCenter)
        wordcloud_layout_b.setSpacing(10)
        wordcloud_layout_b.addLayout(set_choice_layout)
        wordcloud_layout_b.addWidget(self.num_of_words)
        wordcloud_layout_b.addWidget(self.wordcloud_button)  # , alignment=Qt.AlignLeft
        wordcloud_widget_b.setLayout(wordcloud_layout_b)

        wordcloud_layout = QVBoxLayout()
        wordcloud_layout.setSizeConstraint(QVBoxLayout.SizeConstraint.SetMinAndMaxSize)
        wordcloud_layout.setAlignment(Qt.AlignCenter)
        wordcloud_layout.setSpacing(10)
        wordcloud_label = QLabel('Облако слов')
        wordcloud_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        wordcloud_layout.addWidget(wordcloud_label, alignment=(Qt.AlignHCenter | Qt.AlignBottom))
        wordcloud_layout.addWidget(wordcloud_widget_b, alignment=Qt.AlignTop)

        # Меню выизуализации матрицы с сжатым признаковым пространством
        method_layout = QHBoxLayout()
        method_layout.setAlignment(Qt.AlignLeft)
        method_layout.setSpacing(10)
        method_layout.setSizeConstraint(QHBoxLayout.SizeConstraint.SetMinAndMaxSize)
        method_layout.addWidget(QLabel('Метод снижения:'))
        self.method = QComboBox()
        self.method.addItems(['PCA', 't-SNE', 'UMAP'])
        self.method.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        method_layout.addWidget(self.method)

        n_dimensions_layout = QHBoxLayout()
        n_dimensions_layout.setAlignment(Qt.AlignLeft)
        n_dimensions_layout.setSpacing(5)
        n_dimensions_layout.setSizeConstraint(QHBoxLayout.SizeConstraint.SetMinAndMaxSize)
        n_dimensions_layout.addWidget(QLabel('Количество признаков:'))
        self.n_dims = QComboBox()
        self.n_dims.addItems(['2', '3'])
        self.n_dims.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        n_dimensions_layout.addWidget(self.n_dims)

        self.clusterization_checkbox = QCheckBox('кластеризация')
        self.reduce_data_button = QPushButton('Показать')
        self.reduce_data_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        reduce_data_widget_b = QWidget()
        reduce_data_layout_b = QVBoxLayout()
        reduce_data_layout_b.setAlignment(Qt.AlignHCenter)
        reduce_data_layout_b.setSpacing(10)
        reduce_data_layout_b.addLayout(method_layout)
        reduce_data_layout_b.addLayout(n_dimensions_layout)
        reduce_data_layout_b.addWidget(self.clusterization_checkbox, alignment=Qt.AlignLeft)
        reduce_data_layout_b.addWidget(self.reduce_data_button, alignment=Qt.AlignLeft)
        reduce_data_widget_b.setLayout(reduce_data_layout_b)

        reduce_data_layout = QVBoxLayout()
        reduce_data_layout.setAlignment(Qt.AlignTop)
        reduce_data_layout.setSpacing(10)
        reduce_data_layout.addWidget(QLabel('Снижение пространства признаков'),
                                     alignment=(Qt.AlignHCenter | Qt.AlignBottom))
        reduce_data_layout.addWidget(reduce_data_widget_b, alignment=Qt.AlignTop)

        # Таблица полученного фрейма данных
        # df_table_layout = QVBoxLayout()
        # df_table_layout.setAlignment(Qt.AlignTop)
        # df_table_layout.setSpacing(10)
        # df_table_layout.addWidget(QLabel('Отображение взвешенной матрицы'),
        #                           alignment=(Qt.AlignHCenter | Qt.AlignBottom))
        # self.show_df_button = QPushButton('Показать')
        # self.show_df_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        # df_table_layout.addWidget(self.show_df_button, alignment=(Qt.AlignHCenter | Qt.AlignTop))

        # Меню визуализации корелляции классов
        # correlation_layout = QVBoxLayout()
        # correlation_layout.setSpacing(20)
        # correlation_layout.setAlignment(Qt.AlignHCenter)
        # correlation_layout.addWidget(QLabel('Диаграмма корреляции центроидов классов'),
        #                              alignment=(Qt.AlignHCenter | Qt.AlignBottom))
        # self.correlate_data_button = QPushButton('Показать')
        # self.correlate_data_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        # correlation_layout.addWidget(self.correlate_data_button, alignment=(Qt.AlignHCenter | Qt.AlignTop))

        self.addLayout(wordcloud_layout, 0, 0, alignment=Qt.AlignCenter)
        self.addLayout(reduce_data_layout, 0, 1, alignment=Qt.AlignCenter)
        # self.addLayout(df_table_layout, 1, 0, 1, 2)
        # self.addLayout(correlation_layout, 1, 0, 1, 2)

    @Slot(object)
    def reduce_data(self, selection_matrix: SelectionMatrix):
        if selection_matrix is None:
            self.warning = QMessageBox(QMessageBox.Icon.Critical, 'Ошибка', 'Не задана выборка')
            self.warning.show()
            return

        win_title = f'{selection_matrix.selection_name}_{selection_matrix.weight_method}_{selection_matrix.class_size}'
        reduced_data = reduce_data_features_class(selection_matrix, self.method.currentText(),
                                                  int(self.n_dims.currentText()))
        # classes_names = correct_queries_for_output(selection_matrix.classes_names)
        classes_names = selection_matrix.classes_names
        self.reduced_data_window = ReducedDataWindow(reduced_data, classes_names, selection_matrix.db_name)
        self.reduced_data_window.setMinimumSize(640, 480)
        self.reduced_data_window.setWindowTitle(win_title)
        self.reduced_data_window.setAttribute(Qt.WA_DeleteOnClose, True)
        self.reduced_data_window.setWindowFlags(Qt.MSWindowsOwnDC)
        self.reduced_data_window.setWindowState(Qt.WindowState.WindowMaximized)
        self.reduced_data_window.show()
        self.reduced_data_window.setSizes([1680, 240])

    @Slot(object)
    def wordcloud_data(self, selection_matrix: SelectionMatrix):
        if not self.num_of_words.text():
            self.warning = QMessageBox(QMessageBox.Icon.Critical, 'Ошибка', 'Введите количество отображемых слов')
            self.warning.show()
            return
        if self.num_of_words.validator().validate(self.num_of_words.text(), 0)[0] != QIntValidator.Acceptable:
            self.warning = QMessageBox(QMessageBox.Icon.Critical, 'Ошибка',
                                       'Некорректное число отображемых слов(10-300)')
            self.warning.show()
            return

        tab = QTabWidget()
        frequencies_list = list()
        words_count = int(self.num_of_words.text())
        for i, df in enumerate(selection_matrix.split_dataframe_by_classes(), 1):
            canvas = FigureCanvas(Figure((5, 3)))
            toolbar = NavigationToolbar(canvas, tab, coordinates=False)
            ax = canvas.figure.add_subplot()
            frequencies = df.sum(0).sort_values(ascending=False)[:words_count]
            frequencies_list.append(frequencies)
            make_word_cloud_plot(frequencies, ax)
            layout = QVBoxLayout()
            layout.setSizeConstraint(QVBoxLayout.SizeConstraint.SetMinAndMaxSize)
            layout.addWidget(toolbar)
            layout.addWidget(canvas)
            widget = QWidget()
            widget.setLayout(layout)
            tab.addTab(widget, f"Класс {i}")

        canvas = FigureCanvas(Figure((5, 3)))
        toolbar = NavigationToolbar(canvas, tab, coordinates=False)
        ax = canvas.figure.add_subplot()
        make_word_cloud_plot(selection_matrix.df.sum(0), ax, max_words=words_count)
        layout = QVBoxLayout()
        layout.setSizeConstraint(QVBoxLayout.SizeConstraint.SetMinAndMaxSize)
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        widget = QWidget()
        widget.setLayout(layout)
        tab.addTab(widget, "Все классы")

        jaccard_matrix = calculate_jaccard_matrix(frequencies_list)
        table_shape = jaccard_matrix.shape
        table = QTableWidget(table_shape[0], table_shape[1])
        table.setHorizontalHeaderLabels([f'Класс {i+1}' for i in range(table_shape[0])])
        table.setVerticalHeaderLabels([f'Класс {i+1}' for i in range(table_shape[1])])
        for i, row in enumerate(jaccard_matrix):
            for j, el in enumerate(row):
                table_cell = QTableWidgetItem(str(round(el, 4)))
                table_cell.setFlags(Qt.ItemFlag.ItemIsEnabled)
                table_cell.setTextAlignment(Qt.AlignCenter)
                table.setItem(i, j, table_cell)
        tab.addTab(table, 'Матрица коэффициентов Жаккара')

        weighted_jaccard_matrix = calculate_weighted_jaccard_matrix(frequencies_list)
        table_shape = weighted_jaccard_matrix.shape
        table = QTableWidget(table_shape[0], table_shape[1])
        table.setHorizontalHeaderLabels([f'Класс {i + 1}' for i in range(table_shape[0])])
        table.setVerticalHeaderLabels([f'Класс {i + 1}' for i in range(table_shape[1])])
        for i, row in enumerate(weighted_jaccard_matrix):
            for j, el in enumerate(row):
                table_cell = QTableWidgetItem(str(round(el, 4)))
                table_cell.setFlags(Qt.ItemFlag.ItemIsEnabled)
                table_cell.setTextAlignment(Qt.AlignCenter)
                table.setItem(i, j, table_cell)
        tab.addTab(table, 'Матрица взвешенных коэффициентов Жаккара')

        scroll = make_selection_scroll(selection_matrix.classes_names)
        
        self.wordcloud_window = make_default_window(selection_matrix)
        self.wordcloud_window.addWidget(scroll)
        self.wordcloud_window.addWidget(tab)
        self.wordcloud_window.show()

    # @Slot(object)
    # def correlate_data(self, selection_matrix: SelectionMatrix):
    #
    #     self.correlation_window = make_default_window(selection_matrix)
    #
    #     canvas = FigureCanvas(Figure((5, 3)))
    #     toolbar = NavigationToolbar(canvas, self.correlation_window, coordinates=False)
    #     ax = canvas.figure.add_subplot()
    #     im = make_correlation_matrix_plot(selection_matrix, ax)
    #     canvas.figure.colorbar(im, ax=ax)
    #     ax.set_anchor('S', share=True)
    #
    #     scroll = make_selection_scroll(selection_matrix.classes_names)
    #
    #     layout = QVBoxLayout()
    #     layout.addWidget(toolbar)
    #     layout.addWidget(canvas)
    #
    #     figure_widget = QWidget()
    #     figure_widget.setLayout(layout)
    #
    #     self.correlation_window.addWidget(scroll)
    #     self.correlation_window.addWidget(figure_widget)
    #     self.correlation_window.show()

    @Slot(object)
    def cluster_data_with_reduce(self, selection_matrix: SelectionMatrix):

        n_components = int(self.n_dims.currentText())
        clusterization_data = clusterization(selection_matrix)
        cluster_indexes = sorted(clusterization_data['cluster indexes'].items(), key=lambda x: x[0])

        reduced_cluster_data = reduce_data_features_cluster(selection_matrix,
                                                            cluster_indexes,
                                                            clusterization_data['cluster centers'],
                                                            self.method.currentText(),
                                                            n_components
                                                            )

        # График классов
        # classes_names = correct_queries_for_output(selection_matrix.classes_names)
        classes_names = selection_matrix.classes_names
        reduced_classes = reduce_data_features_class(selection_matrix, self.method.currentText(), n_components)
        classes_tab = ReducedDataWindow(reduced_classes,
                                        [f'Класс {i}: {class_name}' for i, class_name in enumerate(classes_names, 1)],
                                        selection_matrix.db_name,
                                        reduced_cluster_data['centroids']
                                        )
        # classes_tab.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        classes_tab.setSizes([1680, 240])

        # График кластеров
        cluster_tab = ReducedDataWindow(reduced_cluster_data['dots for each cluster'],
                                        [f'Кластер {i+1}' for i in map(lambda x: x[0], cluster_indexes)],
                                        selection_matrix.db_name,
                                        reduced_cluster_data['centroids']
                                        )
        # cluster_tab.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        cluster_tab.setSizes([1680, 240])

        classes_tab.size_changed.connect(cluster_tab.change_dots_size)
        classes_tab.size_changed.connect(cluster_tab.dots_radius.setSliderPosition)
        cluster_tab.size_changed.connect(classes_tab.change_dots_size)
        cluster_tab.size_changed.connect(classes_tab.dots_radius.setSliderPosition)

        # Таблица кластеризации
        table_data = clusterization_data['table data']
        table_size = len(table_data)
        cluster_info = QTableWidget(table_size - 1, table_size)
        cluster_info.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        # cluster_info.setSizeAdjustPolicy(QTableWidget.AdjustToContents)
        cluster_info.setHorizontalHeaderLabels(table_data.keys())
        cluster_info.setVerticalHeaderLabels([f'Кластер {i}' for i in range(1, table_size)])
        bold_font = QFont(cluster_info.horizontalHeaderItem(table_size - 1).font())
        bold_font.setBold(True)
        cluster_info.horizontalHeaderItem(table_size - 1).setFont(bold_font)
        for j, (_, count_list) in enumerate(table_data.items()):
            for i, count in enumerate(count_list):
                table_cell = QTableWidgetItem(count)
                table_cell.setFlags(Qt.ItemFlag.ItemIsEnabled)
                table_cell.setTextAlignment(Qt.AlignCenter)
                if j == len(table_data) - 1:
                    table_cell.setFont(bold_font)
                cluster_info.setItem(i, j, table_cell)

        # Таблица топа слов центроидов
        words_info = QTableWidget(len(selection_matrix.classes_names), 10)
        words_info.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        # words_info.setSizeAdjustPolicy(QTableWidget.AdjustToContents)
        words_info.setVerticalHeaderLabels([f'Кластер {i}' for i in range(1, len(selection_matrix.classes_names)+1)])
        for i, center in enumerate(clusterization_data['cluster centers']):
            sorted_words_and_values = sorted(zip(selection_matrix.df.columns, center), key=lambda x: x[1], reverse=True)
            center_words = list(map(lambda y: y[0], sorted_words_and_values[:10]))
            for j, word in enumerate(center_words):
                table_cell = QTableWidgetItem(word)
                table_cell.setFlags(Qt.ItemFlag.ItemIsEnabled)
                table_cell.setTextAlignment(Qt.AlignCenter)
                words_info.setItem(i, j, table_cell)

        tab_tables = QTabWidget()
        tab_tables.addTab(cluster_info, 'Населённость кластеров')
        tab_tables.addTab(words_info, 'Наиболее часто встречающиеся слова')

        tab_figs = QTabWidget()
        tab_figs.addTab(classes_tab, 'Исходные классы')
        tab_figs.addTab(cluster_tab, 'Результат кластеризации')

        self.reduced_data_window = make_default_window(selection_matrix)
        self.reduced_data_window.addWidget(tab_tables)
        self.reduced_data_window.addWidget(tab_figs)
        self.reduced_data_window.show()

    # def show_df(self, selection_matrix: SelectionMatrix):
    #     df = selection_matrix.df
    #     rows_count, cols_count = df.shape
    #     urls = df.index
    #     words = df.columns
    #     db_name = selection_matrix.db_name
    #
    #     self.dataframe_table = QTableWidget(rows_count, cols_count)
    #     self.dataframe_table.setMinimumSize(640, 480)
    #     win_title = f'{selection_matrix.selection_name}_{selection_matrix.weight_method}_{selection_matrix.class_size}'
    #     self.dataframe_table.setAttribute(Qt.WA_DeleteOnClose, True)
    #     self.dataframe_table.setWindowTitle(win_title)
    #     self.dataframe_table.setWindowFlags(Qt.MSWindowsOwnDC)
    #     self.dataframe_table.setWindowState(Qt.WindowState.WindowMaximized)
    #     self.dataframe_table.setSizeAdjustPolicy(QTableWidget.SizeAdjustPolicy.AdjustToContents)
    #
    #     self.dataframe_table.setHorizontalHeaderLabels(words)
    #     self.dataframe_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    #     self.dataframe_table.horizontalHeader().setSectionsMovable(True)
    #     self.dataframe_table.setVerticalHeaderLabels(urls)
    #     # self.dataframe_table.verticalHeader().setHidden(True)
    #     self.dataframe_table.verticalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
    #     self.dataframe_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    #     self.dataframe_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
    #     self.dataframe_table.cellDoubleClicked.connect(self.show_full_article)
    #
    #     for i, url in enumerate(urls):
    #         for j, word in enumerate(words):
    #             table_item = QTableNumberItem(str(round(df.iloc[i, j], 5)))
    #             table_item.setFlags((Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable))
    #             table_item.setTextAlignment(Qt.AlignCenter)
    #             self.dataframe_table.setItem(i, j, table_item)
    #
    #     self.dataframe_table.setSortingEnabled(True)
    #     self.dataframe_table.show()
    #
    # def show_full_article(self):
    #     fields_for_output = ['url', 'tags_codes', 'tags_names', 'authors', 'title', 'abstract', 'submitted']
    #     url = self.sender().selectedItems()[0].text()
    #     article = Database(f'data/{self.db_name}', []).find_article_by_url(url, fields_for_output)
    #     if article is None:
    #         error_text = f'В {self.db_name} не найдено статьи с адресом {url}'
    #         self.warning = QMessageBox(QMessageBox.Icon.Critical, 'Ошибка', error_text)
    #         self.warning.show()
    #         return
    #
    #     self.full_article_window = FullArticle(article, fields_for_output)
    #     self.full_article_window.setWindowTitle(url)
    #     self.full_article_window.show()


class ClassificationTab(QGridLayout):
    start_frequency_analysis_button = Signal(QPushButton)

    def __init__(self):
        super().__init__()

        self.warning = None
        self.classification_report_window = None
        self.frequency_analysis_window = None
        self.sizes = None

        classification_method_layout = QHBoxLayout()
        classification_method_layout.setSizeConstraint(QHBoxLayout.SizeConstraint.SetMinAndMaxSize)
        classification_method_layout.setAlignment(Qt.AlignHCenter)
        classification_method_layout.setSpacing(10)
        classification_method_layout.addWidget(QLabel('Метод:'))
        self.classification_method = QComboBox()
        self.classification_method.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.classification_method.addItems(CLASSIFICATION_METHODS.keys())
        classification_method_layout.addWidget(self.classification_method)

        self.kfold_split_num = QLineEdit()
        self.kfold_split_num.setFixedWidth(215)
        validator = QIntValidator(2, 10)
        self.kfold_split_num.setValidator(validator)
        self.kfold_split_num.setPlaceholderText('Количество разбиений K-Fold')

        self.best_split_only = QCheckBox('Только лучшая тестовая модель')
        self.best_split_only.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        # self.frequency_analysis = QCheckBox('Анализ высокочастотных слов')
        # self.frequency_analysis.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.classification_button = QPushButton('Результат')
        self.classification_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        classification_widget_b = QWidget()
        classification_layout_b = QVBoxLayout()
        classification_layout_b.setAlignment(Qt.AlignHCenter)
        classification_layout_b.addLayout(classification_method_layout)
        classification_layout_b.addWidget(self.kfold_split_num)
        classification_layout_b.addWidget(self.best_split_only)
        classification_layout_b.addWidget(self.classification_button, alignment=Qt.AlignLeft)
        classification_widget_b.setLayout(classification_layout_b)

        classification_layout = QVBoxLayout()
        classification_layout.setSpacing(10)
        classification_layout.setAlignment(Qt.AlignHCenter)
        classification_layout.setSizeConstraint(QVBoxLayout.SizeConstraint.SetMinAndMaxSize)
        classification_layout.addWidget(QLabel('Классификация'), alignment=(Qt.AlignHCenter | Qt.AlignBottom))
        classification_layout.addWidget(classification_widget_b, alignment=(Qt.AlignHCenter | Qt.AlignTop))

        frequency_analysis_layout = QVBoxLayout()
        frequency_analysis_layout.setSpacing(10)
        frequency_analysis_layout.setAlignment(Qt.AlignHCenter)
        frequency_analysis_layout.addWidget(QLabel('Анализ и удаление высокочастотных слов'),
                                            alignment=(Qt.AlignHCenter | Qt.AlignBottom))
        self.frequency_analysis_button = QPushButton('Окно для анализа')
        self.frequency_analysis_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        frequency_analysis_layout.addWidget(self.frequency_analysis_button, alignment=(Qt.AlignHCenter | Qt.AlignTop))

        self.addLayout(classification_layout, 1, 1)
        self.addLayout(frequency_analysis_layout, 2, 1)

    @Slot(object)
    def classification_report(self, matrix: SelectionMatrix):
        if not self.kfold_split_num.text():
            self.warning = QMessageBox(QMessageBox.Icon.Critical, 'Ошибка', 'Введите количество разбиений')
            self.warning.show()
            return
        if self.kfold_split_num.validator().validate(self.kfold_split_num.text(), 0)[0] != QIntValidator.Acceptable:
            self.warning = QMessageBox(QMessageBox.Icon.Critical, 'Ошибка', 'Некорректное число разбиений (1-10)')
            self.warning.show()
            return

        reports = classification_report_for_table(matrix, self.classification_method.currentText(),
                                                  int(self.kfold_split_num.text()), self.best_split_only.isChecked())

        self.classification_report_window = ClassificationReport(reports,
                                                                 matrix.classes_names,
                                                                 self.classification_method.currentText(),
                                                                 self.kfold_split_num.text()
                                                                 )
        win_title = f'{matrix.selection_name}_{matrix.weight_method}_{matrix.class_size}'
        self.classification_report_window.setAttribute(Qt.WA_DeleteOnClose, True)
        self.classification_report_window.setWindowTitle(win_title)
        # self.classification_report_window.setWindowFlags((Qt.WindowCloseButtonHint | Qt.WindowStaysOnTopHint))
        self.classification_report_window.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.MinimumExpanding)
        self.classification_report_window.show()

    @Slot(str)
    def classification_frequency_analysis(self, win_title: str):
        self.frequency_analysis_window = FrequencyAnalysis(win_title)
        self.start_frequency_analysis_button.emit(self.frequency_analysis_window.start_button)
        self.frequency_analysis_window.show()


class SelectionTab(QVBoxLayout):

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignHCenter)
        self.setSpacing(25)
        self.setSizeConstraint(QVBoxLayout.SizeConstraint.SetMinAndMaxSize)

        self.warning = None
        self.selection_info_window = None
        self.selection_articles_window = None

        change_name_layout = QVBoxLayout()
        change_name_layout.setAlignment((Qt.AlignTop | Qt.AlignHCenter))
        change_name_layout.setSpacing(5)
        change_name_layout.addWidget(QLabel('Сменить имя выборки'), alignment=Qt.AlignHCenter)
        self.name_input = QLineEdit()
        self.name_input.setFixedWidth(220)
        self.change_name_button = QPushButton('Изменить')
        change_name_layout.addWidget(self.name_input, alignment=Qt.AlignHCenter)
        change_name_layout.addWidget(self.change_name_button, alignment=Qt.AlignHCenter)

        delete_words_layout = QVBoxLayout()
        delete_words_layout.setSpacing(5)
        delete_words_layout.setAlignment((Qt.AlignTop | Qt.AlignHCenter))
        delete_words_layout.addWidget(QLabel('Удалить слова из матрицы'), alignment=Qt.AlignHCenter)
        words_to_delete_tooltip = """Правила для удаления слов:
        
        можно удалить срез слов (int:int);
        отдельные слова (int, ..., int или str, ..., str)
        """
        self.words_to_delete_input = QLineEdit()
        self.words_to_delete_input.setToolTip(words_to_delete_tooltip)
        self.words_to_delete_input.setFixedWidth(220)
        self.delete_words_button = QPushButton('Удалить')
        # delete_words_button.clicked.connect()
        delete_words_layout.addWidget(self.words_to_delete_input, alignment=Qt.AlignHCenter)
        delete_words_layout.addWidget(self.delete_words_button, alignment=Qt.AlignHCenter)

        show_selection_info_layout = QVBoxLayout()
        show_selection_info_layout.setAlignment((Qt.AlignTop | Qt.AlignHCenter))
        show_selection_info_layout.setSpacing(5)
        show_selection_info_layout.addWidget(QLabel('Основная информация'), alignment=Qt.AlignHCenter)
        self.show_selection_info_button = QPushButton('Показать')
        # show_selection_info_button.clicked.connect()
        show_selection_info_layout.addWidget(self.show_selection_info_button, alignment=Qt.AlignHCenter)

        # show_stopwords_layout = QVBoxLayout()
        # show_stopwords_layout.setAlignment((Qt.AlignTop | Qt.AlignHCenter))
        # show_stopwords_layout.setSpacing(5)
        # show_stopwords_layout.addWidget(QLabel('Показать стоп-слова'), alignment=Qt.AlignHCenter)
        # self.show_stopwords_button = QPushButton('Показать')
        # # show_stopwords_button.clicked.connect()
        # show_stopwords_layout.addWidget(self.show_stopwords_button, alignment=Qt.AlignHCenter)
        # 
        # show_classes_names_layout = QVBoxLayout()
        # show_classes_names_layout.setAlignment((Qt.AlignTop | Qt.AlignHCenter))
        # show_classes_names_layout.setSpacing(5)
        # show_classes_names_layout.addWidget(QLabel('Показать имена классов'), alignment=Qt.AlignHCenter)
        # self.show_classes_names_button = QPushButton('Показать')
        # # show_classes_names_button.clicked.connect()
        # show_classes_names_layout.addWidget(self.show_classes_names_button, alignment=Qt.AlignHCenter)
        # 
        # show_most_common_words_layout = QVBoxLayout()
        # show_most_common_words_layout.setAlignment((Qt.AlignTop | Qt.AlignHCenter))
        # show_most_common_words_layout.setSpacing(5)
        # # show_most_common_words_layout.setSizeConstraint(QVBoxLayout.SizeConstraint.SetMinAndMaxSize)
        # show_most_common_words_layout.addWidget(QLabel('Показать популярные слова для выборки'),
        #                                         alignment=Qt.AlignHCenter)
        # self.show_most_common_words_button = QPushButton('Показать')
        # # show_most_common_words_button.clicked.connect()
        # show_most_common_words_layout.addWidget(self.show_most_common_words_button, alignment=Qt.AlignHCenter)

        show_selection_articles_layout = QVBoxLayout()
        show_selection_articles_layout.setAlignment((Qt.AlignTop | Qt.AlignHCenter))
        show_selection_articles_layout.setSpacing(5)
        show_selection_articles_layout.setSizeConstraint(QVBoxLayout.SizeConstraint.SetMinAndMaxSize)
        show_selection_articles_layout.addWidget(QLabel('Показать статьи из выборки'), alignment=Qt.AlignHCenter)
        self.show_selection_articles_button = QPushButton('Показать')
        # show_selection_articles_button.clicked.connect()
        show_selection_articles_layout.addWidget(self.show_selection_articles_button, alignment=Qt.AlignHCenter)

        self.addLayout(change_name_layout)
        self.addLayout(delete_words_layout)
        self.addLayout(show_selection_info_layout)
        # self.addLayout(show_stopwords_layout)
        # self.addLayout(show_classes_names_layout)
        # self.addLayout(show_most_common_words_layout)
        self.addLayout(show_selection_articles_layout)

    def show_selection_info(self, matrix: SelectionMatrix):
        classes_names = QLabel('\n'.join([f'Класс {i}: {class_name}'
                                          for i, class_name in enumerate(matrix.classes_names, 1)]))
        classes_names.setAlignment(Qt.AlignTop)
        classes_names.setTextInteractionFlags(Qt.TextSelectableByMouse)
        classes_names_tab = QScrollArea()
        classes_names_tab.setWidget(classes_names)

        stopwords_lines = []
        row = []
        row_width = 50
        for word in matrix.stop_words:
            if len(', '.join(row)) + len(word) + 2 > row_width:
                stopwords_lines.append(', '.join(row))
                row.clear()
            else:
                row.append(word)
        stopwords = QLabel(',\n'.join(stopwords_lines))
        stopwords.setTextInteractionFlags(Qt.TextSelectableByMouse)
        stopwords_tab = QScrollArea()
        stopwords_tab.setWidget(stopwords)

        most_common_words = ';\n'.join([f'{i}: {word}, {count} слов'
                                        for i, (word, count) in enumerate(matrix.most_common_words, 1)])
        most_common_words_tab = QScrollArea()
        most_common_words_tab.setWidget(QLabel(most_common_words))

        self.selection_info_window = QTabWidget()
        self.selection_info_window.setMinimumSize(320, 280)
        self.selection_info_window.addTab(classes_names_tab, 'Названия классов')
        self.selection_info_window.addTab(stopwords_tab, 'Стоп-слова')
        self.selection_info_window.addTab(most_common_words_tab, 'Топ слов')
        self.selection_info_window.show()

    def show_selection_articles(self, matrix: SelectionMatrix):
        all_urls = list(matrix.df.index)
        classes_names = matrix.classes_names.copy()
        class_size = matrix.class_size
        fields_for_output = ['title', 'tags_names', 'submitted']

        self.selection_articles_window = QTableWidget(len(all_urls), len(fields_for_output) + 1)
        self.selection_articles_window.setHorizontalHeaderLabels(['Класс'] + [TABLE_FIELDS_TRANSLATED[f]
                                                                              for f in fields_for_output])
        self.selection_articles_window.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.selection_articles_window.horizontalHeader().setStretchLastSection(True)
        self.selection_articles_window.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.selection_articles_window.setVerticalHeaderLabels(all_urls)
        self.selection_articles_window.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)

        db = Database(f'data/{matrix.db_name}', [])
        for i, url in enumerate(all_urls):
            article = db.find_article_by_url(url, fields_for_output)
            if article:
                class_name = classes_names[int(i / class_size)]
                table_cell = QTableWidgetItem(class_name)
                table_cell.setFlags(Qt.ItemFlag.ItemIsEnabled)
                table_cell.setTextAlignment(Qt.AlignCenter)
                self.selection_articles_window.setItem(i, 0, table_cell)

                for j, field in enumerate(article, 1):
                    text = ''
                    row_width = 90
                    row_text = ''
                    for word in field.split():
                        if len(row_text) + len(word) + 1 > row_width:
                            text += '\n'
                            row_text = ''
                        row_text += word + ' '
                        text += word + ' '

                    table_cell = QTableWidgetItem(text)
                    table_cell.setFlags(Qt.ItemFlag.ItemIsEnabled)
                    table_cell.setTextAlignment(Qt.AlignCenter)
                    self.selection_articles_window.setItem(i, j, table_cell)
        self.selection_articles_window.show()


class AnalysisMenu(QVBoxLayout):
    current_matrix = Signal(object)

    def __init__(self):
        super().__init__()
        self.setSpacing(5)
        self.warning = None
        self.selection_matrix = None

        matrix_from_file_layout = QHBoxLayout()
        matrix_from_file_layout.setAlignment((Qt.AlignTop | Qt.AlignLeft))
        matrix_from_file_layout.setSizeConstraint(QHBoxLayout.SizeConstraint.SetMinAndMaxSize)
        self.selection_matrix_combo = QComboBox()
        self.selection_matrix_combo.addItems(find_dat_files())
        self.selection_matrix_combo.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.selection_matrix_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.apply_matrix_button = QPushButton('Загрузить матрицу')
        self.apply_matrix_button.clicked.connect(self.get_matrix_from_file)
        self.apply_matrix_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        matrix_from_file_layout.addWidget(self.selection_matrix_combo)
        matrix_from_file_layout.addWidget(self.apply_matrix_button)

        self.selection_info = QHBoxLayout()
        self.selection_info.setSizeConstraint(QHBoxLayout.SizeConstraint.SetMinAndMaxSize)
        self.selection_info.setSpacing(20)
        selection_name = QLabel('Название:')
        selection_name.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)
        selection_name.setMaximumWidth(200)
        self.selection_info.addWidget(selection_name)
        self.selection_info.addWidget(QLabel('Классов: 0'))
        self.selection_info.addWidget(QLabel('Статей на класс: 0'))
        self.selection_info.addWidget(QLabel('Метод взвешивания:'))
        selection_info_widget = QWidget()
        selection_info_widget.setLayout(self.selection_info)

        self.more_selection_info = QHBoxLayout()
        self.more_selection_info.setSizeConstraint(QHBoxLayout.SizeConstraint.SetMinAndMaxSize)
        self.more_selection_info.setSpacing(20)
        self.more_selection_info.addWidget(QLabel('Размерность матрицы:'))
        self.more_selection_info.addWidget(QLabel('Столбцов удалено:'))
        self.more_selection_info.addWidget(QLabel('Всего слов: 0'))
        more_selection_info_widget = QWidget()
        more_selection_info_widget.setLayout(self.more_selection_info)

        save_matrix_button = QPushButton('Сохранить')
        delete_current_matrix_button = QPushButton('Удалить')

        current_matrix_buttons_layout = QHBoxLayout()
        current_matrix_buttons_layout.setAlignment(Qt.AlignHCenter)
        current_matrix_buttons_layout.setSpacing(15)
        current_matrix_buttons_layout.addWidget(save_matrix_button)
        current_matrix_buttons_layout.addWidget(delete_current_matrix_button)
        current_matrix_buttons_widget = QWidget()
        current_matrix_buttons_widget.setLayout(current_matrix_buttons_layout)

        self.visualization_tab_layout = VisualizationTab()
        visualization_tab = QWidget()
        visualization_tab.setLayout(self.visualization_tab_layout)

        self.classification_tab_layout = ClassificationTab()
        classification_tab = QWidget()
        classification_tab.setLayout(self.classification_tab_layout)

        self.selection_tab_layout = SelectionTab()
        selection_tab = QScrollArea()
        selection_tab.setAlignment(Qt.AlignHCenter)
        # selection_tab.setWidgetResizable(True)
        # selection_tab.setSizeAdjustPolicy(QScrollArea.SizeAdjustPolicy.AdjustToContents)
        selection_tab_widget = QWidget()
        selection_tab_widget.setLayout(self.selection_tab_layout)
        selection_tab.setWidget(selection_tab_widget)
        # selection_tab.setLayout(self.selection_tab_layout)

        # СИГНАЛЫ

        self.visualization_tab_layout.reduce_data_button.clicked.connect(self.get_current_matrix)
        self.visualization_tab_layout.wordcloud_button.clicked.connect(self.get_current_matrix)
        # self.visualization_tab_layout.show_df_button.clicked.connect(self.get_current_matrix)
        # self.visualization_tab_layout.correlate_data_button.clicked.connect(self.get_current_matrix)

        self.classification_tab_layout.classification_button.clicked.connect(self.get_current_matrix)
        self.classification_tab_layout.frequency_analysis_button.clicked.connect(self.get_current_matrix)
        self.classification_tab_layout.start_frequency_analysis_button.connect(self.connect_button_signal_with_matrix)
        # self.classification_tab_layout.frequency_analysis_window.start_button.clicked.connect(self.get_current_matrix)

        self.selection_tab_layout.change_name_button.clicked.connect(self.change_selection_name)
        self.selection_tab_layout.show_selection_articles_button.clicked.connect(self.get_current_matrix)
        self.selection_tab_layout.show_selection_info_button.clicked.connect(self.get_current_matrix)
        # self.selection_tab_layout.delete_words_button.clicked.connect(self.get_current_matrix)
        save_matrix_button.clicked.connect(self.save_current_matrix)
        delete_current_matrix_button.clicked.connect(self.delete_current_matrix)

        self.tab = QTabWidget()
        self.tab.addTab(visualization_tab, 'Визуализация')
        self.tab.addTab(classification_tab, 'Классификация')
        self.tab.addTab(selection_tab, 'Выборка')

        self.addLayout(matrix_from_file_layout)
        self.addWidget(selection_info_widget, alignment=(Qt.AlignTop | Qt.AlignHCenter))
        self.addWidget(more_selection_info_widget, alignment=(Qt.AlignTop | Qt.AlignHCenter))
        self.addWidget(current_matrix_buttons_widget, alignment=(Qt.AlignTop | Qt.AlignHCenter))
        self.addWidget(self.tab)

    def connect_button_signal_with_matrix(self, button):
        button.clicked.connect(self.get_current_matrix)

    def delete_current_matrix(self):
        # if self.selection_matrix is None:
        #     self.warning = QMessageBox(QMessageBox.Icon.Critical, 'Ошибка', f'Матрица не задана')
        #     self.warning.show()
        #     return
        #
        # name = self.selection_matrix.selection_name
        self.selection_matrix = None

        selection_info_list = self.selection_info.parentWidget().findChildren(QLabel)
        selection_info_list[0].setText('Название:')
        selection_info_list[1].setText('Классов: 0')
        selection_info_list[2].setText('Статей на класс: 0')
        selection_info_list[3].setText('Метод взвешивания:')

        more_selection_info_list = self.more_selection_info.parentWidget().findChildren(QLabel)
        more_selection_info_list[0].setText(f'Размерность матрицы:')
        more_selection_info_list[1].setText(f'Столбцов удалено:')
        more_selection_info_list[2].setText(f'Всего слов: 0')

        # self.warning = QMessageBox(QMessageBox.Icon.Information, 'Успех', f'Матрица {name} удалена')
        # self.warning.show()

    def save_current_matrix(self):
        if self.selection_matrix is None:
            self.warning = QMessageBox(QMessageBox.Icon.Critical, 'Ошибка', 'Матрица не задана')
            self.warning.show()
            return

        name = self.selection_matrix.to_pickle()
        if name not in [self.selection_matrix_combo.itemText(i) for i in range(self.selection_matrix_combo.count())]:
            self.selection_matrix_combo.addItem(name)
        self.warning = QMessageBox(QMessageBox.Icon.Information, 'Успех', f'Матрица сохранена под имением:\n{name}')
        self.warning.show()

    def get_matrix_from_file(self):
        # print(self.selection_matrix)
        if self.selection_matrix is None:
            self.delete_current_matrix()
        file_name = self.selection_matrix_combo.currentText()
        selection_matrix = SelectionMatrix('', *get_info_from_name_of_dat_file(file_name))
        selection_matrix.get_dataframe_from_file()
        self.set_selection_matrix(selection_matrix)

    @Slot(object)
    def set_selection_matrix(self, selection_matrix: SelectionMatrix):
        self.selection_matrix = selection_matrix
        # print(self.selection_matrix)
        self.set_selection_info()
        self.update()

    @Slot()
    def set_selection_info(self):
        selection_info_list = self.selection_info.parentWidget().findChildren(QLabel)
        selection_info_list[0].setText(f'Название: {self.selection_matrix.selection_name}')
        selection_info_list[1].setText(f'Классов: {len(self.selection_matrix.classes_names)}')
        selection_info_list[2].setText(f'Статей на класс: {self.selection_matrix.class_size}')
        selection_info_list[3].setText(f'Метод взвешивания: {self.selection_matrix.weight_method}')

        more_selection_info_list = self.more_selection_info.parentWidget().findChildren(QLabel)
        more_selection_info_list[0].setText(f'Размерность матрицы: {self.selection_matrix.df.shape}')
        more_selection_info_list[1].setText(f'Столбцов удалено: {len(self.selection_matrix.deleted_words)}')
        more_selection_info_list[2].setText(f'Всего слов: {self.selection_matrix.words_count}')

    def change_selection_name(self):
        if self.selection_matrix is None:
            self.warning = QMessageBox(QMessageBox.Icon.Critical, 'Ошибка', 'Матрица не задана')
            self.warning.show()
            return

        self.selection_info.parentWidget().findChildren(QLabel)[0].setText(f'Название: '
                                                                           f'{self.selection_matrix.selection_name}')
        self.selection_matrix.selection_name = self.selection_tab_layout.name_input.text()

    def get_current_matrix(self):
        if self.selection_matrix is None:
            self.warning = QMessageBox(QMessageBox.Icon.Critical, 'Ошибка', 'Матрица не задана')
            self.warning.show()
            return

        elif self.sender() is self.visualization_tab_layout.wordcloud_button:
            self.visualization_tab_layout.wordcloud_data(self.selection_matrix)

        elif self.sender() is self.visualization_tab_layout.reduce_data_button:
            if self.visualization_tab_layout.clusterization_checkbox.isChecked():
                self.visualization_tab_layout.cluster_data_with_reduce(self.selection_matrix)
            else:
                self.visualization_tab_layout.reduce_data(self.selection_matrix)

        # elif self.sender() is self.visualization_tab_layout.show_df_button:
        #     self.visualization_tab_layout.show_df(self.selection_matrix)

        # elif self.sender() is self.visualization_tab_layout.correlate_data_button:
        #     self.visualization_tab_layout.correlate_data(self.selection_matrix)

        elif self.sender() is self.selection_tab_layout.show_selection_info_button:
            self.selection_tab_layout.show_selection_info(self.selection_matrix)

        elif self.sender() is self.selection_tab_layout.show_selection_articles_button:
            self.selection_tab_layout.show_selection_articles(self.selection_matrix)

        elif self.sender() is self.classification_tab_layout.classification_button:
            self.classification_tab_layout.classification_report(self.selection_matrix)

        elif self.sender() is self.classification_tab_layout.frequency_analysis_button:
            win_title = f'{self.selection_matrix.selection_name}_{self.selection_matrix.weight_method}_' \
                        f'{self.selection_matrix.class_size}'
            self.classification_tab_layout.classification_frequency_analysis(win_title)

        elif self.sender() is self.classification_tab_layout.frequency_analysis_window.start_button:
            self.classification_tab_layout.frequency_analysis_window.start_plotting(self.selection_matrix)
