from PySide6.QtWidgets import (
    QMainWindow, QTabWidget, QWidget
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QCloseEvent
from download_menu import DownloadMenu
from selection_creating_menu import SelectionCreatingMenu
from analysis_menu import AnalysisMenu
from word_doc_matrix import find_dat_files


class MainWindow(QMainWindow):
    quit_app = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Дипломная работа')
        self.setMinimumSize(640, 480)
        # self.setMaximumSize(1280, 720)
        self.setWindowFlags(Qt.MSWindowsOwnDC)
        # self.setAttribute(Qt.WA_QuitOnClose, True)

        self.download_menu_layout = DownloadMenu()
        download_menu_widget = QWidget()
        download_menu_widget.setLayout(self.download_menu_layout)

        self.selection_creating_menu_layout = SelectionCreatingMenu()
        selection_creating_menu_widget = QWidget()
        selection_creating_menu_widget.setLayout(self.selection_creating_menu_layout)

        self.analysis_menu_layout = AnalysisMenu()
        analysis_menu_widget = QWidget()
        analysis_menu_widget.setLayout(self.analysis_menu_layout)

        self.download_menu_layout.add_new_db.connect(self.add_new_database_to_selection_creating_menu)
        self.selection_creating_menu_layout.calculation_started.connect(self.analysis_menu_layout.delete_current_matrix)
        self.selection_creating_menu_layout.calculation_started.connect(self.make_disabled)
        self.selection_creating_menu_layout.calculation_finished.connect(self.add_new_matrix_to_analysis_menu)
        self.selection_creating_menu_layout.calculation_finished.connect(self.make_enabled)
        self.selection_creating_menu_layout.matrix.connect(self.analysis_menu_layout.set_selection_matrix)

        self.tab = QTabWidget()
        self.tab.addTab(download_menu_widget, 'Скачать статьи')
        self.tab.addTab(selection_creating_menu_widget, 'Создать матрицу')
        self.tab.addTab(analysis_menu_widget, 'Анализ матрицы')
        self.tab.setTabPosition(QTabWidget.TabPosition.West)
        # self.tab.setself.tabBarAutoHide(True)
        self.setCentralWidget(self.tab)
        self.resize(self.minimumSize())

    @Slot()
    def make_disabled(self):
        self.setEnabled(False)

    @Slot(bool)
    def make_enabled(self, _: bool):
        self.setEnabled(True)

    @Slot(str)
    def add_new_database_to_selection_creating_menu(self, db_name: str):
        if self.selection_creating_menu_layout.add_db_and_tables_choose_menu(db_name):
            self.tab.setCurrentIndex(1)

    @Slot(bool)
    def add_new_matrix_to_analysis_menu(self, add_new_matrix: bool):
        if add_new_matrix:
            self.analysis_menu_layout.selection_matrix_combo.clear()
            self.analysis_menu_layout.selection_matrix_combo.addItems(find_dat_files())
        self.tab.setCurrentIndex(2)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.quit_app.emit()
