from PySide6.QtWidgets import (
    QWidget, QComboBox, QMainWindow, QCheckBox, QScrollArea, QVBoxLayout, QLabel, QHBoxLayout,
    QSizePolicy, QStackedWidget, QPushButton, QLineEdit, QMessageBox, QProgressBar, QTableWidget, QTableWidgetItem,
    QSplitter, QAbstractScrollArea, QTabWidget, QToolBar, QTextEdit, QDialog, QApplication, QHeaderView
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtGui import QIntValidator, QIcon
from db_sqlite3 import Database, Table, find_sqlite_files, correct_queries_for_output, correct_query_for_output
from word_doc_matrix import SelectionMatrix, Worker, DEFAULT_STOPWORDS
from db_manager_window import DBManagerWindow
from typing import List


class SelectionCreatingDialog(QDialog):

    def __init__(self, db_name: str, requested_tables: List[str]):
        super().__init__()
        self.setWindowTitle('Создание выборки')

        self.warning = None
        self.new_class_name_dialog = None

        db_label = QLabel(f'База данных: {db_name}')
        db_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.selection_name = QLineEdit()
        self.selection_name.setFixedWidth(220)
        self.selection_name.setPlaceholderText('Название выборки')
        self.selection_name.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.articles_per_class = QLineEdit()
        self.articles_per_class.setFixedWidth(220)
        self.articles_per_class.setPlaceholderText('Количество статей на класс (-1 макс)')
        self.articles_per_class.setValidator(QIntValidator(-1, 9999))
        self.articles_per_class.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        weight_method_layout = QHBoxLayout()
        weight_method_layout.setAlignment(Qt.AlignHCenter)
        weight_method_layout.setSizeConstraint(weight_method_layout.SizeConstraint.SetMinAndMaxSize)
        weight_method_layout.setSpacing(10)
        weight_method_layout.addWidget(QLabel('Метод взвешивания:'))
        self.weight_method = QComboBox()
        self.weight_method.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.weight_method.addItems(['tf', 'tfidf', 'tfc'])
        self.weight_method.setCurrentIndex(2)
        weight_method_layout.addWidget(self.weight_method)

        self.with_save = QCheckBox('Сохранить матрицу в файл')
        # self.with_save.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.use_random = QCheckBox('Случайные статьи')
        self.use_random.setChecked(True)
        self.use_random.stateChanged.connect(self.use_random_state_changed)

        random_state_layout = QHBoxLayout()
        random_state_layout.setAlignment(Qt.AlignLeft)
        random_state_layout.setSizeConstraint(QHBoxLayout.SetMinAndMaxSize)
        random_state_label = QLabel('      Random State:')
        random_state_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        random_state_layout.addWidget(random_state_label)
        self.random_state = QLineEdit('0')
        self.random_state.setFixedWidth(40)
        validator = QIntValidator(0, 9999)
        self.random_state.setValidator(validator)
        random_state_layout.addWidget(self.random_state)

        top_layout = QVBoxLayout()
        top_layout.setAlignment(Qt.AlignHCenter)
        top_layout.addWidget(db_label, alignment=Qt.AlignHCenter)
        top_layout.addWidget(self.selection_name)
        top_layout.addWidget(self.articles_per_class)
        top_widget = QWidget()
        top_widget.setLayout(top_layout)

        mid_layout = QVBoxLayout()
        mid_layout.addLayout(weight_method_layout)
        mid_layout.addWidget(self.with_save)
        mid_layout.addWidget(self.use_random)
        mid_layout.addLayout(random_state_layout)
        mid_widget = QWidget()
        mid_widget.setLayout(mid_layout)

        db = Database(f'data/{db_name}', requested_tables)
        self.stacked_widget = QStackedWidget()

        tables_names = db.get_source_tables_names()
        self.classes_names_table = QTableWidget(len(tables_names), 1)
        self.classes_names_table.horizontalHeader().setHidden(True)
        self.classes_names_table.horizontalHeader().setDefaultSectionSize(490)
        self.classes_names_table.setVerticalHeaderLabels([f'Класс {i}' for i in range(1, len(tables_names)+1)])
        self.classes_names_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.classes_names_table.verticalHeader().setSectionsMovable(True)
        self.classes_names_table.verticalHeader().sectionMoved.connect(self.move_class)
        for i, table in enumerate(tables_names):
            corrected_query = correct_query_for_output(table)
            class_name = QLineEdit(corrected_query)
            class_name.setFrame(False)
            class_name.setCursorPosition(0)
            class_name.setClearButtonEnabled(True)
            class_name.setFixedWidth(490)
            class_name.setObjectName(table)
            class_name.textEdited.connect(self.change_name_from_line_edit)
            self.classes_names_table.setCellWidget(i, 0, class_name)

        classes_names_buttons_layout = QHBoxLayout()
        classes_names_buttons_layout.setAlignment(Qt.AlignLeft)
        ok_button = QPushButton('Создать')
        ok_button.clicked.connect(self.check_input)
        join_tables_button = QPushButton('Объединить классы')
        join_tables_button.clicked.connect(self.switch_stacked_widget)
        classes_names_buttons_layout.addWidget(ok_button)
        classes_names_buttons_layout.addWidget(join_tables_button)

        classes_names_layout = QVBoxLayout()
        classes_names_layout.addWidget(self.classes_names_table)
        classes_names_layout.addLayout(classes_names_buttons_layout)
        classes_names_menu = QWidget()
        classes_names_menu.setLayout(classes_names_layout)
        self.stacked_widget.addWidget(classes_names_menu)

        self.classes_union_table = QTableWidget(len(tables_names), 2)
        self.classes_union_table.setShowGrid(False)
        self.classes_union_table.horizontalHeader().setHidden(True)
        self.classes_union_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.classes_union_table.verticalHeader().setSectionsMovable(True)
        self.classes_union_table.verticalHeader().setHidden(True)
        self.classes_union_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        # self.classes_union_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        for i, table in enumerate(tables_names):
            class_checkbox = QCheckBox(correct_query_for_output(table))
            class_checkbox.setObjectName(table)
            # class_checkbox.setFixedHeight(16)
            self.classes_union_table.setCellWidget(i, 0, class_checkbox)
            blank_item = QTableWidgetItem('')
            blank_item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.classes_union_table.setItem(i, 1, blank_item)

        classes_union_buttons_layout = QHBoxLayout()
        classes_union_buttons_layout.setAlignment(Qt.AlignLeft)
        make_union_button = QPushButton('Объединить')
        make_union_button.clicked.connect(self.make_class_union)
        cancel_button = QPushButton('Отменить объединение')
        cancel_button.clicked.connect(self.switch_stacked_widget)
        classes_union_buttons_layout.addWidget(make_union_button)
        classes_union_buttons_layout.addWidget(cancel_button)

        classes_union_layout = QVBoxLayout()
        # classes_union_layout.setAlignment(Qt.AlignTop)
        classes_union_layout.addWidget(self.classes_union_table)
        classes_union_layout.addLayout(classes_union_buttons_layout)
        classes_union_menu = QWidget()
        classes_union_menu.setLayout(classes_union_layout)
        self.stacked_widget.addWidget(classes_union_menu)

        layout = QVBoxLayout()
        # layout.setAlignment(Qt.AlignTop)
        # layout.addWidget(class_size_info)
        # layout.addWidget(use_random)
        # layout.addLayout(random_state_layout)
        layout.addWidget(top_widget, alignment=Qt.AlignHCenter)
        layout.addWidget(mid_widget, alignment=Qt.AlignHCenter)
        layout.addWidget(self.stacked_widget)
        self.resize(600, 500)
        self.setLayout(layout)

    def renumber_classes(self):
        visual_rows = [self.classes_names_table.visualRow(i) for i in range(self.classes_names_table.rowCount())]
        for i in range(self.classes_names_table.rowCount()):
            self.classes_names_table.setVerticalHeaderItem(visual_rows.index(i), QTableWidgetItem(f'Класс {i + 1}'))

    @Slot(int, int, int)
    def move_class(self, _: int, old_pos: int, new_pos: int):
        self.classes_union_table.verticalHeader().moveSection(old_pos, new_pos)
        self.renumber_classes()

    @Slot(int)
    def use_random_state_changed(self, state: int):
        if state:
            self.random_state.setEnabled(True)
        else:
            self.random_state.setEnabled(False)

    def change_name_from_line_edit(self):
        class_name = self.sender().text()
        self.classes_union_table.findChild(QCheckBox, self.sender().objectName()).setText(class_name)

    def make_class_union(self):
        all_checkboxes = self.classes_union_table.findChildren(QCheckBox)
        checked_checkboxes = [checkbox for checkbox in all_checkboxes if checkbox.isChecked()]

        if not checked_checkboxes:
            self.warning = QMessageBox(QMessageBox.Icon.Critical, 'Ошибка', 'Не выбраны классы для объединения')
            self.warning.show()
            return

        self.new_class_name_dialog = QDialog()
        # self.new_class_name_dialog.setAttribute(Qt.WA_DeleteOnClose, True)

        layout = QVBoxLayout()
        layout.addWidget(QLabel('Имя нового класса'))
        class_name = QTextEdit('; '.join(map(lambda x: x.text(), checked_checkboxes)))
        layout.addWidget(class_name)
        ok_button = QPushButton('ОК')
        ok_button.clicked.connect(self.new_class_name_dialog.accept)
        layout.addWidget(ok_button)

        self.new_class_name_dialog.setLayout(layout)
        if self.new_class_name_dialog.exec():
            new_class_name = class_name.document().toRawText().replace('\u2029', ' ').strip()
            checkbox = checked_checkboxes[0]
            line = self.classes_names_table.findChild(QLineEdit, checkbox.objectName())
            checkbox.setText(new_class_name)
            checkbox.setObjectName('; '.join(map(lambda x: x.objectName(), checked_checkboxes)))
            line.setText(new_class_name)
            line.setObjectName('; '.join(map(lambda x: x.objectName(), checked_checkboxes)))
            for row in [all_checkboxes.index(checkbox) for checkbox in checked_checkboxes[:0:-1]]:
                self.classes_names_table.removeRow(row)
                self.classes_union_table.removeRow(row)
            self.renumber_classes()
            self.stacked_widget.setCurrentIndex(0)

    def switch_stacked_widget(self):
        if self.stacked_widget.currentIndex() == 0:
            for checkbox in self.stacked_widget.findChildren(QCheckBox):
                checkbox.setChecked(False)
            self.stacked_widget.setCurrentIndex(1)
        elif self.stacked_widget.currentIndex() == 1:
            self.stacked_widget.setCurrentIndex(0)

    def check_input(self):
        try:
            articles_per_class = int(self.articles_per_class.text())
        except ValueError:
            self.warning = QMessageBox(QMessageBox.Icon.Critical, 'Ошибка', 'Не задано количество статей')
            self.warning.show()
            return
        if articles_per_class < -1 or 0 <= articles_per_class <= 1:
            self.warning = QMessageBox(QMessageBox.Icon.Critical, 'Ошибка', 'Количество статей: от 2 до 9999')
            self.warning.show()
            return
        self.accept()

    def get_result(self):
        # if self.use_random:
        #     if self.random_state.validator().validate(self.random_state.text(), 0)[0] != QIntValidator.Acceptable:

        visual_rows = [self.classes_names_table.visualRow(i) for i in range(self.classes_names_table.rowCount())]
        classes_with_tables = dict()
        for i in range(self.classes_names_table.rowCount()):
            line = self.classes_names_table.cellWidget(visual_rows.index(i), 0)
            classes_with_tables[line.text()] = line.objectName().split('; ')

        selection_params = {
            'selection name': self.selection_name.text(),
            'articles per class': int(self.articles_per_class.text()),
            'weight method': self.weight_method.currentText(),
            'with save': True if self.with_save.checkState() else False,
            'random state': int(self.random_state.text()) if self.use_random.checkState() else -1
        }
        return classes_with_tables, selection_params


class SelectionCreatingMenu(QVBoxLayout):
    calculation_started = Signal()
    calculation_finished = Signal(bool)
    matrix = Signal(object)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignHCenter)
        self.setSizeConstraint(self.SizeConstraint.SetMinAndMaxSize)
        self.setSpacing(10)

        self.warning = None
        # self.duplicates_table = None
        self.info_window = None
        self.db_manager_window = None
        self.selection_dialog = None
        self.stopwords_window = None
        self.stopwords = DEFAULT_STOPWORDS

        self.progress_dialog = None
        # self.timer = QTimer()
        # self.timer.setInterval(200)

        # self.selection_name = QLineEdit()
        # self.selection_name.setFixedWidth(220)
        # self.selection_name.setPlaceholderText('Название выборки')
        # self.selection_name.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        #
        # self.articles_per_class = QLineEdit()
        # self.articles_per_class.setFixedWidth(220)
        # self.articles_per_class.setPlaceholderText('Количество статей на класс (-1 макс)')
        # self.articles_per_class.setValidator(QIntValidator(-1, 9999))
        # self.articles_per_class.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        #
        # weight_method_layout = QHBoxLayout()
        # weight_method_layout.setAlignment(Qt.AlignHCenter)
        # weight_method_layout.setSizeConstraint(weight_method_layout.SizeConstraint.SetMinAndMaxSize)
        # weight_method_layout.setSpacing(5)
        # weight_method_layout.addWidget(QLabel('Метод взвешивания:'))
        # self.weight_method = QComboBox()
        # self.weight_method.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        # self.weight_method.addItems(['tf', 'tfidf', 'tfc'])
        # self.weight_method.setCurrentIndex(2)
        # weight_method_layout.addWidget(self.weight_method)

        self.stacked_widget = QStackedWidget()
        self.db_combo_box = QComboBox()
        self.db_combo_box.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        # self.db_combo_box.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.db_combo_box.currentIndexChanged.connect(self.stacked_widget.setCurrentIndex)
        sqlite_files = find_sqlite_files()
        for db in sqlite_files:
            self.add_db_and_tables_choose_menu(db)

        self.addWidget(QLabel('Создание взвешенной матрицы документ-термин'), alignment=Qt.AlignHCenter)
        # self.addWidget(self.selection_name, alignment=Qt.AlignHCenter)  #
        # self.addWidget(self.articles_per_class, alignment=Qt.AlignHCenter)  #
        # self.addLayout(weight_method_layout)

        # self.with_save = QCheckBox('Сохранить матрицу в файл')
        # self.with_save.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        # self.addWidget(self.with_save, alignment=Qt.AlignHCenter)

        # duplicates_button = QPushButton('Таблица дубликатов')
        # duplicates_button.clicked.connect(self.make_duplicate_table_matrix)

        info_icon = QIcon()
        info_icon.addFile('icons/info.png')
        info_button = QPushButton(icon=info_icon)
        info_button.setToolTip('Информация о выбранных таблицах')
        info_button.setFixedSize(50, 30)
        info_button.clicked.connect(self.make_info_window)

        db_manage_icon = QIcon()
        db_manage_icon.addFile('icons/db manager 1.png')
        db_manage_button = QPushButton(icon=db_manage_icon)
        db_manage_button.setToolTip('Управление базами данных')
        db_manage_button.setFixedSize(50, 30)
        # db_manage.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        db_manage_button.clicked.connect(self.open_db_manager_window)

        toolbar = QToolBar(parent=self.stacked_widget)
        toolbar.addWidget(info_button)
        toolbar.addWidget(db_manage_button)
        toolbar.setFloatable(False)
        toolbar.setMovable(False)

        # buttons_layout = QHBoxLayout()
        # buttons_layout.setAlignment(Qt.AlignLeft)
        # buttons_layout.setSpacing(8)
        # # buttons_layout.addWidget(self.ok_button)
        # # buttons_layout.addWidget(duplicates_button)
        # buttons_layout.addWidget(info_button)
        # buttons_layout.addWidget(db_manage)
        # toolbar.addLayout(buttons_layout)

        self.addWidget(toolbar)
        self.addWidget(self.db_combo_box)
        self.addWidget(self.stacked_widget)

        self.ok_button = QPushButton('Создать выборку')
        self.ok_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.ok_button.clicked.connect(self.make_selection_matrix)

        stopwords_button = QPushButton('Стоп-слова')
        stopwords_button.clicked.connect(self.open_stopwords_window)

        bottom_line = QHBoxLayout()
        bottom_line.setSpacing(8)
        bottom_line.setAlignment(Qt.AlignLeft)
        bottom_line.addWidget(self.ok_button)
        bottom_line.addWidget(stopwords_button)
        self.addLayout(bottom_line)

    def get_requested_tables(self):
        requested_tables = list()
        widget = self.stacked_widget.currentWidget()
        check_boxes = widget.findChildren(QCheckBox)
        for check_box in check_boxes:
            if check_box.isChecked():
                requested_tables.append(check_box.objectName())
        return requested_tables

    @Slot(str)
    def delete_db(self, db_name: str):
        index = [self.db_combo_box.itemText(i) for i in range(self.db_combo_box.count())].index(db_name)
        self.db_combo_box.removeItem(index)
        self.stacked_widget.removeWidget(self.stacked_widget.widget(index))

    @Slot(str, str)
    def delete_table(self, db_name: str, table_name: str):
        index = [self.db_combo_box.itemText(i) for i in range(self.db_combo_box.count())].index(db_name)
        table_checkbox = self.stacked_widget.widget(index).findChild(QCheckBox, table_name)
        table_checkbox.close()
        table_checkbox.deleteLater()

    def add_db_and_tables_choose_menu(self, db):
        tables = Database(f'data/{db}', []).get_source_tables_info()
        if not tables:
            return False
        try:
            index = [self.db_combo_box.itemText(i) for i in range(self.db_combo_box.count())].index(db)

        except ValueError:
            layout = QVBoxLayout()
            layout.setAlignment(Qt.AlignTop)
            for table, count in tables.items():
                table_name = correct_query_for_output(table)
                widget = QCheckBox(f"{table_name}: {count}")
                widget.setChecked(True)
                widget.setObjectName(table)
                layout.addWidget(widget)
            widget = QWidget()
            widget.setLayout(layout)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(widget)
            self.db_combo_box.addItem(db)
            self.stacked_widget.addWidget(scroll)

        else:
            layout = self.stacked_widget.widget(index).findChild(QVBoxLayout)
            tables_ui = [c.objectName() for c in self.stacked_widget.widget(index).findChildren(QCheckBox)
                         if not c.isHidden()]
            tables_db = Database(f'data/{db}', []).get_source_tables_names()

            for table_db in tables_db:
                if table_db not in tables_ui:
                    table_name = correct_query_for_output(table_db)
                    widget = QCheckBox(f"{table_name}: {Table(table_db, f'data/{db}').get_n_rows()}")
                    widget.setObjectName(table_db)
                    layout.addWidget(widget)
            self.stacked_widget.widget(index).update()

        finally:
            return True

    def open_db_manager_window(self):
        self.db_manager_window = DBManagerWindow()
        self.db_manager_window.written_db.connect(self.add_db_and_tables_choose_menu)
        self.db_manager_window.db_deleted.connect(self.delete_db)
        self.db_manager_window.table_deleted.connect(self.delete_table)
        self.db_manager_window.show()

    def make_selection_matrix(self):
        db_name = self.db_combo_box.currentText()
        requested_tables = self.get_requested_tables()
        self.selection_dialog = SelectionCreatingDialog(db_name, requested_tables)

        if self.selection_dialog.exec():
            self.calculation_started.emit()
            classes_with_tables, selection_params = self.selection_dialog.get_result()

            selection_name = selection_params['selection name']
            articles_per_class = selection_params['articles per class']
            weight_method = selection_params['weight method']
            with_save = selection_params['with save']
            random_state = selection_params['random state']

            articles = Database(f'data/{db_name}', requested_tables).get_selection_articles(classes_with_tables,
                                                                                            articles_per_class,
                                                                                            random_state)
            # print(articles)

            self.progress_dialog = QWidget()
            self.progress_dialog.setFixedSize(220, 45)
            self.progress_dialog.setWindowTitle('Создание матрицы')
            self.progress_dialog.setWindowFlags((Qt.WindowStaysOnTopHint | Qt.CustomizeWindowHint | Qt.WindowTitleHint))
            # self.progress_dialog.setWindowFlags(Qt.Dialog)
            progress_bar = QProgressBar()
            # progress_bar.setHidden(True)
            progress_bar.setFixedSize(200, 25)
            progress_bar.setAlignment(Qt.AlignCenter)
            layout = QVBoxLayout()
            layout.addWidget(progress_bar)
            self.progress_dialog.setLayout(layout)

            worker = Worker(db_name, selection_name, articles_per_class, weight_method,
                            articles, self.stopwords, with_save)

            worker.progress_max.connect(progress_bar.setMaximum)
            worker.progress_current_action.connect(progress_bar.setFormat)
            worker.progress_current_i.connect(progress_bar.setValue)
            # worker.progress_current_i.connect(self.timer.start)
            worker.process_finished.connect(self.push_matrix_to_analysis_menu)
            worker.calculation_finished.connect(self.progress_dialog.close)
            worker.calculation_finished.connect(self.emit_calculation_finished)

            self.progress_dialog.show()
            worker.start_work()

    @Slot(bool)
    def emit_calculation_finished(self, with_save: bool):
        self.calculation_finished.emit(with_save)

    @Slot(object)
    def push_matrix_to_analysis_menu(self, matrix: SelectionMatrix):
        self.matrix.emit(matrix)
        self.progress_dialog = None

    def make_duplicate_table_matrix(self, requested_tables):
        if not self.db_combo_box.count():
            self.warning = QMessageBox(QMessageBox.Icon.Critical, 'Ошибка', 'Нет баз данных для работы')
            self.warning.show()
            return

        db_name = self.db_combo_box.currentText()
        db = Database(f'data/{db_name}', requested_tables)
        duplicates_table = db.get_info_for_duplicates_table()

        table_widget = QTableWidget(len(duplicates_table), len(duplicates_table))
        table_widget.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        table_widget.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        table_widget.setHorizontalHeaderLabels([f'Таблица {i}' for i in range(1, len(duplicates_table) + 1)])
        table_widget.setVerticalHeaderLabels([f'Таблица {i}' for i in range(1, len(duplicates_table) + 1)])
        for i, (_, row) in enumerate(duplicates_table.items()):
            for j, (_, num_of_duplicates) in enumerate(row.items()):
                table_cell = QTableWidgetItem(str(num_of_duplicates))
                table_cell.setFlags(Qt.ItemFlag.ItemIsEnabled)
                table_cell.setTextAlignment(Qt.AlignCenter)
                table_widget.setItem(i, j, table_cell)

        table_layout = QVBoxLayout()
        table_layout.setAlignment(Qt.AlignTop)
        table_layout.addWidget(QLabel('Матрица дубликатов'), alignment=Qt.AlignHCenter)
        # alignment=(Qt.AlignHCenter | Qt.AlignTop))
        table_layout.addWidget(table_widget)
        duplicate_matrix = QWidget()
        duplicate_matrix.setLayout(table_layout)
        return duplicate_matrix

    def make_info_window(self):
        if not self.db_combo_box.count():
            self.warning = QMessageBox(QMessageBox.Icon.Critical, 'Ошибка', 'Нет баз данных для работы')
            self.warning.show()
            return
        requested_tables = self.get_requested_tables()
        if not requested_tables:
            self.warning = QMessageBox(QMessageBox.Icon.Critical, 'Ошибка', 'Нет выбранных таблиц')
            self.warning.show()
            return

        duplicate_matrix = self.make_duplicate_table_matrix(requested_tables)

        db_name = self.db_combo_box.currentText()
        db = Database(f'data/{db_name}', requested_tables)
        info = db.get_selection_info()

        tables_tab = QTabWidget()
        tables_tab.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        corrected_queries = correct_queries_for_output(list(info.keys()))
        for i, table in enumerate(info):
            scroll_layout = QVBoxLayout()
            tags_output = ';\n'.join(map(lambda x: f'\t{x[0]}: {x[1]} статей', info[table]['tags'].most_common(10)))
            minmax_dates = tuple(map(lambda x: '.'.join(reversed(x.split('-'))), info[table]['minmax dates']))
            submitted_output = f'Статьи опубликованы от {minmax_dates[0]} до {minmax_dates[1]}'
            label_table = QLabel(f'{corrected_queries[i]}: {info[table]["num of articles"]} статей')
            label_table.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            label_tags = QLabel(f'{tags_output}\n\n{submitted_output}')
            label_tags.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            scroll_layout.addWidget(label_table)
            scroll_layout.addWidget(label_tags)

            scroll_widget = QWidget()
            scroll_widget.setLayout(scroll_layout)
            scroll = QScrollArea()
            scroll.setWidget(scroll_widget)

            tables_tab.addTab(scroll, f'Таблица {i + 1}')

        self.info_window = QSplitter(Qt.Orientation.Vertical)
        self.info_window.addWidget(duplicate_matrix)
        self.info_window.addWidget(tables_tab)
        self.info_window.setWindowTitle('Информация')
        # self.info_window.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.info_window.show()

    def open_stopwords_window(self):
        row = ''
        text = ''
        row_length = 150
        for stopword in self.stopwords[:-1]:
            if len(row) + len(stopword) + 2 > row_length:
                text += '\n'
                row = ''
            row += stopword + ', '
            text += stopword + ', '
        text += self.stopwords[-1]

        self.stopwords_window = QDialog()
        self.stopwords_window.setWindowTitle('Стоп-слова')
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignHCenter)
        text_edit = QTextEdit(text)
        layout.addWidget(text_edit)
        ok_button = QPushButton('Ок')
        ok_button.clicked.connect(self.stopwords_window.accept)
        layout.addWidget(ok_button)
        self.stopwords_window.setLayout(layout)

        if self.stopwords_window.exec() == QDialog.Accepted:
            # self.stopwords = list()
            self.stopwords = text_edit.document().toRawText().replace('\u2029', ' ').split(', ')
            self.stopwords = [sw.strip() for sw in self.stopwords]
            # print(self.stopwords)


class SelectionCreatingWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setWindowTitle('Формирование выборки')
        self.setMinimumSize(640, 480)
        self.setMaximumSize(1280, 720)
        # self.resize(1280, 720)
        self.setWindowFlags(Qt.MSWindowsOwnDC)
        widget = QWidget()
        widget.setLayout(SelectionCreatingMenu())
        self.setCentralWidget(widget)


if __name__ == '__main__':
    app = QApplication()

    window = SelectionCreatingDialog('target_base.sqlite3', [])
    window.show()

    app.exec()
