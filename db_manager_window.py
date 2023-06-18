from PySide6.QtWidgets import (
    QWidget, QComboBox, QCheckBox, QScrollArea, QVBoxLayout, QLabel, QHBoxLayout,
    QSizePolicy, QPushButton, QDialog, QTableWidget, QTableWidgetItem, QHeaderView
)
from PySide6.QtCore import Qt, QSize, Slot, Signal
from PySide6.QtGui import QIcon, QKeyEvent
from db_sqlite3 import correct_query_for_output, find_sqlite_files, Database, Table
from typing import List
# from math import ceil


def delete_dialog(text: str):
    dialog = QDialog()
    dialog.setWindowTitle('Удаление')
    layout = QVBoxLayout()
    layout.addWidget(QLabel(f'Удалить {text}?'))
    ok_button = QPushButton('Да')
    ok_button.clicked.connect(dialog.accept)
    cancel_button = QPushButton('Нет')
    cancel_button.clicked.connect(dialog.reject)
    buttons_layout = QHBoxLayout()
    buttons_layout.setAlignment(Qt.AlignHCenter)
    buttons_layout.addWidget(ok_button)
    buttons_layout.addWidget(cancel_button)
    layout.addLayout(buttons_layout)
    dialog.setLayout(layout)
    return dialog


class QTableDateItem(QTableWidgetItem):

    def __lt__(self, other):
        try:
            # return '-'.join(self.text().split('.')[::-1]) < '-'.join(other.text().split('.')[::-1])
            # print(self.data(Qt.UserRole))
            return self.data(Qt.UserRole) < other.data(Qt.UserRole)
        except ValueError:
            return super(QTableDateItem, self).__lt__(other)


class FullArticle(QTableWidget):
    
    def __init__(self, article: tuple, fields_for_output: List[str]):
        super().__init__(len(fields_for_output), 1)
        default_fields_for_output = ['url', 'tags_codes', 'tags_names', 'authors', 'title', 'abstract', 'submitted']
        fields_for_output = [field for field in fields_for_output if field in default_fields_for_output]
        if not fields_for_output:
            fields_for_output = default_fields_for_output
        # print(fields_for_output)
        self.output_fields_translated = {'url': 'Адрес',
                                         'tags_codes': 'Теги\n(кратко)',
                                         'tags_names': 'Теги',
                                         'authors': 'Авторы',
                                         'title': 'Название',
                                         'abstract': 'Описание',
                                         'submitted': 'Дата'}

        # table_widget.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.MinimumExpanding)
        self.setSizeAdjustPolicy(QTableWidget.AdjustToContents)
        self.setFixedWidth(400)
        self.horizontalHeader().setHidden(True)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.verticalHeader().setDefaultAlignment((Qt.AlignHCenter | Qt.AlignTop))
        self.verticalHeader().setHighlightSections(True)
        self.setVerticalHeaderLabels([self.output_fields_translated[string] for string in fields_for_output])
        font = self.verticalHeader().font()
        font.setBold(True)
        self.verticalHeader().setFont(font)

        for i, field in enumerate(article[:2]):
            table_cell = QTableWidgetItem(field)
            table_cell.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.setItem(i, 0, table_cell)
        for i, field in enumerate(article[2:-1], 2):
            row_width = 55
            row = ''
            text = ''
            for word in field.split():
                if len(row) + len(word) + 1 > row_width:
                    text += '\n'
                    row = ''
                row += ' ' + word
                text += ' ' + word

            table_cell = QTableWidgetItem(text)
            table_cell.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.setItem(i, 0, table_cell)

        table_cell = QTableWidgetItem('.'.join(article[-1].split('-')[::-1]))
        table_cell.setFlags(Qt.ItemFlag.ItemIsEnabled)
        self.setItem(len(article) - 1, 0, table_cell)


class TableEditor(QTableWidget):
    article_deleted = Signal()

    def __init__(self, table_name, db_name):

        self.full_article_window = None
        self.delete_dialog_window = None

        self.table = Table(table_name, f'data/{db_name}')
        fields_for_output = ['url', 'title', 'tags_names', 'submitted']
        self.output_fields_translated = {'url': 'Адрес',
                                         'tags_codes': 'Теги\n(кратко)',
                                         'tags_names': 'Теги',
                                         'authors': 'Авторы',
                                         'title': 'Название',
                                         'abstract': 'Описание',
                                         'submitted': 'Дата'}

        table_data = self.table.get_data(fields_for_output)
        super(TableEditor, self).__init__(len(table_data), len(fields_for_output))

        self.setObjectName(f'{db_name}: {table_name}')
        self.setWindowTitle(correct_query_for_output(table_name))
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setWindowFlags(Qt.MSWindowsOwnDC)
        self.setWindowState(Qt.WindowState.WindowMaximized)
        self.setMinimumSize(640, 480)
        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.setSizeAdjustPolicy(QTableWidget.AdjustToContents)

        self.setHorizontalHeaderLabels([self.output_fields_translated[string] for string in fields_for_output])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)

        self.verticalHeader().sectionDoubleClicked.connect(self.show_full_article)
        self.cellDoubleClicked.connect(self.show_full_article)

        date_col = len(fields_for_output) - 1
        for i, row in enumerate(table_data):
            # print(len(row))
            for j, field in enumerate(row):
                # print(field)
                if j != date_col:
                    text = ''
                    if j == 0:
                        text = field

                    else:
                        row_width = 90
                        row_text = ''
                        for word in field.split():
                            if len(row_text) + len(word) + 1 > row_width:
                                text += '\n'
                                row_text = ''
                            row_text += word + ' '
                            text += word + ' '

                    table_cell = QTableWidgetItem(text)
                    table_cell.setFlags((Qt.ItemFlag.ItemIsEnabled | Qt.ItemIsSelectable))
                    table_cell.setTextAlignment(Qt.AlignCenter)
                    self.setItem(i, j, table_cell)

                else:
                    table_cell = QTableDateItem('.'.join(field.split('-')[::-1]))  #
                    table_cell.setData(Qt.UserRole, field)
                    table_cell.setFlags((Qt.ItemFlag.ItemIsEnabled | Qt.ItemIsSelectable))
                    table_cell.setTextAlignment(Qt.AlignCenter)
                    self.setItem(i, j, table_cell)

        self.setSortingEnabled(True)

    @Slot(int)
    def show_full_article(self, row: int):
        fields_for_output = ['url', 'tags_codes', 'tags_names', 'authors', 'title', 'abstract', 'submitted']
        url = self.item(row, 0).text()
        article = self.table.get_article(fields_for_output, url)

        self.full_article_window = FullArticle(article, fields_for_output)
        self.full_article_window.setWindowTitle(url)
        self.full_article_window.show()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key == Qt.Key.Key_Delete and self.selectedItems():
            row = self.currentIndex().row()
            # print(self.item(row-1, 0).text())
            url = self.item(row, 0).text()
            self.delete_dialog_window = delete_dialog(f'строку {row+1} с адресом\n{url}')
            if self.delete_dialog_window.exec():
                self.table.delete_article(url)
                self.removeRow(row)


class TableRow(QHBoxLayout):
    removed_table = Signal(QWidget)
    table_to_edit = Signal(str)

    def __init__(self, table: str, count: int):
        super().__init__()
        self.setAlignment(Qt.AlignLeft)
        table_name = correct_query_for_output(table)
        delete_icon = QIcon()
        delete_icon.addFile("icons/delete1.png", size=QSize(25, 25))
        edit_icon = QIcon()
        edit_icon.addFile("icons/edit1.png", size=QSize(25, 25))
        self.table_delete_button = QPushButton(icon=delete_icon)
        self.table_delete_button.setFixedSize(QSize(25, 25))
        self.table_delete_button.clicked.connect(self.removed_table_row)
        self.table_edit_button = QPushButton(icon=edit_icon)
        self.table_edit_button.setFixedSize(QSize(25, 25))
        self.table_edit_button.clicked.connect(self.edit_table)
        self.table_checkbox = QCheckBox(f'{table_name}: {count}')
        self.table_checkbox.setObjectName(table)
        self.addWidget(QLabel('    '))
        self.addWidget(self.table_delete_button)
        self.addWidget(self.table_edit_button)
        self.addWidget(self.table_checkbox)

        # self.table_delete_button.setHidden(True)
        # self.table_edit_button.setHidden(True)

    def removed_table_row(self):
        self.removed_table.emit(self.parentWidget())

    def edit_table(self):
        self.table_to_edit.emit(self.table_checkbox.objectName())


class DBColumn(QVBoxLayout):
    removed_db_column = Signal(QWidget)
    removed_table_row = Signal(str, QWidget)
    delete_dialog = Signal()
    db_table_to_edit = Signal(str, str)

    def __init__(self, db_name):
        super().__init__()
        # self.setObjectName(db_name)

        tables_info = Database(f'data/{db_name}', []).get_source_tables_info()

        db_layout = QHBoxLayout(self.parentWidget())
        delete_icon = QIcon()
        delete_icon.addFile("icons/delete1.png", size=QSize(25, 25))
        db_delete_button = QPushButton(icon=delete_icon)
        db_delete_button.setFixedSize(QSize(25, 25))
        # db_delete_button.
        db_delete_button.clicked.connect(self.removed_db)
        self.db_checkbox = QCheckBox(db_name)
        self.db_checkbox.clicked.connect(self.include_or_exclude_db)
        db_font = self.db_checkbox.font()
        db_font.setBold(True)
        self.db_checkbox.setFont(db_font)
        db_layout.addWidget(db_delete_button)
        db_layout.addWidget(self.db_checkbox)
        self.addLayout(db_layout)

        for table, count in tables_info.items():
            self.add_table(table, count)

    def include_or_exclude_db(self):
        db_check_state = self.db_checkbox.checkState()
        for checkbox in self.parentWidget().findChildren(QCheckBox):
            checkbox.setCheckState(db_check_state)

    def will_delete_table(self):
        self.delete_dialog.emit()

    @Slot(QWidget)
    def remove_table_row(self, table_row: QWidget):
        self.removed_table_row.emit(self.db_checkbox.text(), table_row)

    def removed_db(self):
        self.removed_db_column.emit(self.parentWidget())
        # self.db_checkbox.setEnabled(False)

    @Slot(str)
    def db_table_edit(self, table_name: str):
        self.db_table_to_edit.emit(self.db_checkbox.text(), table_name)

    def is_db_checked(self):
        checkboxes = [checkbox for checkbox in self.parentWidget().findChildren(QCheckBox) if checkbox.isEnabled()]
        # print(list(map(lambda c: c.checkState(), checkboxes)))
        if any(map(lambda c: c.checkState(), checkboxes[1:])):
            self.db_checkbox.setCheckState(Qt.Checked)
        else:
            self.db_checkbox.setCheckState(Qt.Unchecked)

    def add_table(self, table, count):
        table_layout = TableRow(table, count)
        # table_layout.setParent(self.parentWidget())
        table_layout.removed_table.connect(self.remove_table_row)
        table_layout.table_to_edit.connect(self.db_table_edit)
        table_layout.table_checkbox.clicked.connect(self.is_db_checked)
        table_widget = QWidget()
        table_widget.setLayout(table_layout)
        self.addWidget(table_widget, alignment=Qt.AlignTop)


class AllDBsArea(QScrollArea):
    db_deleted = Signal(str)
    table_deleted = Signal(str, str)

    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)

        self.table_edit_window = None
        self.delete_ensure_window = None

        self.dbs_and_tables = QVBoxLayout()
        self.dbs_and_tables.setAlignment(Qt.AlignTop)
        databases = find_sqlite_files()
        for db_name in databases:
            self.add_db_column(db_name)

        widget = QWidget()
        widget.setLayout(self.dbs_and_tables)
        self.setWidget(widget)

    @Slot(QWidget)
    def delete_db(self, db_widget: QWidget):
        # print(db_widget.layout().db_checkbox.text())
        answer = self.delete_ensure_dialog(f'{db_widget.layout().db_checkbox.text()}')
        if answer:
            self.dbs_and_tables.removeWidget(db_widget)
            db_widget.deleteLater()
            db_widget.close()
            Database(f'data/{db_widget.layout().db_checkbox.text()}', []).delete_database()
            self.db_deleted.emit(db_widget.layout().db_checkbox.text())

    @Slot(str, QWidget)
    def delete_table(self, db_name: str, table_row: QWidget):
        answer = self.delete_ensure_dialog(f'{table_row.layout().table_checkbox.text().split(":")[0]} из {db_name}')
        # print(answer)
        # print(db_name)
        if answer:
            table_row.layout().table_checkbox.setEnabled(False)
            self.dbs_and_tables.removeWidget(table_row)
            table_row.close()
            table_row.deleteLater()
            Table(table_row.layout().table_checkbox.objectName(), f'data/{db_name}').delete_table()
            self.table_deleted.emit(db_name, table_row.layout().table_checkbox.objectName())

    @Slot(str, str)
    def open_edit_window(self, db_name, table_name):
        # print(db_name, table_name)
        self.table_edit_window = TableEditor(table_name, db_name)
        self.table_edit_window.article_deleted.connect(self.update_info)
        self.table_edit_window.show()

    def delete_ensure_dialog(self, text: str):
        self.delete_ensure_window = delete_dialog(text)
        return self.delete_ensure_window.exec()

    @Slot(str)
    def update_info(self, db_and_table: str):
        db_name, table_name = db_and_table.split(': ')
        count = Table(table_name, db_name).get_n_rows()
        checkbox = self.dbs_and_tables.findChild(QCheckBox, db_name)
        checkbox.setText(f'{table_name}: {count}')

    def add_db_column(self, db_name):
        db_and_tables_layout = DBColumn(db_name)
        db_and_tables_layout.db_table_to_edit.connect(self.open_edit_window)
        # db_and_tables_layout.delete_dialog.connect(self.delete_ensure_dialog)
        db_and_tables_layout.removed_db_column.connect(self.delete_db)
        db_and_tables_layout.removed_table_row.connect(self.delete_table)
        db_and_tables_widget = QWidget()
        db_and_tables_widget.setLayout(db_and_tables_layout)
        db_and_tables_widget.setObjectName(db_name)
        self.dbs_and_tables.addWidget(db_and_tables_widget)

    # def closeEvent(self, event:PySide6.QtGui.QCloseEvent) -> None:


class DBManagerWindow(QWidget):
    db_deleted = Signal(str)
    table_deleted = Signal(str, str)
    written_db = Signal(str)
    article_deleted = Signal(str, str)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 480)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setWindowTitle('Управление базами данных')
        self.setWindowFlags(Qt.MSWindowsOwnDC)
        # self.setWindowState(Qt.WindowState.WindowMaximized)

        self.db_combo_box = QComboBox()
        self.db_combo_box.addItems(find_sqlite_files())
        self.db_combo_box.setEditable(True)
        self.db_combo_box.lineEdit().setPlaceholderText('Имя базы данных (sqlite)')
        self.db_combo_box.setCurrentIndex(-1)
        self.db_combo_box.setFixedWidth(220)

        self.all_dbs_area = AllDBsArea()
        self.all_dbs_area.db_deleted.connect(self.deleted_db_name)
        self.all_dbs_area.table_deleted.connect(self.deleted_table_name)

        ok_button = QPushButton('Записать')
        ok_button.clicked.connect(self.write_in_db)

        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.addWidget(self.db_combo_box, alignment=Qt.AlignHCenter)
        layout.addWidget(self.all_dbs_area)
        layout.addWidget(ok_button, alignment=Qt.AlignLeft)
        self.setLayout(layout)

    @Slot(str)
    def deleted_db_name(self, db_name: str):
        self.db_combo_box.removeItem([self.db_combo_box.itemText(i)
                                      for i in range(self.db_combo_box.count())
                                      ].index(db_name))
        self.db_deleted.emit(db_name)

    @Slot(str)
    def deleted_table_name(self, db_name: str, table_name: str):
        self.table_deleted.emit(db_name, table_name)

    def find_db_column(self, db_name: str):
        db_column_widget = self.all_dbs_area.findChildren(QWidget, db_name)
        db_column_layout = None
        if db_column_widget:
            db_column_layout = db_column_widget[0].findChild(DBColumn)
        return db_column_widget, db_column_layout

    def update_dbs_area(self, db_name: str):
        db_column_widget = self.all_dbs_area.findChildren(QWidget, db_name)
        if db_column_widget:
            db_column_layout = db_column_widget[0].findChild(DBColumn)
            tables_ui = [c.objectName() for c in db_column_widget[0].findChildren(QCheckBox)[1:] if c.isVisible()]
            # print(tables_ui)
            tables_db = Database(f'data/{db_name}', []).get_source_tables_names()
            # print(tables_db)
            for table_db in tables_db:
                if table_db not in tables_ui:
                    print(table_db)
                    db_column_layout.add_table(table_db, Table(table_db, f'data/{db_name}').get_n_rows())
        else:
            self.all_dbs_area.add_db_column(db_name)
            self.db_combo_box.addItem(db_name)

    def write_in_db(self):
        db_name_to_write = self.db_combo_box.currentText()

        for db_column in self.all_dbs_area.dbs_and_tables.parentWidget().children()[1:]:
            checkboxes = [check for check in db_column.findChildren(QCheckBox) if check.isChecked()]
            if checkboxes:
                db_name_to_read = checkboxes[0].text()
                for table_name in map(lambda x: x.objectName(), checkboxes[1:]):
                    table_to_write = Table(table_name, f'data/{db_name_to_write}')
                    table_to_write.create_table()
                    table_to_write.append_data(Table(table_name, f'data/{db_name_to_read}').get_data([]))

        self.update_dbs_area(db_name_to_write)
        self.written_db.emit(db_name_to_write)
