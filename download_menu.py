from PySide6.QtWidgets import (
    QMainWindow, QComboBox, QLineEdit, QScrollArea, QPushButton, QLabel, QMessageBox,
    QHBoxLayout, QVBoxLayout, QWidget, QSizePolicy, QProgressBar
)
from PySide6.QtCore import Qt, Signal, Slot, QSize
from PySide6.QtGui import QIcon, QIntValidator, QRegularExpressionValidator
from load_and_parse import ArxivOrg, ACM
from db_sqlite3 import find_sqlite_files, correct_queries_for_output


class AdvancedQueryRow(QHBoxLayout):
    removed_advanced_query = Signal(QWidget)

    def __init__(self):
        super().__init__()
        self.setSpacing(5)
        self_delete_button = QPushButton(icon=QIcon("icons/delete1.png"))
        self_delete_button.setFixedSize(QSize(25, 25))
        self_delete_button.clicked.connect(self.removed_advanced_query_row)
        self.addWidget(self_delete_button)
        logic_operand = QComboBox()
        logic_operand.addItems(['OR', 'AND', 'NOT'])
        logic_operand.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.addWidget(logic_operand)
        self.query = QLineEdit()
        self.query.setValidator(QRegularExpressionValidator(r'[\w+\s]*'))
        self.query.setFixedWidth(130)
        self.addWidget(self.query)

    def removed_advanced_query_row(self):
        self.removed_advanced_query.emit(self.parentWidget())
        self.query.setEnabled(False)


class QueryColumn(QVBoxLayout):

    def __init__(self):
        super().__init__()
        self.addChildLayout(AdvancedQueryRow())
        self.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        simple_query = QLineEdit()
        simple_query.setFixedWidth(217)
        simple_query.setPlaceholderText('Введите запрос')
        simple_query.setValidator(QRegularExpressionValidator(r'[\w+\s]*'))
        self.addWidget(simple_query, alignment=Qt.AlignHCenter)

    def add_advanced_query_row(self):
        widget_item = QWidget()
        widget_item.setLayout(AdvancedQueryRow())
        print(widget_item)
        widget_item.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))
        widget_item.layout().removed_advanced_query.connect(self.removed_advanced_query_row)
        self.addWidget(widget_item)

    @Slot(QWidget)
    def removed_advanced_query_row(self, widget: QWidget):
        self.removeWidget(widget)
        self.update()


class ClassQuery(QVBoxLayout):
    removed_class = Signal(QWidget)

    def __init__(self, class_num: int):
        super().__init__()
        self.addChildLayout(QueryColumn())
        self.class_num = class_num
        self.setSizeConstraint(self.SizeConstraint.SetMinAndMaxSize)
        self.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        t_layout = QHBoxLayout()
        t_layout.setAlignment(Qt.AlignHCenter)
        self_delete_button = QPushButton(icon=QIcon("icons/delete1.png"))
        self_delete_button.setFixedSize(QSize(25, 25))
        self_delete_button.clicked.connect(self.removed_class_column)
        t_layout.addWidget(self_delete_button)
        self.label = QLabel(f'Запрос {class_num}')
        t_layout.addWidget(self.label, alignment=Qt.AlignHCenter)
        self.addLayout(t_layout)
        layout = QHBoxLayout()
        layout.addWidget(QLabel('Библиотека'))
        self.elib_choice = QComboBox()
        self.elib_choice.addItems(['arXiv', 'ACM'])
        layout.addWidget(self.elib_choice)
        layout.setAlignment(Qt.AlignHCenter)
        self.addLayout(layout)
        widget = QWidget()
        widget.setLayout(QueryColumn())
        widget.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))
        self.addWidget(widget)
        add_advanced_row_button = QPushButton('Добавить')
        add_advanced_row_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        add_advanced_row_button.clicked.connect(widget.layout().add_advanced_query_row)
        add_advanced_row_button.clicked.connect(self.update)
        self.addWidget(add_advanced_row_button, alignment=Qt.AlignHCenter)

    def removed_class_column(self):
        self.removed_class.emit(self.parentWidget())
        self.setEnabled(False)

    def change_class_label(self, class_label: str):
        self.label.setText(class_label)
        self.update()


class ClassesRow(QHBoxLayout):

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        self.addChildLayout(ClassQuery(1))
        self.setSizeConstraint(self.SizeConstraint.SetMinAndMaxSize)
        # self.setSpacing(5)
        widget = QWidget()
        widget.setLayout(ClassQuery(self.count()+1))
        widget.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))
        widget.layout().removed_class.connect(widget.close)
        widget.layout().removed_class.connect(self.delete_class_query)
        self.addWidget(widget)
        self.add_class_query_button = QPushButton('Добавить')
        self.add_class_query_button.clicked.connect(self.add_class_query)
        self.add_class_query_button.clicked.connect(self.update)
        self.add_class_query_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.addWidget(self.add_class_query_button)

    def add_class_query(self):
        widget = QWidget()
        widget.setLayout(ClassQuery(self.count()))
        widget.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))
        widget.layout().removed_class.connect(widget.close)
        widget.layout().removed_class.connect(self.delete_class_query)
        widget.layout().removed_class.connect(self.update)
        self.insertWidget(self.count()-1, widget)

    @Slot(QWidget)
    def delete_class_query(self, class_widget: QWidget):
        self.removeWidget(class_widget)
        self.update()
        self.rename_classes()

    def rename_classes(self):
        for i in range(self.count()):
            labels = self.itemAt(i).widget().findChildren(ClassQuery)
            for label in labels:
                label.change_class_label(f'Запрос {i+1}')


class DownloadMenu(QVBoxLayout):
    add_new_db = Signal(str)

    def __init__(self):
        super().__init__()
        self.warning = None

        self.setSpacing(10)
        self.addWidget(QLabel('Загрузка статей из электронных библиотек'), alignment=Qt.AlignHCenter)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        my_widget = QWidget()
        my_widget.setLayout(ClassesRow())
        self.scroll.setWidget(my_widget)

        self.db_name = QComboBox()
        self.db_name.addItems(find_sqlite_files())
        self.db_name.setInsertPolicy(QComboBox.InsertPolicy.InsertAlphabetically)
        self.db_name.setEditable(True)
        self.db_name.setCurrentIndex(-1)
        self.db_name.lineEdit().setPlaceholderText('Имя базы данных (sqlite)')
        self.db_name.setFixedWidth(220)
        self.addWidget(self.db_name, alignment=Qt.AlignHCenter)

        self.num_of_articles_per_class = QLineEdit()
        self.num_of_articles_per_class.setPlaceholderText('Количество статей на запрос (-1 макс)')
        self.num_of_articles_per_class.setValidator(QIntValidator(-1, 9999))
        self.num_of_articles_per_class.setFixedWidth(220)
        self.addWidget(self.num_of_articles_per_class, alignment=Qt.AlignHCenter)

        layout = QHBoxLayout()
        layout.setSizeConstraint(layout.SizeConstraint.SetMinAndMaxSize)
        self.ok_button = QPushButton('Скачать и записать в базу данных')
        self.ok_button.clicked.connect(self.download_data)
        self.ok_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        layout.addWidget(self.ok_button, alignment=Qt.AlignLeft)
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedSize(250, 20)
        self.progress_bar.setHidden(True)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progress_bar, alignment=Qt.AlignRight)

        self.addWidget(self.scroll)
        self.addLayout(layout)

    def download_data(self):
        if not self.db_name.currentText():
            self.warning = QMessageBox(QMessageBox.Icon.Critical, 'Ошибка', 'Не задано имя базы данных')
            self.warning.show()
            return
        try:
            articles_per_query = int(self.num_of_articles_per_class.text())
        except ValueError:
            self.warning = QMessageBox(QMessageBox.Icon.Critical, 'Ошибка', 'Не задано количество статей')
            self.warning.show()
            return
        if articles_per_query < -1 or articles_per_query == 0:
            self.warning = QMessageBox(QMessageBox.Icon.Critical, 'Ошибка',
                                       'Количество статей: от 1 до 9999\n'
                                       '(-1 для загрузки всех статей в пределах 10000)')
            self.warning.show()
            return

        db_name = self.db_name.currentText()
        classes_row = self.scroll.findChildren(ClassesRow)
        # self.progress_bar.show()
        # self.progress_bar.setFormat('Запрос 1')
        # queries = list()
        classes_queries = classes_row[0].parentWidget().findChildren(ClassQuery)
        # print(classes_queries)
        input_queries = [class_query for class_query in classes_queries if class_query.isEnabled()]
        # print(input_queries)
        elibs_with_queries = {'arXiv': list(), 'ACM': list()}
        for i, input_query in enumerate(input_queries, 1):
            full_query = input_query.parentWidget().findChildren(QLineEdit)
            # print([query.isEnabled() for query in full_query])
            query_fields = [query.text() for query in full_query if query.isEnabled()]
            elib_and_queries_logic_operands = list(map(lambda x: x.currentText(),
                                                       input_query.parentWidget().findChildren(QComboBox)))
            # print(f"{query_fields}\n{elib_and_queries_logic_operands}\n")
            if '' in query_fields or not query_fields:
                message = f'Класс {i}\nОдно или несколько полей не заполнено'
                self.warning = QMessageBox(QMessageBox.Icon.Critical, 'Ошибка', message)
                self.warning.show()
                self.progress_bar.hide()
                return

            if len(query_fields) == 1:
                elibs_with_queries[elib_and_queries_logic_operands[0]].append(query_fields[0].strip().replace(" ", "+"))
            else:
                if elib_and_queries_logic_operands[0] == 'arXiv':
                    base_query = f'AND {query_fields[0].strip().replace(" ", "+")}'
                elif elib_and_queries_logic_operands[0] == 'ACM':
                    base_query = query_fields[0].strip().replace(" ", "+")
                advanced_query = ', '.join([f'{operand} {query.strip().replace(" ", "+")}'
                                            for operand, query in zip(elib_and_queries_logic_operands[1:],
                                                                      query_fields[1:])])
                elibs_with_queries[elib_and_queries_logic_operands[0]].append(', '.join([base_query, advanced_query]))

        print(elibs_with_queries)

        arxiv = ArxivOrg(elibs_with_queries['arXiv'], articles_per_query)
        # arxiv.max_progress_value.connect(self.progress_bar.setMaximum)
        # arxiv.progress_counter.connect(self.progress_bar.setValue)
        # arxiv.query_num_str.connect(self.progress_bar.setFormat)
        acm = ACM(elibs_with_queries['ACM'], articles_per_query)
        failed_queries_arxiv = arxiv.download_and_write_in_db(f'data/{db_name}')
        failed_queries_acm = acm.download_and_write_in_db(f'data/{db_name}')
        # print(failed_queries)
        # self.progress_bar.hide()
        if failed_queries_arxiv or failed_queries_acm:
            message = 'Не нашлось статей:'
            if failed_queries_arxiv:
                message += '\n'
                corrected_failed_queries = correct_queries_for_output(failed_queries_arxiv)
                output_failed_queries = ",\n".join(corrected_failed_queries)
                message += f'\n{output_failed_queries}'
            if failed_queries_acm:
                message += '\n'
                corrected_failed_queries = correct_queries_for_output(failed_queries_acm)
                output_failed_queries = ",\n".join(corrected_failed_queries)
                message += f'\n{output_failed_queries}'
            self.warning = QMessageBox(QMessageBox.Icon.Warning, 'Предупрежение', message)
            self.warning.show()
        if db_name not in [self.db_name.itemText(i) for i in range(self.db_name.count())]:
            self.db_name.addItem(db_name)
        self.add_new_db.emit(db_name)


class DownloadWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 480)
        self.setMaximumSize(1280, 720)
        self.setWindowFlags(Qt.MSWindowsOwnDC)
        widget = QWidget()
        widget.setLayout(DownloadMenu())
        self.setCentralWidget(widget)
