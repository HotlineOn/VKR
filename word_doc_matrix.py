import os
import pickle
import re
import string
# import numba
# from numba import cuda, vectorize
import numpy as np

from collections import Counter
from math import log10, sqrt
from typing import List, Tuple, Dict, Union
from pandas import DataFrame
from PySide6.QtCore import Signal, QObject, QThread

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


DEFAULT_STOPWORDS = stopwords.words('english') + ['using', 'used', 'also', 'use', 'one', 'two', 'three', 'approach', 'task']


def find_dat_files(path="data") -> List[str]:  # path: str
    """
    Поиск .dat файлов в текущем каталоге (os.chdir())
    :param path: Путь к каталогу. Если "", то поиск в каталоге с программой
    :return: список имён .dat файлов
    """
    files = list()
    for file in os.listdir(path):
        if file.endswith('.dat'):
            files.append(file)
    return files


def get_info_from_name_of_dat_file(file_name: str) -> Tuple[str, int, str]:
    """
    Получение необходимой информации из имени файла с датафреймом
    :param file_name: имя файла или путь
    :return: Кортеж из названия таблицы с выборкой, метода взвешивания и объёмом каждого класса
    """
    class_size_start = file_name.rfind('_')
    class_size = int(file_name[class_size_start+1:file_name.rfind('.')])
    weight_method_start = file_name.rfind('_', 0, class_size_start)
    weight_method = file_name[weight_method_start+1:class_size_start]
    selection_name = file_name[:weight_method_start]
    return selection_name, class_size, weight_method


class BagOfWords(QObject):
    """
    Класс для формирования мешка слов
    """
    progress_current_action = Signal(str)
    progress_max = Signal(int)
    progress_current_i = Signal(int)

    def __init__(self, stop_words: List[str]):
        super().__init__()
        self.__spec_chars = string.punctuation + string.digits + '\n'
        self.__lemmatizer = WordNetLemmatizer()
        self.__stops = stop_words

    def get_article_words_list(self, article: str) -> List[str]:
        """
        :param article: Статья с полями 'название' и 'описание'
        :return: Список слов в статье
        """
        raw_text = re.sub(r'\$\S*\$', '', article).lower()
        raw_text = ''.join([ch for ch in raw_text if ch not in self.__spec_chars])
        words = [self.__lemmatizer.lemmatize(word) for word in word_tokenize(raw_text)]
        return [word for word in words if word not in self.__stops and len(word) > 1]

    def make_bag_of_words(self, articles: List[str]) -> List[Counter]:
        """
        :param articles: Статьи с полями 'название' и 'описание'
        :return: Мешок слов, нулевой член которого счётчик всех слов
        """
        num_of_articles = len(articles)
        self.emit_progress_max(num_of_articles)
        self.emit_current_action('Формирование мешка слов')
        bag = [Counter() for _ in range(num_of_articles + 1)]
        for i, article in enumerate(articles, 1):
            self.emit_current_i(i)
            words = self.get_article_words_list(article)
            bag[i].update(words)
            bag[0].update(words)
        self.emit_current_action('Удаление одночастотных слов')
        self.emit_progress_max(len(list(filter(lambda x: x[1] <= 1, bag[0].items()))))
        for i, (word, _) in enumerate(filter(lambda x: x[1] <= 1, bag[0].copy().items()), 1):
            self.emit_current_i(i)
            bag[0].pop(word)
            for doc in bag:
                if word in doc:
                    doc.pop(word)
        return bag

    def emit_current_i(self, i: int):
        self.progress_current_i.emit(i)

    def emit_progress_max(self, maximum: int):
        self.progress_max.emit(maximum)

    def emit_current_action(self, current_action: str):
        self.progress_current_action.emit(current_action)


class WordDocMatrix(QObject):
    progress_current_action = Signal(str)
    progress_max = Signal(int)
    progress_current_i = Signal(int)

    def __init__(self, weight_method='tfc'):
        super().__init__()
        weight_methods = ('tf', 'tfidf', 'tfc')
        if weight_method not in weight_methods:
            raise ValueError(f'{weight_method} not in {", ".join(weight_methods)}')
        self.weight_method = weight_method

    @staticmethod
    def __num_of_docs_where_words_matched(list_of_documents) -> List[float]:
        """Подсчёт документов, в которых находится слово для каждого слова"""
        word_num = [0.0 for _ in range(len(list_of_documents[-1]))]
        for document in list_of_documents:
            for word, _ in filter(lambda x: x[1] > 0, enumerate(document)):
                word_num[word] += 1.0
        return word_num

    @staticmethod
    def tf(document):
        sum_val = sum(document)
        for word, _ in filter(lambda x: x[1] > 0, enumerate(document)):
            document[word] /= sum_val

    @staticmethod
    def tfidf(document, word_num, num_of_documents):
        # num_of_documents = len(self.list_of_documents)
        # word_num = self.__num_of_docs_where_words_matched()
        # for i, document in enumerate(self.list_of_documents, 1):
        sum_val = sum(document)
        for word, _ in filter(lambda x: x[1] > 0, enumerate(document)):
            document[word] *= log10(num_of_documents / word_num[word]) / sum_val

    @staticmethod
    def tfc(document, word_num, num_of_documents):
        sum_val = sum(document)
        norm = sqrt(sum([(count / sum_val * log10(num_of_documents / word_num[word])) ** 2
                         for word, count in enumerate(document) if count > 0]))
        for word, _ in filter(lambda x: x[1] > 0, enumerate(document)):
            document[word] *= log10(num_of_documents / word_num[word]) / norm / sum_val

    def make_matrix(self, bag_of_words: List[Counter]):
        """
        :param bag_of_words: мешок слов
        :return: Матрица документ-термин в виде списка из списков дробных значений
        """
        self.emit_progress_max(len(bag_of_words))
        self.emit_current_action('Формирование матрицы')
        all_words = bag_of_words.pop(0)
        # for word, _ in filter(lambda x: x[1] <= 1, all_words.items()):
        #     print(word)
        list_of_documents = list()
        for i, document in enumerate(bag_of_words, 1):
            self.emit_current_i(i)
            counter_matrix = dict().fromkeys(all_words, 0.0)
            counter_matrix.update(dict(document))
            list_of_documents.append([float(val) for _, val in counter_matrix.items()])
        return list_of_documents

    def compute_weighted_matrix(self, bag: List[Counter]):
        list_of_documents = self.make_matrix(bag)
        self.emit_current_action('Взвешивание матрицы')
        if self.weight_method == 'tf':
            for i, document in enumerate(list_of_documents, 1):
                self.tf(document)
                self.emit_current_i(i)
        else:
            word_num = self.__num_of_docs_where_words_matched(list_of_documents)
            num_of_documents = len(list_of_documents)
            if self.weight_method == 'tfidf':
                for i, document in enumerate(list_of_documents, 1):
                    self.tfidf(document, word_num, num_of_documents)
                    self.emit_current_i(i)
            else:
                for i, document in enumerate(list_of_documents, 1):
                    self.tfc(document, word_num, num_of_documents)
                    self.emit_current_i(i)
        return list_of_documents

    def emit_current_i(self, i: int):
        self.progress_current_i.emit(i)

    def emit_progress_max(self, maximum: int):
        self.progress_max.emit(maximum)

    def emit_current_action(self, current_action: str):
        self.progress_current_action.emit(current_action)


# @vectorize(['void(float32[:], float32[:])'], target='cuda')
# def tf_cuda(list_of_documents):
#     for i in
#     sum_val = sum(filter(lambda x: x > 0, document))
#     for word, _ in filter(lambda x: x[1] > 0, enumerate(document)):
#         document[word] /= sum_val


class WordDocMatrixNP(QObject):
    progress_current_action = Signal(str)
    progress_max = Signal(int)
    progress_current_i = Signal(int)

    def __init__(self, weight_method='tfc'):
        super().__init__()
        weight_methods = ('tf', 'tfidf', 'tfc')
        if weight_method not in weight_methods:
            raise ValueError(f'{weight_method} not in {", ".join(weight_methods)}')
        self.weight_method = weight_method

    @staticmethod
    def __num_of_docs_where_words_matched(list_of_documents: np.ndarray) -> np.ndarray:
        """Подсчёт документов, в которых находится слово для каждого слова"""
        word_num = np.zeros(shape=list_of_documents.shape[1], dtype=np.float32)
        for document in list_of_documents:
            for i in range(len(document)):
                if document[i] > 0:
                    word_num[i] += 1
        return word_num

    def make_matrix(self, bag_of_words: List[Counter]):
        all_words = bag_of_words.pop(0)
        list_of_documents = np.zeros(shape=(len(bag_of_words), len(all_words)), dtype=np.float32)

        self.emit_progress_max(len(bag_of_words) - 1)
        self.emit_current_action('Формирование матрицы')
        for i, document in enumerate(bag_of_words):
            self.emit_current_i(i)
            matrix_row = dict().fromkeys(all_words.keys(), 0.0)
            matrix_row.update(dict(document))
            for j, (_, value) in enumerate(matrix_row.items()):
                list_of_documents[i, j] = value
        return list_of_documents

    def compute_weighted_matrix(self, bag: List[Counter]):
        list_of_documents = self.make_matrix(bag)
        # if self.weight_method == 'tf':
        #     for i, document in enumerate(list_of_documents, 1):
        #         self.tf(document)
        #         self.emit_current_i(i)
        # else:
        #     word_num = self.__num_of_docs_where_words_matched()
        #     num_of_documents = len(self.list_of_documents)
        #     if self.weight_method == 'tfidf':
        #         for i, document in enumerate(self.list_of_documents, 1):
        #             self.tfidf(document, word_num, num_of_documents)
        #             self.emit_current_i(i)
        #     else:
        #         for i, document in enumerate(self.list_of_documents, 1):
        #             self.tfc(document, word_num, num_of_documents)
        #             self.emit_current_i(i)
        return list_of_documents

    def emit_current_i(self, i: int):
        self.progress_current_i.emit(i)

    def emit_progress_max(self, maximum: int):
        self.progress_max.emit(maximum)

    def emit_current_action(self, current_action: str):
        self.progress_current_action.emit(current_action)


class SelectionMatrix(QObject):
    progress_current_action = Signal(str)
    progress_max = Signal(int)
    progress_current_i = Signal(int)
    process_finished = Signal()
    
    """
    Класс для формирования и испоьзования матрицы в виде датафрейма из сбалансированной выборки
    """
    def __init__(self, db_name: str, selection_name: str, class_size: int, stop_words: List[str], weight_method='tfc'):
        """
        Все параметры фигурируют в названиях сохраняемых файлов датафреймов
        :param selection_name: Имя таблицы с выборкой
        :param class_size: Объём статей в каждом классе
        :param weight_method: Метод взвешивания
        """
        super().__init__()
        self.db_name = db_name
        self.selection_name = selection_name
        self.class_size = class_size
        self.classes_names = list()
        self.weight_methods = ('tf', 'tfidf', 'tfc')
        if weight_method not in self.weight_methods:
            raise ValueError(f"{weight_method} не используется")
        self.weight_method = weight_method
        self.df = None
        self.stop_words = stop_words
        self.deleted_words = list()
        self.most_common_words = list()
        self.words_count = int()

        self.__matrix = WordDocMatrix(self.weight_method)
        self.__matrix.progress_current_i.connect(self.emit_current_i)
        self.__matrix.progress_current_action.connect(self.emit_current_action)
        self.__matrix.progress_max.connect(self.emit_progress_max)

        self.__bag_of_words = BagOfWords(self.stop_words)
        self.__bag_of_words.progress_current_i.connect(self.emit_current_i)
        self.__bag_of_words.progress_current_action.connect(self.emit_current_action)
        self.__bag_of_words.progress_max.connect(self.emit_progress_max)

    def split_dataframe_by_classes(self, class_size=-1):  # , with_save=False,  -> List[DataFrame]
        """
        Формирование датафреймов по классам
        :return: Список датафреймов
        """
        if class_size == -1:
            class_size = self.class_size
        num_of_classes = len(self.classes_names)
        dfs = list()
        for i in range(num_of_classes):
            class_df = self.df.iloc[(i*self.class_size):(i*self.class_size+class_size)]
            dfs.append(class_df)
        # print(dfs)
        return dfs

    def compute_dataframe(self, all_articles: Dict[str, List[Tuple[str, str]]], with_save=False):
        """
        Вычисление матрицы документ-термин,
        :param all_articles: Статьи для вычисления
        :param with_save: Сохранение датафрейма в бинарный файл
        :return: None
        """
        # print(self.class_size)
        urls = list()
        classes_names = list()
        articles_for_bag = list()
        for table, articles in all_articles.items():
            # print(table, len(articles))
            classes_names.append(table)
            for article in articles:
                urls.append(article[0])
                articles_for_bag.append(' '.join(article[1:]))
        print(len(articles_for_bag))

        self.classes_names = classes_names
        if self.class_size == -1:
            self.class_size = int(len(articles_for_bag) / len(classes_names))

        bag = self.__bag_of_words.make_bag_of_words(articles_for_bag)
        self.most_common_words = bag[0].most_common(100)
        self.words_count = sum([count for _, count in bag[0].items()])
        all_words = list(bag[0])

        weighted_matrix = self.__matrix.compute_weighted_matrix(bag)
        self.df = DataFrame(weighted_matrix, index=urls, columns=all_words)
        if with_save:
            self.to_pickle()
        self.emit_process_finished()

    def to_pickle(self):
        self.df.attrs['db_name'] = self.db_name
        self.df.attrs['classes_names'] = self.classes_names
        self.df.attrs['deleted_words'] = self.deleted_words
        self.df.attrs['most_common_words'] = self.most_common_words
        self.df.attrs['words_count'] = self.words_count
        self.df.attrs['stop_words'] = self.stop_words
        if not self.deleted_words:
            file_name = f'{self.selection_name}_{self.weight_method}_{self.class_size}.dat'
            self.df.to_pickle(f'data/{file_name}')
            return file_name
        else:
            addition_name = 'reducedWords'
            file_name = f'{self.selection_name}_{addition_name}_{self.weight_method}_{self.class_size}.dat'
            self.df.to_pickle(f'data/{file_name}')
            return file_name

    def get_dataframe_from_file(self):
        """
        Загрузка матрицы из бинарного файла
        :return: SelectionMatrix
        """
        file_name = f'data/{self.selection_name}_{self.weight_method}_{self.class_size}.dat'
        if not os.path.exists(file_name):
            raise ValueError(f"Файла с именем {file_name} не существует")
        with open(file_name, 'rb') as f:
            self.df = pickle.load(f)
        self.db_name = self.df.attrs['db_name']
        self.classes_names = self.df.attrs['classes_names']
        self.deleted_words = self.df.attrs['deleted_words']
        self.most_common_words = self.df.attrs['most_common_words']
        self.words_count = self.df.attrs['words_count']
        self.stop_words = self.df.attrs['stop_words']

    def emit_current_i(self, i: int):
        self.progress_current_i.emit(i)

    def emit_progress_max(self, maximum: int):
        self.progress_max.emit(maximum)

    def emit_current_action(self, current_action: str):
        self.progress_current_action.emit(current_action)

    def emit_process_finished(self):
        self.process_finished.emit()

    # def run(self, all_articles: Dict[str, List[Tuple[str, str]]], with_save=False):
    #     self.compute_dataframe(all_articles, with_save)


class Worker(QThread):
    """Объект для взвешивания матрицы"""
    progress_current_action = Signal(str)
    progress_max = Signal(int)
    progress_current_i = Signal(int)
    process_finished = Signal(object)
    calculation_finished = Signal(bool)

    def __init__(self, db_name: str, selection_name: str, articles_per_class: int, weight_method: str,
                 all_articles: Dict[str, List[Tuple[str, str]]], stop_words, with_save=False):
        super().__init__()
        self.matrix = SelectionMatrix(db_name, selection_name, articles_per_class, stop_words.copy(), weight_method)
        self.matrix.progress_current_i.connect(self.emit_current_i)
        self.matrix.progress_max.connect(self.progress_max)
        self.matrix.progress_current_action.connect(self.emit_current_action)
        # self.matrix.moveToThread(self)
        self.articles = all_articles.copy()
        self.with_save = with_save

    def run(self):
        # print(f'thread {self.currentThread()} is running...')
        self.matrix.compute_dataframe(self.articles, self.with_save)
        self.emit_process_finished()
        # return self.matrix

    def start_work(self):
        self.start()
        self.exec()

    def emit_current_i(self, i: int):
        self.progress_current_i.emit(i)

    def emit_progress_max(self, maximum: int):
        self.progress_max.emit(maximum)

    def emit_current_action(self, current_action: str):
        self.progress_current_action.emit(current_action)

    def emit_process_finished(self):
        self.calculation_finished.emit(self.with_save)
        self.process_finished.emit(self.matrix)
        self.disconnect(self.matrix)
        del self.matrix, self.articles, self.with_save
        self.quit()
        self.deleteLater()
