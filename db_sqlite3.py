import sqlite3
from typing import List, Tuple, Dict, Union
from collections import Counter
import os
import random


CURRENT_DB_NAME = "data/test_articles_3.sqlite3"


TABLE_FIELDS_TRANSLATED = {'url': 'Адрес',
                           'tags_codes': 'Теги\n(кратко)',
                           'tags_names': 'Теги',
                           'authors': 'Авторы',
                           'title': 'Название',
                           'abstract': 'Описание',
                           'submitted': 'Дата'}


TABLE_FIELDS = (
    "url",
    "title",
    "tags_codes",
    "tags_names",
    "authors",
    "abstract",
    "submitted"
)


def find_sqlite_files(path="data") -> List[str]:
    """
    Поиск .sqlite
    :param path: Путь к каталогу. Если "", то поиск в каталоге с программой
    :return: список имён .dat файлов
    """
    files_names = list()
    for file in os.listdir(path):
        if file.endswith('.sqlite') or file.endswith('.sqlite3'):
            files_names.append(file)
    return files_names


def correct_query_for_output(query: str) -> str:
    output_query = query.replace('+', ' ')
    output_query = output_query.title().replace('_Arxiv', ' (arXiv)').replace('_Acm', ' (ACM)')
    if output_query.startswith('And ', 0, 4):
        output_query = output_query[4:].replace('And', 'AND').replace('Or', 'OR').replace('Not', 'NOT')
    return output_query


def correct_queries_for_output(queries: List[str]) -> List[str]:
    corrected_queries = list()
    for query in queries:
        corrected_queries.append(correct_query_for_output(query))
    return corrected_queries


class Table:
    """Класс для работы с таблицей базы данных"""
    def __init__(self, table_name: str, db_name=CURRENT_DB_NAME):
        self.__db_name = db_name
        self.__table_name = table_name
        self.all_columns = TABLE_FIELDS

    def create_table(self) -> None:
        """Создание таблицы в бд по запросу"""
        sql = f"CREATE TABLE IF NOT EXISTS '{self.__table_name}'(\n" \
              f"    url TEXT,\n" \
              f"    title TEXT UNIQUE,\n" \
              f"    tags_codes TEXT,\n" \
              f"    tags_names TEXT,\n" \
              f"    authors TEXT,\n" \
              f"    abstract TEXT,\n" \
              f"    submitted TEXT\n" \
              f"    )"
        # print(os.getcwd())
        with sqlite3.connect(self.__db_name) as con:
            cur = con.cursor()
            cur.execute(sql)
            con.commit()

    def create_selection_table(self):
        sql = f"CREATE TABLE IF NOT EXISTS '{self.__table_name}'(\n" \
              f"    url TEXT,\n" \
              f"    title TEXT,\n" \
              f"    tags_codes TEXT,\n" \
              f"    tags_names TEXT,\n" \
              f"    authors TEXT,\n" \
              f"    abstract TEXT,\n" \
              f"    submitted TEXT\n" \
              f"    )"
        with sqlite3.connect(self.__db_name) as con:
            cur = con.cursor()
            cur.execute(sql)
            con.commit()

    def get_n_rows(self):
        """Количество статей в таблице"""
        sql = f"SELECT url FROM '{self.__table_name}'"
        with sqlite3.connect(self.__db_name) as con:
            cur = con.cursor()
            data = cur.execute(sql).fetchall()
        return len(data)

    def delete_table(self):
        """Удаление таблицы"""
        sql = f"DROP TABLE IF EXISTS '{self.__table_name}'"
        # print(self.__table_name)
        with sqlite3.connect(self.__db_name) as con:
            cur = con.cursor()
            cur.execute(sql)
            con.commit()

    def append_data(self, articles: List[Tuple[str]]) -> None:
        """Заполнение таблицы"""
        sql = f"INSERT INTO '{self.__table_name}' " \
              f"(url, title, tags_codes, tags_names, authors, abstract, submitted) VALUES (?, ?, ?, ?, ?, ?, ?)"
        with sqlite3.connect(self.__db_name) as con:
            cur = con.cursor()
            for article in articles:
                try:
                    cur.execute(sql, article)
                except sqlite3.IntegrityError:
                    print('duplicate')
                    pass
            con.commit()

    def get_data(self, columns: List[str], n_rows=-1) -> List[tuple]:
        """
        Получение выборочных столбцов n_rows статей для запроса
        :param columns: Столбцы бд
        :param n_rows: Количество получаемых строк
        :return: Список из значений
        """
        cols = [col for col in columns if col in self.all_columns]
        if not columns:
            cols = self.all_columns
        sql = f"SELECT {', '.join(cols)} FROM '{self.__table_name}'"
        row_count = self.get_n_rows()
        if n_rows == -1:
            n_rows = row_count
        if n_rows > row_count:
            n_rows = row_count
        with sqlite3.connect(self.__db_name) as con:
            cur = con.cursor()
            data = cur.execute(sql).fetchall()[:n_rows]
        return data

    def get_article(self, columns: List[str], url: str):
        cols = [col for col in columns if col in self.all_columns]
        if not columns:
            cols = self.all_columns
        sql = f"SELECT {', '.join(cols)} FROM '{self.__table_name}' WHERE url = '{url}'"
        # print(sql)
        with sqlite3.connect(self.__db_name) as con:
            cur = con.cursor()
            article = cur.execute(sql).fetchall()
        try:
            return article[0]
        except IndexError:
            return None

    def delete_article(self, url: str):
        # print(url)
        # print(self.get_article([], url))
        sql = f"DELETE FROM '{self.__table_name}' WHERE url = '{url}'"
        with sqlite3.connect(self.__db_name) as con:
            cur = con.cursor()
            cur.execute(sql)
            con.commit()

    def get_minmax_date(self) -> Tuple[str, str]:
        sql = f"SELECT min(submitted), max(submitted) FROM '{self.__table_name}'"
        with sqlite3.connect(self.__db_name) as con:
            cur = con.cursor()
            minmax_dates = cur.execute(sql).fetchall()
        # print(minmax_dates)
        return minmax_dates[0]


class Database:
    """
    Класс для создания выборки из базы данных
    """
    def __init__(self, db_name: str, requested_tables: List[str]):
        """
        Задание базы данных, если её не существует, то создаётся автоматически\n
        :param db_name: Имя базы данных
        :param requested_tables: Запрашиваемые таблицы, если пустой список, то все таблицы
        """
        self.__db_name = db_name
        # print(self.get_tables_names())
        # print(self.get_selectable_tables_names())
        self.__requested_tables = requested_tables
        if requested_tables:
            tables = self.get_all_tables_names()
            for requested_table in requested_tables:
                if requested_table not in tables:
                    raise ValueError(f'{requested_table} нельзя выбрать, можете выбрать:\n{tables}')
        else:
            self.__requested_tables = self.get_all_tables_names()
        # print(self.__requested_tables)
        self.all_columns = TABLE_FIELDS

    def set_requested_tables(self, requested_tables: List[str]):
        """Задание используемых таблиц"""
        self.__requested_tables = requested_tables

    def delete_database(self):
        """Удаление файла с бд"""
        os.remove(self.__db_name)

    def get_all_tables_names(self) -> List[str]:
        """Имена таблиц в базе данных"""
        sql = f"SELECT name FROM sqlite_master WHERE type = 'table'"
        # print(self.__db_name)
        with sqlite3.connect(self.__db_name) as con:
            cur = con.cursor()
            tables_names = list(map(lambda x: x[0], cur.execute(sql).fetchall()))
        return tables_names

    def find_article_by_url(self, url: str, fields_for_output: List[str]) -> tuple:
        # print(self.__db_name)
        for table_name in self.get_all_tables_names():
            article = Table(table_name, self.__db_name).get_article(fields_for_output, url)
            if article:
                return article

    def get_source_tables_names(self) -> List[str]:
        """Имена таблиц, которые могут участвовать в формировании выборки"""
        return [table_name for table_name in self.__requested_tables
                if table_name.endswith('_arXiv') or table_name.endswith('_ACM')]

    def get_selection_tables_names(self) -> List[str]:
        """Имена таблиц, которые являются выборками"""
        return [table_name for table_name in self.__requested_tables
                if not (table_name.endswith('_arXiv') or table_name.endswith('_ACM'))]  # '_arXiv'

    def get_source_tables_info(self) -> Dict[str, int]:
        """Количество статей в каждой таблице"""
        return {table: Table(table, self.__db_name).get_n_rows() for table in self.get_source_tables_names()}

    def get_min_n_rows(self) -> int:
        """Минимальное количество статей среди всех запросов (для сбалансированности выборки)"""
        return min([Table(table, self.__db_name).get_n_rows() for table in self.__requested_tables])

    def get_all_articles(self) -> Dict[str, List[Tuple[str, str]]]:
        """
        Получение сбалансированной выборки (возможно наличие дубликатов)
        :param n_per_table: Количество статей из каждой таблицы
        :return: Список статей, отнесённый к каждой таблице
        """
        titles_list = list()
        all_articles = {table: Table(table, self.__db_name).get_data(['url', 'title', 'tags_names', 'abstract'])
                        for table in self.__requested_tables}
        articles_without_duplicates = {table: list() for table in all_articles.keys()}
        for table, articles in sorted(all_articles.copy().items(), key=lambda x: len(x[1])):
            for article in articles:
                if article[1] not in titles_list:
                    titles_list.append(article[1])
                    articles_without_duplicates[table].append(article)
        return articles_without_duplicates

    def get_selection_articles(self, classes_with_tables: Dict[str, List[str]], n_per_class=-1, random_state=-1):
        print(f'random state = {random_state}')
        all_articles = self.get_all_articles()
        selection_articles = {key: list() for key in classes_with_tables.keys()}
        for class_name, tables in classes_with_tables.items():
            # if len(tables) == 1:
            #     selection_articles[class_name].extend(all_articles[tables[0]])
            # else:
            #     for i in range(max([len(all_articles[table]) for table in tables])):
            #         for table in tables:
            #             try:
            #                 selection_articles[class_name].append(all_articles[table][i])
            #             except IndexError:
            #                 tables.remove(table)
            for table in sorted(tables, key=lambda x: len(all_articles[x]), reverse=True):
                print(table)
                selection_articles[class_name].extend(all_articles[table])

        return {class_name: articles for class_name, articles in selection_articles.items()}
        # min_n_rows = min([len(articles) for _, articles in selection_articles.items()])
        # if n_per_class == -1:
        #     n_per_class = min_n_rows
        # class_size = min(n_per_class, min_n_rows)

        # if random_state == -1:
        #     # print({class_name: articles[:class_size] for class_name, articles in selection_articles.items()})
        #     return {class_name: articles[:class_size] for class_name, articles in selection_articles.items()}
        # else:
        #     random.seed(random_state)
        #     return {class_name: random.sample(articles, class_size)
        #             for class_name, articles in selection_articles.items()}

    def get_info_for_duplicates_table(self) -> Dict[str, Dict[str, int]]:
        all_articles = {table: Table(table, self.__db_name).get_data(['title'])
                        for table in self.__requested_tables}
        duplicates_table = {key: dict().fromkeys(all_articles) for key in all_articles.keys()}
        for table_row, articles_row in sorted(all_articles.items(), key=lambda x: len(x[1]), reverse=True):
            for table_col, articles_col in sorted(all_articles.items(), key=lambda x: len(x[1]), reverse=True):
                duplicates_table[table_row][table_col] = len(set(articles_row).intersection(articles_col))
        return duplicates_table

    def get_selection_info(self) -> Dict[str, Dict[str, Union[Counter, tuple, int]]]:
        info = dict()
        for table_name in self.__requested_tables:
            tags = list()
            table = Table(table_name, self.__db_name)
            info[table_name] = {'tags': Counter(), 'minmax dates': tuple(), 'num of articles': int()}
            for article_tags in map(lambda x: x[0], table.get_data(['tags_names'])):
                tags.extend(article_tags.split('; '))
            info[table_name]['tags'].update(tags)
            info[table_name]['minmax dates'] = table.get_minmax_date()
            info[table_name]['num of articles'] = table.get_n_rows()
        return info

    def make_selection_table(self, selection_name: str, class_size: int) -> int:
        """
        Формирование сбалансированной выборки в бд
        """
        min_n_rows = self.get_min_n_rows()
        if class_size > min_n_rows:
            class_size = min_n_rows
        Table(selection_name, self.__db_name).create_selection_table()
        titles = list()
        num_of_duplicates = 0

        for table_name in self.__requested_tables:
            sql = f"INSERT INTO '{selection_name}' " \
                  f"(url, title, tags_codes, tags_names, authors, abstract, submitted) VALUES " \
                  f"(?, ?, ?, ?, ?, ?, ?)"
            i = 0
            table = Table(table_name, self.__db_name)
            with sqlite3.connect(self.__db_name) as con:
                cur = con.cursor()
                for article in table.get_data(list(table.all_columns), table.get_n_rows()):
                    if i == class_size:
                        break
                    if article[1] not in titles:
                        titles.append(article[1])
                    else:
                        num_of_duplicates += 1
                        continue
                    cur.execute(sql, article)
                    i += 1
            if i < class_size:
                raise ValueError(f"Попробуйте class_size = {i}")
        return num_of_duplicates
