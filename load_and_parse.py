from calendar import month_name
from concurrent.futures import ThreadPoolExecutor
from math import ceil
from typing import List, Tuple
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from db_sqlite3 import Table, CURRENT_DB_NAME
from PySide6.QtCore import QObject, Signal
from time import sleep


def download_page(url: str) -> str:
    """Скачивание html кода страницы"""
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:45.0) Gecko/20100101 Firefox/45.0'}
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('https://', adapter)
    print(url)
    response = session.get(url, headers=headers)
    if response.status_code != 200:
        raise ValueError(f'Статус код: {response.status_code}\nПлохой url: {url}')
    return response.text


def make_typed_date(submitted: str) -> str:
    """YYYY-MM-DD"""
    date_list = submitted.split(' ')
    typed_date_list = list()
    typed_date_list.append(date_list[-1])
    month = str(list(month_name).index(date_list[1]))
    for i in (month, date_list[0]):
        if len(i) == 1:
            i = "0" + i
        typed_date_list.append(i)
    return '-'.join(typed_date_list)


class ArxivOrg(QObject):
    query_num_str = Signal(str)
    progress_counter = Signal(int)
    max_progress_value = Signal(int)

    def __init__(self, queries_list: List[str], n_for_each_query: int):
        super().__init__()
        self.m_pages_counter = 0
        self.__n_for_each_query = n_for_each_query
        self.__articles_per_page = 200
        if 0 < n_for_each_query <= 50:
            self.__articles_per_page = 50
        elif 0 < n_for_each_query <= 100:
            self.__articles_per_page = 100
        elif not (0 < n_for_each_query <= 10000 or n_for_each_query == -1):
            raise ValueError('некорректное n для каждого класса')
        self.__queries = list(set(queries_list))

    def set_queries(self, queries_list: List[str]):
        self.__queries = queries_list

    def set_n_for_each_query(self, n_for_each_query):
        self.__n_for_each_query = n_for_each_query

    def __download_page(self, url: str):
        html = download_page(url)
        self.m_pages_counter += 1
        self.progress_counter.emit(self.m_pages_counter)
        return html

    def __make_simple_url(self, query: str) -> str:
        """Формирование простого запроса для сайта arxiv.org"""
        return f"https://arxiv.org/search/cs?query={query}" \
               f"&searchtype=abstract&source=header&order=" \
               f"&date-filter_by=date_range&date-year=&date-from_date=2010&date-to_date=2022" \
               f"&size={self.__articles_per_page}&abstracts=show&date-date_type=submitted_date"

    def __make_advanced_url(self, query: str) -> str:
        """Формирование сложного запроса для сайта arxiv.org"""
        url_base = f"https://arxiv.org/search/advanced?advanced=&classification-computer_science=y" \
                   f"&classification-physics_archives=all&classification-include_cross_list=include" \
                   f"&date-filter_by=date_range&date-year=&date-from_date=2010&date-to_date=2022&" \
                   f"date-date_type=submitted_date" \
                   f"&abstracts=show&size={self.__articles_per_page}&order="
        query_params = ("operator", "term")
        url_query = str()
        for i, expr in enumerate(query.split(',')):
            # print(expr)
            for j, word in enumerate(expr.strip().split(' ')):
                # print(word)
                url_query += f"&terms-{i}-{query_params[j]}={word}"
            url_query += f"&terms-{i}-field=abstract"
        return url_base + url_query

    def __make_simple_or_advanced_url(self, query: str):
        """Автоматический подбор сложности запроса"""
        if len(query.split(', ')) > 1:
            return self.__make_advanced_url(query)
        else:
            return self.__make_simple_url(query)

    def __trial_request(self, query: str) -> Tuple[int, str]:
        """
        Максимальное количество страниц для запроса на сайтеи и первая страница
        :param query: Запрос для получения статей
        :return: Количество статей для запроса, а также страница с статьями
        """
        url = self.__make_simple_or_advanced_url(query)
        html = self.__download_page(url)
        soup = BeautifulSoup(html, 'lxml')
        str_with_num = soup.find('h1', {'class': 'title'}).text
        try:
            total_num_of_articles = int(str_with_num.split()[3].replace(',', '').replace(';', ''))
        except IndexError:
            # print(str_with_num.split())
            raise ValueError('Некорректный запрос')
        return total_num_of_articles, html

    # def make_trial_request(self, query: str):
    #     """Для проверки соединения и получения данных о количестве статей"""

    def __download_pages(self, query: str) -> List[str]:
        """Скачивание html страниц с сайта arxiv.org"""
        trial_request = self.__trial_request(query)
        print(trial_request[0])
        if trial_request[0] == 0:
            print('Что-то пошло не так')
            raise requests.exceptions.ConnectionError('Что-то пошло не так)')
        if trial_request[0] <= 200 or 0 < self.__n_for_each_query <= 200:
            if trial_request[0] <= 50 or self.__n_for_each_query <= 50:
                self.__articles_per_page = 50
            elif trial_request[0] <= 100 or self.__n_for_each_query <= 100:
                self.__articles_per_page = 100
            elif trial_request[0] > 100 or self.__n_for_each_query > 100:
                self.__articles_per_page = 200
            self.max_progress_value.emit(1)
            return [trial_request[1]]
        else:
            num_of_articles_to_download = self.__n_for_each_query
            if num_of_articles_to_download == -1:
                if trial_request[0] > 10000:
                    num_of_articles_to_download = 10000
                else:
                    num_of_articles_to_download = trial_request[0]
            if trial_request[0] < num_of_articles_to_download:
                num_of_articles_to_download = trial_request[0]
            print(num_of_articles_to_download)

            basic_url = self.__make_simple_or_advanced_url(query)
            num_of_pages = ceil(num_of_articles_to_download / self.__articles_per_page)
            self.max_progress_value.emit(num_of_pages)
            htmls_list = [trial_request[1]]
            for i in range(1, num_of_pages):
                url = basic_url + f"&start={self.__articles_per_page * i}"
                htmls_list.append(download_page(url))
                sleep(3)
            # with ThreadPoolExecutor() as executor:
            #     try:
            #         htmls = executor.map(self.__download_page, urls)
            #     except requests.exceptions.ConnectionError:
            #         pass

            return htmls_list

    def __parse_htmls(self, htmls: List[str]) -> List[tuple]:  # , class_id: int
        """Парсинг скачанных страниц"""
        articles = list()
        for html in htmls:
            soup = BeautifulSoup(html, 'lxml')
            for article in soup.find_all('li', {'class': 'arxiv-result'}):
                # url
                url = article.find('p', {'class': 'list-title'}).a['href']
                assert len(url) > 0, "url не распознано"
                # title
                title = article.find('p', {'class': 'title'}).text.strip()
                assert len(title) > 0, "Название статьи не распознано"
                # tags
                tags_html = article.find('div', {'class': 'tags'}).find_all('span', {'class': 'tag'})
                assert len(tags_html) > 0, "Теги не распознаны"
                tag_codes_list = list()
                tag_names_list = list()
                for tag in tags_html:
                    tag_names_list.append(tag['data-tooltip'])
                    tag_codes_list.append(tag.text)
                tags_names = '; '.join(tag_names_list)
                tags_codes = ', '.join(tag_codes_list)
                # authors
                authors_html = article.find('p', {'class': 'authors'}).find_all('a')
                assert len(authors_html) > 0, "Имена авторов не распознаны"
                authors_list = list()
                for author in authors_html:
                    authors_list.append(author.text)
                authors = ', '.join(authors_list)
                # abstract
                abstract_html = article.find('span', {'class': 'abstract-full'})
                assert len(abstract_html) > 0, "Описание текста статьи не распознано"
                abstract = abstract_html.text.strip().split('\n')[0]
                # submitted
                submission_date_html = article.find('p', {'class': 'is-size-7'})
                assert len(submission_date_html) > 0, "Дата публикации не распознана"
                date = submission_date_html.contents[1].strip()[:-1].replace(',', '')
                submitted = make_typed_date(date)
                # article adding
                articles.append((url, title, tags_codes, tags_names, authors, abstract, submitted))  # , class_id
            #     if len(articles) == self.__n_for_each_query:
            #         break
            # if len(articles) == self.__n_for_each_query:
            #     break
        return articles

    def get_parsed_articles(self):
        """
        Скачивает и парсит страницы со статьями
        :return: Словарь со статьями, где ключи - запросы
        """
        articles = dict()
        for query in self.__queries.copy():
            try:
                htmls = self.__download_pages(query)
            except (requests.exceptions.ConnectionError, ValueError):
                self.__queries.remove(query)
                continue
            else:
                articles[query] = self.__parse_htmls(htmls)
        return articles

    def download_and_write_in_db(self, db_name=CURRENT_DB_NAME):
        """
        Скачивает и парсит страницы со статьями, попутно записывая их в бд
        :param db_name: Имя базы данных
        :return: Запросы, статьи по которым удалось записать
        """
        failed_queries = list()
        for i, query in enumerate(self.__queries.copy(), 1):
            self.query_num_str.emit(f'Запрос {i}')
            try:
                htmls = self.__download_pages(query)
            except requests.exceptions.ConnectionError:
                print('плохо')
                failed_queries.append(query)
                continue
            else:
                table = Table(f'{query}_arXiv', db_name)
                table.create_table()
                table.append_data(self.__parse_htmls(htmls))
            self.m_pages_counter = 0
            self.progress_counter.emit(0)
        return failed_queries


class ACM:

    def __init__(self, queries_list: List[str], n_for_each_query: int):
        self.__n_for_each_query = n_for_each_query
        self.__articles_per_page = 200
        if 0 < n_for_each_query <= 50:
            self.__articles_per_page = 50
        elif 0 < n_for_each_query <= 100:
            self.__articles_per_page = 100
        elif not (0 < n_for_each_query <= 10000 or n_for_each_query == -1):
            raise ValueError('некорректное n для каждого класса')
        self.__queries = list(set(queries_list))

    def make_url_for_query(self, query: str) -> str:
        url_base = 'https://dl.acm.org/action/doSearch?'
        query_params = {'fillQuickSearch': 'false',
                        'target': 'advanced',
                        'expand': 'dl',
                        'field1': 'AllField',
                        'text1': query.replace(', ', '+').replace(' ', '+'),
                        'AfterYear': '2010',
                        'BeforeYear': '2022',
                        'pageSize': str(self.__articles_per_page),
                        'rel': 'nofollow',
                        'ContentItemType': 'research-article'
                        }

        return url_base + '&'.join([f'{param}={value}' for param, value in query_params.items()])

    def trial_request(self, query: str):
        html = download_page(self.make_url_for_query(query))
        soup = BeautifulSoup(html, 'lxml')
        print(soup.find('a', {'class': 'search-result__nav__item active'}).text)
        if soup.find('a', {'class': 'search-result__nav__item active'}).text != ' Results':
            raise ValueError('Некорректный запрос')
        total_num_of_articles = int(soup.find('span', {'class': 'hitsLength'}).text.replace(',', ''))
        return total_num_of_articles, html

    def download_pages(self, query: str) -> List[str]:
        """Скачивание html страниц с сайта arxiv.org"""
        trial_request = self.trial_request(query)
        print(trial_request[0])
        if trial_request[0] == 0:
            print('Что-то пошло не так')
            raise requests.exceptions.ConnectionError('Что-то пошло не так)')
        if trial_request[0] <= 200 or 0 < self.__n_for_each_query <= 200:
            if trial_request[0] <= 50 or self.__n_for_each_query <= 50:
                self.__articles_per_page = 50
            elif trial_request[0] <= 100 or self.__n_for_each_query <= 100:
                self.__articles_per_page = 100
            elif trial_request[0] > 100 or self.__n_for_each_query > 100:
                self.__articles_per_page = 200
            # self.max_progress_value.emit(1)
            return [trial_request[1]]
        else:
            if self.__n_for_each_query == -1:
                if trial_request[0] > 10000:
                    self.__n_for_each_query = 10000
                else:
                    self.__n_for_each_query = trial_request[0]
            if trial_request[0] < self.__n_for_each_query:
                self.__n_for_each_query = trial_request[0]
            # print(self.__n_for_each_query)
            basic_url = self.make_url_for_query(query)
            urls = list()
            num_of_pages = ceil(self.__n_for_each_query / self.__articles_per_page)
            # self.max_progress_value.emit(num_of_pages)
            # for i in range(num_of_pages):
            #     urls.append(url + f"&startPage={self.__articles_per_page * i}")
            htmls_list = [trial_request[1]]
            for i in range(1, num_of_pages):
                url = basic_url + f"&start={self.__articles_per_page * i}"
                htmls_list.append(download_page(url))
                sleep(3)
            return htmls_list

    def parse_htmls(self, htmls: List[str]) -> List[tuple]:  # , class_id: int
        """Парсинг скачанных страниц"""
        articles = list()
        for html in htmls:
            soup = BeautifulSoup(html, 'lxml')
            for article in soup.find_all('li', {'class': 'search__item'}):

                title_and_url_html = article.find('span', {'class': 'hlFld-Title'})

                # title
                title = title_and_url_html.text.strip()
                assert len(title) > 0, "Название статьи не распознано"
                # print(title)

                # url
                url = 'https://dl.acm.org' + title_and_url_html.find('a')['href']
                assert len(url) > 0, "url не распознано"

                # authors
                authors_html = article.find('ul', {'aria-label': 'authors'})
                assert len(authors_html) > 0, "Имена авторов не распознаны"
                authors_list = list()
                for author in authors_html.findAll('a'):
                    authors_list.append(author.find('span').text)
                authors = ', '.join(authors_list)
                # print(authors)

                # submitted
                submission_date_html = article.find('div', {'class': 'bookPubDate'})
                assert len(submission_date_html) > 0, "Дата публикации не распознана"
                # print(submission_date_html['data-title'])
                date = submission_date_html['data-title'].replace('Published: ', '')
                submitted = make_typed_date(date)
                # print(submitted)

                try:
                    article_html = download_page(url)
                except (ConnectionError, ValueError):
                    continue

                sleep(3)
                article_soup = BeautifulSoup(article_html, 'lxml')

                # abstract
                try:
                    abstract = article_soup.find('div', {'class': 'abstractSection'}).find('p').text
                except AttributeError:
                    continue

                # tags
                tags_html = article_soup.find('ol', {'class': 'rlist organizational-chart'})
                tags_codes = ''
                tags_names = ''
                if tags_html:
                    tags_names = '; '.join([tag.text for tag in tags_html.findAll('p')])

                # article adding
                articles.append((url, title, tags_codes, tags_names, authors, abstract, submitted))

                if len(articles) == self.__n_for_each_query:
                    break
            if len(articles) == self.__n_for_each_query:
                break
        return articles

    def download_and_write_in_db(self, db_name=CURRENT_DB_NAME):
        """
        Скачивает и парсит страницы со статьями, попутно записывая их в бд
        :param db_name: Имя базы данных
        :return: Запросы, статьи по которым удалось записать
        """
        failed_queries = list()
        for i, query in enumerate(self.__queries.copy(), 1):
            try:
                htmls = self.download_pages(query)
            except requests.exceptions.ConnectionError:
                print('плохо')
                failed_queries.append('AND' + query)
                continue
            else:
                table = Table(f'AND {query}_ACM', db_name)
                table.create_table()
                table.append_data(self.parse_htmls(htmls))
        return failed_queries


if __name__ == '__main__':
    queries = [
        # 1
        # "AND machine+learning, OR classification, OR categorization, OR clustering",
        "natural+language+processing",
        # 2
        "AND recommender+systems, OR information+filtering, OR user+profile+construction, OR user+feedback",
        # 3
        "AND information+retrieval+systems, OR automated+retrieval+systems, OR information+overload,"
        " OR web+search+engine, OR question+answering",
        # 4
        "AND expert+systems, OR expert+estimates, OR expert+rules",
        # 5
        "AND neural+nets, OR artificial+neural+network",
        # 6
        "AND fuzzy+logic, OR fuzzy+sets, OR fuzzy+rules, OR membership+function, OR model+uncertainty,"
        " OR linguistic+variable",
        # 7
        "AND computational+complexity, OR data+structures+and+algorithms, OR computer+algorithms+analysis,"
        " OR efficient+algorithm",
        # 8
        "AND computer+vision, OR autonomous+robots"
    ]

    ArxivOrg(queries, 50).get_parsed_articles()
