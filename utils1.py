import string
from enum import Enum
from math import sqrt, log2
from time import time
from typing import List, Dict, Callable, Union, Tuple
import re

import numpy as np
from nltk.stem import WordNetLemmatizer
from pandas import Series
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def tokenize(document: str, stop_words: List[str] = None) -> List[str]:
    if not stop_words:
        stop_words = list()

    document = re.sub(r'\$\S+\$', '', document).lower().strip()
    spec_chars = string.punctuation + string.digits + '\n'
    raw_document = ''.join([char for char in document if char not in spec_chars]).lower()

    tokens = list()
    for word in raw_document.split():
        if word not in stop_words:
            tokens.append(word)

    return tokens


def preprocess(lemma: WordNetLemmatizer, document: str, stop_words: List[str] = None) -> str:
    tokens = tokenize(document, stop_words)
    out_doc = list()
    for word in tokens:
        out_doc.append(lemma.lemmatize(word))
    return str(out_doc)


class ProfileType(Enum):
    Q = 1
    SS1_SOKAL_SNIS_1 = 2
    SS2_SOKAL_SNIS_2 = 3
    RT_ROJERS_TANIMOTO = 4
    S_JOINT_OCCUR = 5
    RO = 6
    RR_RASSEL_RAO = 7
    H_HAMMAN = 8
    J_JACCARD = 9
    MI = 10
    NMI = 11
    SN1 = 12
    SN2 = 13
    CHI_SQUARED = 14


def get_formula_for_profile_type(profile: ProfileType) -> Callable[[int, int, int, int], float]:
    """
    Получение формулы для вычисления профиля
    :param profile: тип профиля
    :return: функция
    """
    if profile == ProfileType.Q:
        return lambda a, b, c, d: (a*d-c*b)/(a*d+c*b)
    elif profile == ProfileType.SS1_SOKAL_SNIS_1:
        return lambda a, b, c, d: 2*(a+d)/(2*(a+d)+b+c)
    elif profile == ProfileType.SS2_SOKAL_SNIS_2:
        return lambda a, b, c, d: a/(a+2*(b+c))
    elif profile == ProfileType.RT_ROJERS_TANIMOTO:
        return lambda a, b, c, d: (a+d)/(a+d+2*(b+c))
    elif profile == ProfileType.S_JOINT_OCCUR:
        return lambda a, b, c, d: (a+d)/(a+b+c+d)
    elif profile == ProfileType.RO:
        return lambda a, b, c, d: (a*d-c*b)/sqrt((a+b)*(c+d)*(a+c)*(b+d))
    elif profile == ProfileType.RR_RASSEL_RAO:
        return lambda a, b, c, d: a/(a+b+c+d)
    elif profile == ProfileType.H_HAMMAN:
        return lambda a, b, c, d: (a+d-b-c)/(a+b+c+d)
    elif profile == ProfileType.J_JACCARD:
        return lambda a, b, c, d: a/(a+b+c)
    elif profile == ProfileType.MI:
        return lambda a, b, c, d: log2(a*(a+b+c+d)/((a+b)*(a+c)))
    elif profile == ProfileType.NMI:
        return lambda a, b, c, d: a*log2(a*(a+b+c+d)/((a+b)*(a+c))) / (a+b)*log2((a+b+c+d)/(a+b))
    elif profile == ProfileType.SN1:
        return lambda a, b, c, d: (c+b)/(a+b+c+d)
    elif profile == ProfileType.SN2:
        return lambda a, b, c, d: (c+b)/(a+b+c)
    elif profile == ProfileType.CHI_SQUARED:
        return lambda a, b, c, d: (a+b+c+d)*(a*d-b*c)**2/((a+b)*(c+d)*(a+c)*(b+d))


class Profile:

    def __init__(self, coefficients: Dict[str, Dict[str, int]], profile_type: ProfileType, profile_length=None):
        """
        Инициализация профиля
        :param coefficients: {word: {Union['A', 'B', 'C', 'D']: coefficient}}
        :param profile_type: Тип профиля
        :param profile_length: Длина профиля
        """
        self.type = profile_type
        self.length = profile_length
        profile_formula = get_formula_for_profile_type(profile_type)

        weights = dict().fromkeys(coefficients.keys(), 0.0)
        for word in coefficients:
            a = coefficients[word]['A']
            b = coefficients[word]['B']
            c = coefficients[word]['C']
            d = coefficients[word]['D']
            try:
                weights[word] = profile_formula(a, b, c, d)
            except (ZeroDivisionError, ValueError):
                pass
        self.weights = Series(dict(list(sorted(weights.items(), key=lambda x: x[1], reverse=True))[:self.length]))
        self.all_words = self.weights.index.tolist()

    @staticmethod
    def calculate_tf(X: List[str], y=None) -> Dict[str, float]:
        doc_len = len(X)
        output = dict()
        for word in set(X):
            output[word] = X.count(word)
        return output

    def calculate_similarity(self, X: List[str]) -> float:
        """
        Вычисление схожести документа с профилем
        :param X: Список токенов документа
        :param vectorizer: Преобразователь документа в вектор
        :return: Степень схожести
        """

        # return sum([0 if word not in self.all_words else self.weights[word]
        #             for word, weight in self.calculate_tf(X).items()])
        return sum([0 if word not in self.all_words else self.weights[word] * weight
                    for word, weight in self.calculate_tf(X).items()])


class ProfileClassifierDefault:

    def __init__(self, profile_type: ProfileType, profile_length: int = None, max_vocabulary_size: int = None):
        self.profile_type = profile_type
        self.profile_length = profile_length
        self.max_vocabulary_size = max_vocabulary_size
        self.coefficients = None      # Dict[str, DataFrame] - класс и датафрейм коэффициентов для каждого слова
        self.profiles = None          # Dict[str, Profile] - класс и профиль из наиболее частотных слов заданной длины
        self.classes = None           # Dict[str, int] - закодированный список используемых классов
        self.vocabulary_size = 0

    @staticmethod
    def str_to_list(text: str) -> List[str]:
        return text.replace("'", '').replace("[", '').replace("]", '').split(', ')

    def calculate_profiles(self):
        self.profiles = {cls: Profile(self.coefficients[cls], self.profile_type, self.profile_length)
                         for cls in self.coefficients.keys()}

    def get_vocabulary(self, X: List[str]):
        vect = CountVectorizer(token_pattern=r'[a-zA-Zа-яА-Я]{3,}', max_features=self.max_vocabulary_size)
        vect.fit(X)
        return vect.get_feature_names_out()

    def fit(self, X: List[str], y: List[str]) -> None:
        """
        Построение профилей для каждого класса
        :param X: Список токенизированных документов (токены записаны единой строкой, где разделитель ', ')
        :param y: Список классов, к которым относится соответствующий документ
        :return: None
        """

        t1 = time()
        self.classes = set(y)
        vocabulary = set(self.get_vocabulary(X))
        self.vocabulary_size = len(vocabulary)
        t2 = time()
        print(f'vocabulary fit = {t2 - t1}')
        print(f'vocabulary size = {len(vocabulary)}')

        t1 = time()
        self.coefficients = {cls: {word: {'A': 0, 'B': 0, 'C': 0, 'D': 0} for word in vocabulary}
                             for cls in self.classes}

        for i, (document, cls) in enumerate(zip(X, y)):
            document_words = set(self.str_to_list(document))
            other_words = vocabulary - document_words
            other_classes = set(self.classes)
            other_classes.remove(cls)

            for word in document_words:
                if word in vocabulary:
                    self.coefficients[cls][word]['A'] += 1
                    for other_cls in other_classes:
                        self.coefficients[other_cls][word]['B'] += 1
            for word in other_words:
                self.coefficients[cls][word]['C'] += 1
                for other_cls in other_classes:
                    self.coefficients[other_cls][word]['D'] += 1

        t2 = time()
        print(f"profiles fit time = {t2 - t1}")
        self.calculate_profiles()

    def set_profile_type(self, new_profile_type: ProfileType) -> None:
        if not self.coefficients:
            return
        if not self.classes:
            return
        if new_profile_type == self.profile_type:
            return

        self.profile_type = new_profile_type
        self.calculate_profiles()

    def set_profile_length(self, new_length: int) -> None:
        if not self.coefficients:
            return
        if not self.classes:
            return
        if new_length <= 0:
            return
        if new_length == self.profile_length:
            return

        self.profile_length = new_length
        if new_length > self.vocabulary_size:
            self.profile_length = self.vocabulary_size
        self.calculate_profiles()

    def calculate_probabilities(self, X: List[str], y=None) -> Dict[str, float]:
        return {cls: profile.calculate_similarity(X) for cls, profile in self.profiles.items()}

    def predict_class(self, X: List[str], y=None) -> str:
        return list(sorted(self.calculate_probabilities(X).items(), key=lambda x: x[1], reverse=True))[0][0]

    def predict(self, X: List[str], y=None) -> Union[List[str], None]:
        """
        Предсказать к каким классам относятся новые наблюдения
        :param X: Список токенизированных документов (токены записаны единой строкой, где разделитель ', ')
        :param y: Игнорируется
        :return: Список меток классов
        """
        if not self.profiles:
            return
        if not self.classes:
            return

        output = list()
        for x in X:
            output.append(self.predict_class(self.str_to_list(x)))
        return output

    def predict_proba(self, X: List[str], y=None) -> Union[List[Dict[str, float]], None]:
        """
        Предсказать вероятности отношения к классам
        :param X: Список токенизированных документов (токены записаны единой строкой, где разделитель ', ')
        :param y: Игнорируется
        :return: Список вероятностей отнесения к каждому классу
        """
        if not self.profiles:
            return
        if not self.classes:
            return

        output = list()
        for x in X:
            probabilities = self.calculate_probabilities(self.str_to_list(x))
            sum_proba = sum(map(lambda t: abs(t[1]), probabilities.items()))
            if not sum_proba:
                sum_proba = 1
            for cls, proba in probabilities.items():
                probabilities[cls] = abs(proba) / sum_proba
            output.append(probabilities)
        return output


class ProfileClassifierExtended(ProfileClassifierDefault):

    def fit(self, X: List[str], y: List[str]) -> None:
        """
        Построение профилей для каждого класса
        :param X: Список токенизированных документов (токены записаны единой строкой, где разделитель ', ')
        :param y: Список классов, к которым относится соответствующий документ
        :return: None
        """

        t1 = time()
        self.classes = {cls: i for i, cls in enumerate(set(y))}
        vocabulary = set(self.get_vocabulary(X))
        self.vocabulary_size = len(vocabulary)
        t2 = time()
        print(f'vocabulary fit = {t2 - t1}')
        print(f'vocabulary size = {len(vocabulary)}')

        t1 = time()
        self.coefficients = {(cls1, cls2): {word: {'A': 0, 'B': 0, 'C': 0, 'D': 0} for word in vocabulary}
                             for _, cls1 in self.classes.items() for _, cls2 in self.classes.items()}

        for i, (document, cls) in enumerate(zip(X, y)):
            document_words = set(self.str_to_list(document))
            other_words = vocabulary - document_words
            other_classes = set(self.classes.keys())
            other_classes.remove(cls)
            cls_code = self.classes[cls]

            for word in document_words:
                if word in vocabulary:
                    self.coefficients[(cls_code, cls_code)][word]['A'] += 1
                    for other_cls in other_classes:
                        other_cls_code = self.classes[other_cls]
                        self.coefficients[(other_cls_code, other_cls_code)][word]['B'] += 1
                        self.coefficients[(cls_code, other_cls_code)][word]['A'] += 1
                        self.coefficients[(other_cls_code, cls_code)][word]['B'] += 1
            for word in other_words:
                self.coefficients[(cls_code, cls_code)][word]['C'] += 1
                for other_cls in other_classes:
                    other_cls_code = self.classes[other_cls]
                    self.coefficients[(other_cls_code, other_cls_code)][word]['D'] += 1
                    self.coefficients[(cls_code, other_cls_code)][word]['C'] += 1
                    self.coefficients[(other_cls_code, cls_code)][word]['D'] += 1

        t2 = time()
        print(f"profiles fit time = {t2 - t1}")
        self.calculate_profiles()

    def calculate_matrix(self, X: List[str], y=None) -> np.ndarray:
        matrix = np.zeros((len(self.classes), len(self.classes)), float)
        for (i, j), profile in self.profiles.items():
            if i != j:
                matrix[i, j] = profile.calculate_similarity(X)
        return matrix

    def calculate_probabilities(self, X: List[str], y=None) -> Dict[str, float]:
        """Можно поиграться с формулами"""
        matrix = self.calculate_matrix(X)
        probabilities = dict().fromkeys(self.classes.keys(), 0.0)
        # classes_count = (len(self.classes) - 1) * 2
        classes_count = len(self.classes) - 1
        for cls, i in self.classes.items():
            row_sum = 0
            for j, el in enumerate(matrix[i]):
                if i != j and el > 0:
                    row_sum += el
            col_sum = 0
            for j, el in enumerate(matrix[:, i]):
                if i != j and el > 0:
                    col_sum += el
            probabilities[cls] = row_sum - col_sum  # matrix[i, i]
        return probabilities

    def predict_class(self, X: List[str], y=None) -> str:
        return list(sorted(self.calculate_probabilities(X).items(), key=lambda x: x[1], reverse=True))[0][0]


class ProfilesEnsemble:
    def __init__(self, profile_types_with_lengths: List[Tuple[ProfileType, int]], max_vocabulary_size: int = None):
        self.base_estimator = ProfileClassifierDefault(ProfileType.RO, 0, max_vocabulary_size)
        self.profile_types_with_lengths = profile_types_with_lengths
        self.profiles = None
        self.coefficients = None
        self.classes = None
        self.vocabulary_size = 0

    @staticmethod
    def str_to_list(text: str) -> List[str]:
        return text.replace("'", '').replace("[", '').replace("]", '').split(', ')

    def calculate_profiles(self):
        self.profiles = {profile_type: {cls: Profile(self.coefficients[cls], profile_type, profile_length)
                                        for cls in self.coefficients.keys()}
                         for profile_type, profile_length in self.profile_types_with_lengths}

    def fit(self, X: List[str], y: List[str]) -> None:
        self.base_estimator.fit(X, y)
        self.classes = self.base_estimator.classes
        self.coefficients = self.base_estimator.coefficients
        self.calculate_profiles()

    def set_profile_types_with_lengths(self, new_profile_types_with_lengths: List[Tuple[ProfileType, int]]) -> None:
        if not self.coefficients:
            return
        if new_profile_types_with_lengths == self.profile_types_with_lengths:
            return

        used_profile_types = list()
        for profile_type, profile_length in new_profile_types_with_lengths:
            if profile_type not in used_profile_types:
                used_profile_types.append(profile_type)
            else:
                raise ValueError('Ошибка: повторяющиеся типы профилей')
            if profile_length < 1:
                raise ValueError('Ошибка: отрицательная длина профиля')

        self.profile_types_with_lengths = new_profile_types_with_lengths
        self.calculate_profiles()

    def calculate_probabilities(self, X: List[str], y=None) -> Dict[str, float]:
        return {cls: profile.calculate_similarity(X)
                for _, profiles in self.profiles.items() for cls, profile in profiles.items()}

    def predict_class(self, X: List[str], y=None) -> str:
        return list(sorted(self.calculate_probabilities(X).items(), key=lambda x: x[1], reverse=True))[0][0]

    def predict(self, X: List[str], y=None) -> Union[List[str], None]:
        """
        Предсказать к каким классам относятся новые наблюдения
        :param X: Список токенизированных документов (токены записаны единой строкой, где разделитель ', ')
        :param y: Игнорируется
        :return: Список меток классов
        """
        if not self.profiles:
            return
        if not self.classes:
            return

        output = list()
        for x in X:
            output.append(self.predict_class(self.str_to_list(x)))
        return output

    def predict_proba(self, X: List[str], y=None) -> Union[List[Dict[str, float]], None]:
        """
        Предсказать вероятности отношения к классам
        :param X: Список токенизированных документов (токены записаны единой строкой, где разделитель ', ')
        :param y: Игнорируется
        :return: Список вероятностей отнесения к каждому классу
        """
        if not self.profiles:
            return
        if not self.classes:
            return

        output = list()
        for x in X:
            probabilities = self.calculate_probabilities(self.str_to_list(x))
            sum_proba = sum(map(lambda t: abs(t[1]), probabilities.items()))
            if not sum_proba:
                sum_proba = 1
            for cls, proba in probabilities.items():
                probabilities[cls] = abs(proba) / sum_proba
            output.append(probabilities)
        return output
