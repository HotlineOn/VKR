from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split
from xgboost import XGBClassifier
from pandas import DataFrame, Series
from collections import Counter
from wordcloud import WordCloud
from word_doc_matrix import SelectionMatrix
from time import time
from typing import List, Tuple
from numpy import ndarray, zeros
from catboost import CatBoostClassifier
import umap


CLASSIFICATION_METHODS = {
    'Наивный Байес': MultinomialNB(),
    'Логистическая регрессия': LogisticRegression(class_weight='balanced', random_state=0),
    'К-ближайших соседей': KNeighborsClassifier(),
    'Дерево решений': DecisionTreeClassifier(class_weight='balanced', max_depth=15, random_state=0),
    'Бэггинг': BaggingClassifier(DecisionTreeClassifier(max_depth=15, class_weight='balanced'),
                                 random_state=0),
    'Случайный лес': RandomForestClassifier(n_estimators=10, max_depth=15, class_weight='balanced', random_state=0),
    'Градиентный бустинг': CatBoostClassifier(iterations=100, task_type="GPU", devices='0:1')  # XGBClassifier()  # GradientBoostingClassifier(loss='log_loss', random_state=0)
}


def jaccard_similarity(words_set_1: List[str], words_set_2: List[str]):
    intersection = len(list(set(words_set_1).intersection(words_set_2)))
    union = (len(words_set_1) + len(words_set_2)) - intersection
    return float(intersection) / union


def weighted_jaccard_similarity(words_with_frequencies_1: Series, words_with_frequencies_2: Series):
    intersection_set = set(words_with_frequencies_1.index).intersection(words_with_frequencies_2.index)
    intersection = sum([max(words_with_frequencies_1[word], words_with_frequencies_2[word])
                        for word in intersection_set])
    union = sum(words_with_frequencies_1) + sum(words_with_frequencies_2) - intersection
    return float(intersection) / union


def calculate_jaccard_matrix(words_with_frequencies_for_each_class: List[Series]) -> ndarray:
    words_for_each_class = [series.index for series in words_with_frequencies_for_each_class]
    classes_count = len(words_for_each_class)
    matrix = zeros((classes_count, classes_count))
    for i in range(classes_count):
        for j in range(classes_count):
            matrix[i, j] = jaccard_similarity(words_for_each_class[i], words_for_each_class[j])
    return matrix


def calculate_weighted_jaccard_matrix(words_with_frequencies_for_each_class: List[Series]) -> ndarray:
    classes_count = len(words_with_frequencies_for_each_class)
    matrix = zeros((classes_count, classes_count))
    for i in range(classes_count):
        for j in range(classes_count):
            matrix[i, j] = weighted_jaccard_similarity(words_with_frequencies_for_each_class[i],
                                                       words_with_frequencies_for_each_class[j])
    return matrix


def make_word_cloud_plot(words_with_frequencies, ax, max_words=100, width=2400, height=1100):
    wc = WordCloud(background_color="white", height=height, width=width, max_words=max_words, random_state=0)
    wc.generate_from_frequencies(words_with_frequencies)
    ax.imshow(wc, interpolation='bilinear')  #
    ax.axis("off")


def make_correlation_matrix_plot(matrix, ax):
    dfs = matrix.split_dataframe_by_classes()
    df = DataFrame(list(map(lambda x: x.mean(1).reset_index()[0], dfs)),
                   index=[f'Класс {i}' for i in range(1, len(dfs)+1)])
    corr = df.transpose().corr()
    im = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45)
    ax.set_yticks(range(len(corr.columns)), corr.columns)
    for i in range(1, len(corr.columns)+1):
        for j in range(1, len(corr.columns)+1):
            value = corr[f'Класс {i}'][f'Класс {j}']
            ax.text(i-1, j-1, str(round(value, 3)), va='center', ha='center')
    return im


def reduce_data(matrix: SelectionMatrix, method='PCA', n_components=2):
    if method not in ('PCA', 't-SNE', 'UMAP'):
        raise ValueError('Используются только pca и tsne')

    estimator = None
    if method == 'PCA':
        estimator = PCA(n_components=n_components, random_state=0)
    if method == 't-SNE':
        estimator = TSNE(n_components=n_components, random_state=0)
    if method == 'UMAP':
        estimator = umap.UMAP(n_neighbors=100, n_components=n_components, random_state=0)

    reduced_data = estimator.fit_transform(matrix.df)
    coordinates = ['x', 'y'] if n_components == 2 else ['x', 'y', 'z']
    return DataFrame(reduced_data, index=matrix.df.index, columns=coordinates), estimator


def reduce_data_features_class(matrix: SelectionMatrix, method='PCA', n_components=2):
    df, _ = reduce_data(matrix, method, n_components)
    reduced_classes = list()
    for i in range(len(matrix.classes_names)):
        reduced_classes.append(df.iloc[i*matrix.class_size:(i+1)*matrix.class_size])
        # print(reduced_classes[i])
    return reduced_classes


def reduce_data_features_cluster(matrix: SelectionMatrix, cluster_indexes, centroids, method='PCA', n_components=2):
    df, estimator = reduce_data(matrix, method, n_components)
    reduced_clusters = list()
    for _, cluster in cluster_indexes:
        reduced_clusters.append(df.iloc[cluster])

    if method == 'PCA':
        reduced_centroids = estimator.transform(centroids)
        return {'dots for each cluster': reduced_clusters, 'centroids': reduced_centroids}
    if method == 't-SNE':
        return {'dots for each cluster': reduced_clusters, 'centroids': None}


def clusterization(matrix: SelectionMatrix) -> dict:
    dfs = matrix.split_dataframe_by_classes()
    n_clusters = len(dfs)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    start = time()
    kmeans.fit(matrix.df)
    kmeans_fit_time = time()
    print(f'Время обучения KMeans: {kmeans_fit_time - start}')
    cluster_indexes = {i: list() for i in range(n_clusters)}
    for i, label in enumerate(kmeans.labels_):
        cluster_indexes[label].append(i)

    table_data = dict()
    n_for_each_cluster = [0 for _ in range(n_clusters)]
    for i, dataframe in enumerate(dfs, 1):
        prediction = Counter({i: 0 for i in range(n_clusters)})
        prediction.update(kmeans.predict(dataframe))
        p = prediction.most_common(n_clusters)
        for j, cluster_n in p:
            n_for_each_cluster[j] += cluster_n
        table_data[f'Класс {i}'] = [str(count) if count else '0' for _, count in sorted(p, key=lambda x: x[0])]
    table_data['Всего'] = [str(i) for i in n_for_each_cluster]
    # print(table_data)
    return {'table data': table_data, 'cluster centers': kmeans.cluster_centers_, 'cluster indexes': cluster_indexes}


def classification_report_for_table(matrix: SelectionMatrix, method: str, k: int, best_test_only: bool) \
        -> List[Tuple[dict, ndarray]]:
    selection_size = matrix.df.shape[0]
    num_of_classes = len(matrix.classes_names)
    estimator = CLASSIFICATION_METHODS[method]
    if method == 'К-ближайших соседей':
        estimator.set_params(n_neighbors=int(selection_size/num_of_classes*0.3), weights='distance', n_jobs=-1)

    y = list()
    for i in range(num_of_classes):
        y.extend([i] * int(selection_size / num_of_classes))

    x_train, x_validation, y_train, y_validation = train_test_split(matrix.df, y, test_size=0.3, random_state=0)
    cv = StratifiedKFold(n_splits=k, random_state=0, shuffle=True)
    scores = cross_validate(estimator, x_train, y_train, scoring='balanced_accuracy', cv=cv, return_estimator=True)
    # print(scores)
    if best_test_only:
        best_model_index = scores['test_score'].tolist().index(max(scores['test_score']))
        best_model = scores['estimator'][best_model_index]
        prediction = best_model.predict(x_validation)
        report = classification_report(y_validation, prediction, output_dict=True)
        report['test_accuracy'] = scores['test_score'][best_model_index]
        confusion = confusion_matrix(y_validation, prediction)
        # help(best_model.tree_)
        return [(report, confusion)]
    else:
        reports = list()
        for i in range(k):
            # print(scores['test_score'][i])
            prediction = scores['estimator'][i].predict(x_validation)
            report_i = classification_report(y_validation, prediction, output_dict=True)
            confusion_matrix_i = confusion_matrix(y_validation, prediction)
            # print(y_validation)
            # print(list(prediction))
            report_i['test_accuracy'] = scores['test_score'][i]
            reports.append((report_i, confusion_matrix_i))
        return reports


def classification_result(matrix: DataFrame, n_classes: int, method='Дерево решений', metric='f1-score', k=5,
                          best_test_only=True) -> List[float]:

    score_function = f1_score
    if metric == 'recall':
        score_function = recall_score
    if metric == 'precision':
        score_function = precision_score
    # print(str(metric))
    # print(score_function)

    selection_size = matrix.shape[0]
    estimator = CLASSIFICATION_METHODS[method]
    if method == 'К-ближайших соседей':
        estimator.set_params(n_neighbors=int(selection_size / n_classes*0.3), weights='distance', n_jobs=-1)

    y = list()
    for i in range(n_classes):
        y.extend([i] * int(selection_size / n_classes))

    x_train, x_validation, y_train, y_validation = train_test_split(matrix, y, test_size=0.3, random_state=0)
    cv = StratifiedKFold(n_splits=k, random_state=0, shuffle=True)
    scores = cross_validate(estimator, x_train, y_train, scoring='balanced_accuracy', cv=cv, return_estimator=True)
    if best_test_only:
        best_model_index = scores['test_score'].tolist().index(max(scores['test_score']))
        best_model = scores['estimator'][best_model_index]
        return [scores['test_score'][best_model_index],
                score_function(y_validation, best_model.predict(x_validation), average='macro', zero_division=0)]
    return [score_function(y_validation, scores['estimator'][i].predict(x_validation), average='macro', zero_division=0)
            for i in range(k)]


def train_estimators(matrix: SelectionMatrix, method: str, k: int):
    selection_size = matrix.df.shape[0]
    num_of_classes = len(matrix.classes_names)
    estimator = CLASSIFICATION_METHODS[method]
    if method == 'К-ближайших соседей':
        estimator.set_params(n_neighbors=int(selection_size / num_of_classes * 0.3), weights='distance', n_jobs=-1)

    y = list()
    for i in range(num_of_classes):
        y.extend([i] * int(selection_size / num_of_classes))

    x_train, x_validation, y_train, y_validation = train_test_split(matrix.df, y, test_size=0.3, random_state=0)
    cv = StratifiedKFold(n_splits=k, random_state=0, shuffle=True)
    scores = cross_validate(estimator, x_train, y_train, scoring='balanced_accuracy', cv=cv, return_estimator=True)

    models = {i: dict() for i in range(k)}
    for i, model in enumerate(scores['estimator']):
        predictions = model.predict(x_validation)
        f1 = f1_score(y_validation, predictions, average='macro', zero_division=0)
        print(i, f1)
        models[i]['model'] = model
        models[i]['f1_score'] = f1
    return models

# # TODO попарная классификация (матрица попарной классификации)
# def classification_confusion_matrix(matrix: SelectionMatrix, method: str, k: int, best_test_only: bool) -> List[dict]:
#     selection_size = matrix.df.shape[0]
#     num_of_classes = len(matrix.classes_names)
#     estimator = CLASSIFICATION_METHODS[method]
#     if method == 'К-ближайших соседей':
#         estimator.set_params(n_neighbors=int(selection_size/num_of_classes*1.2), weights='distance', n_jobs=-1)
#
#     y_true = list()
#     for i in range(num_of_classes):
#         y_true.extend([i] * int(selection_size / num_of_classes))
#
#     x_train, x_validation, y_train, y_validation = train_test_split(matrix.df, y_true, test_size=0.3, random_state=0)
#     cv = StratifiedKFold(n_splits=k, random_state=0, shuffle=True)
#     scores = cross_validate(estimator, x_train, y_train, scoring='balanced_accuracy', cv=cv, return_estimator=True)
#     if best_test_only:
#         best_model_index = scores['test_score'].tolist().index(max(scores['test_score']))
#         best_model = scores['estimator'][best_model_index]
#         # print('Обучающая выборка:')
#         # print(confusion_matrix(y_train[best_model_index], best_model.predict(y_train[best_model_index])))
#         # print()
#         print('Валидационная выборка:')
#         print(confusion_matrix(y_validation, best_model.predict(x_validation)))
#     else:
#         # confusion_matrices = list()
#         for i in range(k):
#             # print(scores['test_score'][i])
#             # prediction = scores['estimator'][i].predict(x_validation)
#             # confusion_matrices.append(confusion_matrix(y_validation, prediction))
#             # print('Обучающая выборка:')
#             # print(confusion_matrix(y_train[i], scores['estimator'][i].predict(y_train[i])))
#             # print()
#             print('Валидационная выборка:')
#             print(confusion_matrix(y_validation, scores['estimator'][i].predict(x_validation)))

