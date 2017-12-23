import sys

import nltk
from sklearn.model_selection import train_test_split
import itertools
import timeit

import dataset_loader
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from scipy.sparse import hstack
import operator
from sklearn.metrics import average_precision_score
from nltk.tokenize import TweetTokenizer

LENGTH_FEATURE_STR = 'Length'
HASHTAGE_FEATURE_STR = 'hashtag'
EXCLANMATION_MARK_FEATURE_STR = 'exclamation'
QUESTION_MARK_FEATUR_STR = 'question'
WORD_COUNT_FEATURE_STR = 'word_count'
FAKE_NEWS_FEATURE_STR = 'fake_news'
AMERICA_GREAT_FEATURE_STR = 'america_great'

FEATURES_NAMES = [LENGTH_FEATURE_STR, WORD_COUNT_FEATURE_STR, HASHTAGE_FEATURE_STR, EXCLANMATION_MARK_FEATURE_STR,
                  QUESTION_MARK_FEATUR_STR,FAKE_NEWS_FEATURE_STR,AMERICA_GREAT_FEATURE_STR]
IS_FEATURE_EXTRACTED = None


def write_attributes_frequency_to_file(all_twits):
    unique_user_handle, unique_devices, unique_presidancy_period, trump_switched_to_iphone = get_unique_values(
        all_twits)
    with open('twits_attributes_count.tsv', 'w') as file:
        file.write('User Handle\tDevice\tIn Trump Term\tTrump Using Iphone\tAmount_Of_twits\n')
        for user_handle in unique_user_handle:
            for device in unique_devices:
                for period in unique_presidancy_period:
                    for using_iphone in trump_switched_to_iphone:
                        filtered_twits = [x for x in all_twits if
                                          x.user_handle == user_handle and x.device == device and x.is_in_trump_term == period and x.is_after_trump_switched_to_iphone == using_iphone]
                        if len(filtered_twits) > 0:
                            file.write("{}\t{}\t{}\t{}\t{}\n".format(user_handle, device, period, using_iphone,
                                                                     len(filtered_twits)))


def logistic_regression(x, labels, TEST_SIZE=0.2, RANDOM_STATE=0):
    count_vect = CountVectorizer()
    x = count_vect.fit_transform(x)
    X_train, X_val, y_train, y_val = train_test_split(x, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    clf_lr = LogisticRegression(penalty='l1')
    clf_lr.fit(X_train, y_train)
    y_pred_lr = clf_lr.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, y_pred_lr)
    plot_precision_rcall_curve(y_test=y_val, y_score=y_pred_lr)


def fit_model(X_train, X_val, y_train, y_val, model):
    start = timeit.timeit()
    model.fit(X_train, y_train)
    end = timeit.timeit()

    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)
    evaluation = {}
    evaluation['roc'] = roc_auc_score(y_val, y_pred_proba)
    evaluation['average_precision_score'] = average_precision_score (y_val, y_pred_proba)
    evaluation['accuracy_score'] =accuracy_score(y_val,y_pred)
    evaluation['f1_score'] =f1_score(y_val,y_pred)
    evaluation['precision_score'] =precision_score(y_val,y_pred)
    evaluation['recall_score'] =recall_score(y_val,y_pred)
    evaluation['time'] =end - start

    return evaluation


def plot_precision_rcall_curve(y_test, y_score):
    average_precision = average_precision_score(y_test, y_score)

    # print('Average precision-recall score: {0:0.2f}'.format(
    #     average_precision))
    precision, recall, _ = precision_recall_curve(y_test, y_score)

    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')

    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    # # plt.show()
    return average_precision

def get_unique_values(all_twits):
    unique_user_handle = set([twit.user_handle for twit in all_twits])
    unique_devices = set([twit.device for twit in all_twits])
    unique_presidancy_period = set([twit.is_in_trump_term for twit in all_twits])
    trump_switched_to_iphone = set([twit.is_after_trump_switched_to_iphone for twit in all_twits])
    return unique_user_handle, unique_devices, unique_presidancy_period, trump_switched_to_iphone


def add_vector_feature(matrix, values):
    return np.hstack([matrix, np.array(values).reshape(len(values), 1)])


def add_features(labeled_twits_texts, labeled_twits_matrix, IS_FEATURE_EXTRACTED):
    if IS_FEATURE_EXTRACTED[LENGTH_FEATURE_STR]:
        vector = [len(t) for t in labeled_twits_texts]
        labeled_twits_matrix = add_vector_feature(labeled_twits_matrix, vector)
    if IS_FEATURE_EXTRACTED[WORD_COUNT_FEATURE_STR]:
        word_count_vector = [len(str.split(t)) for t in labeled_twits_texts]
        labeled_twits_matrix = add_vector_feature(labeled_twits_matrix, word_count_vector)
    if IS_FEATURE_EXTRACTED[HASHTAGE_FEATURE_STR]:
        vector = [ t.count('#') for t in labeled_twits_texts]
        labeled_twits_matrix = add_vector_feature(labeled_twits_matrix, vector)
    if IS_FEATURE_EXTRACTED[EXCLANMATION_MARK_FEATURE_STR]:
        vector = [t.count('!') for t in labeled_twits_texts]
        labeled_twits_matrix = add_vector_feature(labeled_twits_matrix, vector)
    if IS_FEATURE_EXTRACTED[QUESTION_MARK_FEATUR_STR]:
        vector = [t.count('?') for t in labeled_twits_texts]
        labeled_twits_matrix = add_vector_feature(labeled_twits_matrix, vector)
    if IS_FEATURE_EXTRACTED[FAKE_NEWS_FEATURE_STR]:
        vector = [1  if 'fake' in t and 'news' in t else 0 for t in labeled_twits_texts ]
        labeled_twits_matrix = add_vector_feature(labeled_twits_matrix, vector)
    if IS_FEATURE_EXTRACTED[AMERICA_GREAT_FEATURE_STR]:
        vector = [1  if 'america' in t and 'great' in t else 0 for t in labeled_twits_texts ]
        labeled_twits_matrix = add_vector_feature(labeled_twits_matrix, vector)
    return labeled_twits_matrix


def generate_features_permutations():
    list = []
    n = len(FEATURES_NAMES)
    list.append([False]*n)
    # for i in range (n):
    #     list.append([True if j==i else False for j in range(n)])
    return list


def update_results(results, classifier_str, vectorizer_str, feature_permutation, evaluation,setteing):
    cur_result = {}
    cur_result['classifier'] = classifier_str
    cur_result['vectorizer'] = vectorizer_str
    cur_result.update(feature_permutation)
    cur_result.update(evaluation)
    cur_result['setting']= str(setteing)
    results.append(cur_result)


def write_results_to_csv(results):
    import csv
    keys = results[0].keys()
    with open('classifications_results/classification_unown.csv', 'wb') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)


def predict_unknown_twits(classifier,vectorizer, unlabeled_twits):
    with open('predict_unknown_twits.tsv', 'w') as file:
        file.write('User Handle\tDevice\tIn Trump Term\tTrump Using Iphone\t1_predictions\tAmount_Of_twits\n')
        unique_user_handle, unique_devices, unique_presidancy_period, trump_switched_to_iphone = get_unique_values(unlabeled_twits)
        for user_handle in unique_user_handle:
            for device in unique_devices:
                for period in unique_presidancy_period:
                    for using_iphone in trump_switched_to_iphone:
                        filtered_twits = [x for x in unlabeled_twits if
                                          x.user_handle == user_handle and x.device == device and x.is_in_trump_term == period and x.is_after_trump_switched_to_iphone == using_iphone]
                        if (len(filtered_twits))==0:
                            continue
                        filtered_twits_texts = [twit.text.lower() for twit in filtered_twits]
                        labeled_twits_matrix = vectorizer.transform(filtered_twits_texts).toarray()
                        predictions = classifier.predict(labeled_twits_matrix)
                        predictions_1 = sum (1 for p in predictions if p==1)
                        file.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(user_handle, device, period, using_iphone,predictions_1,
                                                                 len(filtered_twits)))


def main():
    lst = generate_features_permutations()
    TEST_SIZE = 0.2
    RANDOM_STATE = 0
    all_twits = dataset_loader.read_twits_file()
    write_attributes_frequency_to_file(all_twits)
    labeled_twits, unlabeled_twits = [twit for twit in all_twits if twit.label != None], [twit for twit in all_twits if
                                                                                          twit.label == None]
    labeled_twits_texts = [twit.text.lower() for twit in labeled_twits]
    tokenized_twits_text = []
    tknzr = TweetTokenizer()
    for twit in labeled_twits_texts:
        tokenized_twits_text.append(tknzr.tokenize(twit))
    labels = [twit.label for twit in all_twits]
    all_classifiers = ['LogisticRegression']#, 'SVC','MultinomialNB']
    all_vectorizers = ['CountVectorizer'
                        # ,'HashingVectorizer'
                        # ,'TfidfVectorizer'
                       ]
    classifier_settings = {}
    setting_dict = {'kernel': ['linear', 'poly', 'rbf'], 'shrinking': [True, False], 'C': [0.5,1,1.5]}
    classifier_settings['SVC'] =  generate_all_settings_premutations(setting_dict)
    setting_dict = {'penalty': ['l1', 'l2'], 'fit_intercept': [True, False], 'max_iter': [100,200,300]}
    classifier_settings['LogisticRegression'] =  generate_all_settings_premutations(setting_dict)
    setting_dict = {'alpha': [0.5,0.75,1], 'fit_prior': [True, False], 'class_prior':[None,[float(1113)/3624,float(2511)/3624]] }
    classifier_settings['MultinomialNB'] =  generate_all_settings_premutations(setting_dict)

    results = []
    for classifier_str in all_classifiers:
        for setteing in classifier_settings[classifier_str]:
            for vectorizer_str in all_vectorizers:
                for feature_permutation in lst:
                    IS_FEATURE_EXTRACTED = dict(zip(FEATURES_NAMES, feature_permutation))
                    classifer, vectorizer = evaluate_model_and_featrues_extraction(RANDOM_STATE, TEST_SIZE, classifier_str,setteing,
                                                           IS_FEATURE_EXTRACTED, labeled_twits_texts, labels, results,
                                                           vectorizer_str)
                    predict_unknown_twits(classifer,vectorizer, unlabeled_twits)
    write_results_to_csv(results)


def generate_all_settings_premutations(setting_dict):
    keys, values = zip(*setting_dict.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return list(experiments)


def evaluate_model_and_featrues_extraction(RANDOM_STATE, TEST_SIZE, classifier_str,setteing,IS_FEATURE_EXTRACTED,
                                           labeled_twits_texts, labels, results, vectorizer_str):
    print ('{} {} {}'.format(classifier_str,vectorizer_str,IS_FEATURE_EXTRACTED))
    if classifier_str == 'LogisticRegression':
        classifer = LogisticRegression(**setteing)
    elif classifier_str == 'SVC':
        setteing['probability']= True
        classifer= SVC(**setteing)
    elif classifier_str == 'MultinomialNB':
        classifer= MultinomialNB(**setteing)
    if vectorizer_str == 'CountVectorizer':
        vectorizer = CountVectorizer()
    elif vectorizer_str == 'HashingVectorizer':
        vectorizer = HashingVectorizer()
    elif vectorizer_str == 'TfidfVectorizer':
        vectorizer = TfidfVectorizer()
    labeled_twits_matrix = vectorizer.fit_transform(labeled_twits_texts).toarray()
    labeled_twits_matrix_with_features = add_features(labeled_twits_texts, labeled_twits_matrix,
                                                      IS_FEATURE_EXTRACTED)
    labeled_twits_matrix_with_features = [labeled_twits_matrix_with_features[i] for i in range(len(labels)) if labels[i]==0 or labels[i]==1]
    labels = [labels[i] for i in range(len(labels)) if labels[i]==0 or labels[i]==1]

    X_train, X_val, y_train, y_val = train_test_split(labeled_twits_matrix_with_features, labels,
                                                      test_size=TEST_SIZE, random_state=RANDOM_STATE)
    evaluation = fit_model(X_train, X_val, y_train, y_val, classifer)
    update_results(results, classifier_str, vectorizer_str, IS_FEATURE_EXTRACTED, evaluation, setteing)
    return classifer,vectorizer

if __name__ == '__main__':
    import sys

    print(sys.version)
    main()
