import sys
from datetime import datetime
from sklearn.model_selection import train_test_split

print(sys.version)

twits_file_path ='Data/tweets.tsv'
trump_term_beginning_date = datetime(year=2017, month = 1, day=20)
class Twit:
    def __init__(self,id, user_handle,text, time_stamp, device  ):
        self.id = id
        self.user_handle =user_handle
        self.text =text
        self.time_stamp = datetime.strptime(time_stamp, '%Y-%m-%d %H:%M:%S')
        self.is_in_trump_term = self.time_stamp>= trump_term_beginning_date
        self.device =device
        self.apply_label()

    def apply_label(self):
        if self.device == 'android':
            self.label = 1
        else:
            self.label = 0


def read_file(tweets_file_path):
    all_twits = []
    with open(tweets_file_path) as file:
        for line in file:
            split_line = line.strip('\n').split('\t')
            id = split_line[0]
            user_handle = split_line[1]
            text = split_line[2]
            time_stamp = split_line[3]
            device = split_line[4]
            cur_twitt = Twit(id,user_handle,text,time_stamp, device)
            all_twits.append(cur_twitt)
    return all_twits


def write_attributes_frequency_to_file(all_twits):
    unique_user_handle, unique_devices, unique_period = get_unique_values(all_twits)
    with open('twits_attributes_count.tsv', 'w') as file:
        file.write('User Handle\tDevice\tIn Trump Term\tAmount_Of_twits\n')
        for user_handle in unique_user_handle:
            for device in unique_devices:
                for period in unique_period:
                    filtered_twits = [x for x in all_twits if
                                      x.user_handle == user_handle and x.device == device and x.is_in_trump_term == period]
                    if len(filtered_twits) > 0:
                        file.write("{}\t{}\t{}\t{}\n".format(user_handle, device, period, len(filtered_twits)))

import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression , LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import  cross_val_predict
from sklearn.naive_bayes import GaussianNB,MultinomialNB


def logistic_regression(x,labels,TEST_SIZE=0.2,RANDOM_STATE = 0):
    count_vect = CountVectorizer()
    x = count_vect.fit_transform(x)
    X_train, X_val, y_train, y_val = train_test_split(x, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    clf_lr = LogisticRegression(penalty='l1')
    clf_lr.fit(X_train, y_train)
    y_pred_lr = clf_lr.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, y_pred_lr)
    print (score)
    plot_precision_rcall_curve(y_test= y_val, y_score=y_pred_lr)

def fit_model(X_train, X_val, y_train, y_val ,model):
    model.fit(X_train, y_train)
    y_pred_lr = model.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, y_pred_lr)
    print (score)
    plot_precision_rcall_curve(y_test= y_val, y_score=y_pred_lr)

def plot_precision_rcall_curve(y_test, y_score):
    from sklearn.metrics import average_precision_score
    average_precision = average_precision_score(y_test, y_score)

    print('Average precision-recall score: {0:0.2f}'.format(
        average_precision))
    precision, recall, _ = precision_recall_curve(y_test, y_score)

    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format( average_precision))
    plt.show()


def get_unique_values(all_twits):
    unique_user_handle = set([twit.user_handle for twit in all_twits])
    unique_devices = set([twit.device for twit in all_twits])
    unique_period = set([twit.is_in_trump_term for twit in all_twits])
    return unique_user_handle, unique_devices, unique_period

def main():
    TEST_SIZE = 0.2
    RANDOM_STATE = 0
    all_twits = read_file(twits_file_path)
    x = [twit.text for twit in all_twits]
    labels = [twit.label for twit in all_twits]
    count_vect = CountVectorizer()
    x = count_vect.fit_transform(x)
    X_train, X_val, y_train, y_val = train_test_split(x, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    # logistic_regression(x,labels)
    fit_model(X_train, X_val, y_train, y_val, LogisticRegression(penalty='l1'))
    fit_model(X_train, X_val, y_train, y_val, SVC(probability=True))
    fit_model(X_train, X_val, y_train, y_val, MultinomialNB ())

if __name__ == "__main__":
    main()

    