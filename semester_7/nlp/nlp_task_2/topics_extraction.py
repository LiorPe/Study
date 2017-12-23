# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Lars Buitinck
#         Chyi-Kwei Yau <chyikwei.yau@gmail.com>
# License: BSD 3 clause

from __future__ import print_function
from time import time
from sklearn import metrics

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt

import dataset_loader

import sys


n_samples = 2000
n_features = 1000
n_components = 10
n_top_words = 20
use_kmeans = True
k = 10

def print_top_words(model, feature_names, n_top_words):
    top_words_by_topis = {}
    for topic_idx, topic in enumerate(model.components_):
        top_words_by_topis[topic_idx] = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
    return top_words_by_topis


# Load the 20 newsgroups dataset and vectorize it. We use a few heuristics
# to filter out useless terms early on: the posts are stripped of headers,
# footers and quoted replies, and common English words, words occurring in
# only one document or in at least 95% of the documents are removed.

def reduce_data_by_topics(transformed_data, vectorizer, n):

    print("Fitting LDA models with tf features, "
          "n_samples=%d and n_features=%d..."
          % (n_samples, n_features))
    lda = LatentDirichletAllocation(n_components=n, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    t0 = time()
    X = lda.fit_transform(transformed_data)
    print("done in %0.3fs." % (time() - t0))
    print("\nTopics in LDA model:")
    tf_feature_names = vectorizer.get_feature_names()
    top_words = print_top_words(lda, tf_feature_names, n_top_words)
    return X,lda, top_words

def transform_data_to_vectors(data_samples, tf_idf = False):
    # Use tf-idf features for NMF.
    t0 = time()
    if (tf_idf):
        print("Extracting tf-idf features for NMF...")
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                           max_features=n_features,
                                           stop_words='english')
    else:
        # Use tf (raw term count) features for LDA.
        print("Extracting tf features for LDA...")
        vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                        max_features=n_features,
                                        stop_words='english')
        t0 = time()
    transformed_data = vectorizer.fit_transform(data_samples)
    print("done in %0.3fs." % (time() - t0))
    return transformed_data, vectorizer


def cluster(est, X, labels=None):
    print("Clustering sparse data with %s" % est)
    t0 = time()
    x_cluster_transformed = est.fit_transform(X)
    print("done in %0.3fs" % (time() - t0))
    print()

    if labels!=None:
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, est.labels_))
        print("Completeness: %0.3f" % metrics.completeness_score(labels, est.labels_))
        print("V-measure: %0.3f" % metrics.v_measure_score(labels, est.labels_))
        print("Adjusted Rand-Index: %.3f"
              % metrics.adjusted_rand_score(labels, est.labels_))
        print("Silhouette Coefficient: %0.3f"
              % metrics.silhouette_score(X, est.labels_, sample_size=1000))

    # print()
    # print("Top terms per cluster:")
    # if (use_kmeans):
    #     order_centroids = est.cluster_centers_.argsort()[:, ::-1]
    #     terms = vectorizer.get_feature_names()
    #     for i in range(k):
    #         print("Cluster %d:" % i, end='')
    #         for ind in order_centroids[i, :10]:
    #             print(' %s' % terms[ind], end='')
    #         print()
    # print_top_words(est, terms, n_top_words)


def write_top_words_by_topic_count(top_words_by_topic_count,perplexity_by_topic_count,path):
    with open(path, 'w') as file:
        for topic_count, top_words_by_topic_dict in top_words_by_topic_count.items():
            file.write("Number of topics: {}\n".format(topic_count))
            for topic_idx, top_word_of_topic in top_words_by_topic_dict.items():
                file.write("{},{}\n".format(topic_idx,', '.join(top_word_of_topic)))
            file.write("{},{}\n".format('perplexity',perplexity_by_topic_count[topic_count] ))


def test_topic_extraction_twits():
    twits = dataset_loader.read_twits_file()
    text = [twit.text for twit in twits]
    test_topic_extraction(text,'Trump Tweets')

def test_topic_extraction_comments():
    comments = dataset_loader.get_comments_data()
    filtered_comments = [comments[i] for i in range(len(comments)) if i%2==0]
    test_topic_extraction(comments,'FCC Dataset')


from sklearn.metrics import calinski_harabaz_score, silhouette_score


def tweets_k_means_test():
    data_set_name = 'Tweets'
    twits = dataset_loader.read_twits_file()
    text = [twit.text for twit in twits]
    transformed_data, vectorizer = transform_data_to_vectors(text)
    n_topis_values = [2,3,4,5]
    k_cluster_values = range(2,10)
    generic_k_means_test(data_set_name, k_cluster_values, n_topis_values, transformed_data, vectorizer)

def fcc_k_means_test():
    data_set_name = 'FCC Dataset'
    comments = dataset_loader.get_comments_data()
    filtered_comments = [comments[i] for i in range(len(comments)) if i%2==0]
    transformed_data, vectorizer = transform_data_to_vectors(filtered_comments)
    n_topis_values = [2,3,4,5,6]
    k_cluster_values = range(2,10)
    generic_k_means_test(data_set_name, k_cluster_values, n_topis_values, transformed_data, vectorizer)

def generic_k_means_test(data_set_name, k_cluster_values, n_topis_values, transformed_data, vectorizer):
    all_settings_evaluation = []
    for n_topic in n_topis_values:
        X, lda, top_words = reduce_data_by_topics(transformed_data, vectorizer, n_topic)
        for k_clusters in k_cluster_values:
            for clusetring_algo in ['KMeans']:
                if clusetring_algo == 'KMeans':
                    est = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1,
                                 verbose=True)
                elif clusetring_algo == 'DBSCAN':
                    est = DBSCAN()
                cluster(est, X)
                labels = est.labels_
                print(labels.shape)
                cur_settings_evluation = {'algo': clusetring_algo, 'k_clusters': k_clusters, 'n_topics': n_topic,
                                          'calinski_harabaz_score': calinski_harabaz_score(X=X, labels=labels),
                                          'silhouette_score': silhouette_score(X=X, labels=labels, sample_size=2*10e5)
                                          }
                all_settings_evaluation.append(cur_settings_evluation)
                write_results_to_csv(all_settings_evaluation, 'clustering_results/{}_clustering_evalutation.csv'.format(
                    '_'.join(data_set_name.lower().split())))


def write_results_to_csv(results, path):
    import csv
    keys = results[0].keys()
    with open(path, 'wb') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)

def test_topic_extraction(text,data_set_name):
    top_words_by_topic_count = {}
    perplexity_by_topic_count = {}
    transformed_data, vectorizer = transform_data_to_vectors(text)
    for i in range(2, 50):
        X, lda, top_words = reduce_data_by_topics(transformed_data, vectorizer, i)
        top_words_by_topic_count[i] = top_words
        perplexity_by_topic_count[i] = lda.perplexity(transformed_data)
        write_top_words_by_topic_count(top_words_by_topic_count, perplexity_by_topic_count,
                                       'topics_result/top_words_{}.csv'.format('_'.join(data_set_name.lower().split())))
        plt.plot(perplexity_by_topic_count.keys(), perplexity_by_topic_count.values(), 'ro')
        plt.xlabel('N topics')
        plt.ylabel('Perplexity')
        plt.title('{}: Perplexity against number of topics'.format(data_set_name))
        plt.savefig('topics_result/{}_perplexity_{}.png'.format('_'.join(data_set_name.lower().split()),i), dpi=100)



def main():
    # print("Loading dataset...")
    # t0 = time()
    # data_samples = get_comments_data()
    # print("done in %0.3fs." % (time() - t0))
    # transformed_data, vectorizer = transform_data_to_vectors(data_samples)
    # X, lda = reduce_data_by_topics(transformed_data, vectorizer)
    # if use_kmeans:
    #     est = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1,
    #                 verbose=True)
    # else:
    #     est= DBSCAN()
    # cluster(est, X,vectorizer, k=10, lda=lda)
    # test_topic_extraction_twits()
    # test_topic_extraction_comments()
    # tweets_k_means_test()
    fcc_k_means_test()
if __name__ == main():
    main()