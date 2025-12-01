import numpy as np
import sys
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, f1_score
from munkres import Munkres
import faiss
def evaluate(label, pred):
    nmi = metrics.normalized_mutual_info_score(label, pred)
    ari = metrics.adjusted_rand_score(label, pred)
    f = metrics.fowlkes_mallows_score(label, pred)
    pred_adjusted = get_y_preds(label, pred, len(set(label)))
    acc = metrics.accuracy_score(pred_adjusted, label)
    return nmi, ari, f, acc

def kmeans(feature_vec, class_num):
    d = feature_vec.shape[1]
    kmeans =  faiss.Kmeans(d, class_num, nredo=10, niter=300, gpu=False, spherical=True, seed=0)
    kmeans.train(feature_vec)
    D, I = kmeans.index.search(feature_vec, 1)
    cluster = [int(n[0]) for n in I]
    return cluster

def clustering(x_list, y):
    """Get scores of clustering"""
    n_clusters = np.size(np.unique(y))

    x_final_concat = np.concatenate(x_list[:], axis=1)
    kmeans_assignments, km = get_cluster_sols(x_final_concat, ClusterClass=KMeans, n_clusters=n_clusters,
                                              init_args={'n_init': 10})
    if np.min(y) == 1:
        y = y - 1
    scores, _ = clustering_metric(y, kmeans_assignments, n_clusters)

    ret = {}
    ret['kmeans'] = scores
    return ret

def clustering_index(x_list,y):
    """Get scores of clustering"""
    n_clusters = np.size(np.unique(y))

    x_final_concat = np.concatenate(x_list[:], axis=1)
    kmeans_assignments, km = get_cluster_sols(x_final_concat, ClusterClass=KMeans, n_clusters=n_clusters,
                                              init_args={'n_init': 10})
    if np.min(y) == 1:
        y = y - 1

    return kmeans_assignments


def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))

    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    clusterLabels = np.zeros(n_clusters)
    for i in range(n_clusters):
        clusterLabels[i] = indices[i][1]
    return clusterLabels


def get_y_preds(y_true, cluster_assignments, n_clusters):
    confusion_matrix = metrics.confusion_matrix(y_true, cluster_assignments, labels=None)
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)

    if np.min(cluster_assignments) != 0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred


def classification_metric(y_true, y_pred, average='macro', verbose=True, decimals=4):
    """Get classification metric"""
    # confusion matrix
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    # ACC
    accuracy = metrics.accuracy_score(y_true, y_pred)
    accuracy = np.round(accuracy, decimals)

    # precision
    precision = metrics.precision_score(y_true, y_pred, average=average)
    precision = np.round(precision, decimals)

    # recall
    recall = metrics.recall_score(y_true, y_pred, average=average)
    recall = np.round(recall, decimals)

    # F-score
    f_score = metrics.f1_score(y_true, y_pred, average=average)
    f_score = np.round(f_score, decimals)

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f_measure': f_score}, confusion_matrix


def clustering_metric(y_true, y_pred, n_clusters, verbose=True, decimals=4):
    """Get clustering metric"""
    y_pred_ajusted = get_y_preds(y_true, y_pred, n_clusters)

    classification_metrics, confusion_matrix = classification_metric(y_true, y_pred_ajusted)

    # AMI
    ami = metrics.adjusted_mutual_info_score(y_true, y_pred)
    ami = np.round(ami, decimals)
    # NMI
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
    nmi = np.round(nmi, decimals)
    # ARI
    ari = metrics.adjusted_rand_score(y_true, y_pred)
    ari = np.round(ari, decimals)

    return dict({'AMI': ami, 'NMI': nmi, 'ARI': ari}, **classification_metrics), confusion_matrix


def get_cluster_sols(x, cluster_obj=None, ClusterClass=None, n_clusters=None, init_args={}):
    assert not (cluster_obj is None and (ClusterClass is None or n_clusters is None))
    cluster_assignments = None
    if cluster_obj is None:
        cluster_obj = ClusterClass(n_clusters, **init_args)
        for _ in range(10):
            try:
                cluster_obj.fit(x)
                break
            except:
                print("Unexpected error:", sys.exc_info())
        else:
            return np.zeros((len(x),)), cluster_obj

    cluster_assignments = cluster_obj.predict(x)
    return cluster_assignments, cluster_obj

def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)]

def vote(lsd1, lsd2, label, n=1):
    F_h_h = np.dot(lsd2, np.transpose(lsd1))
    gt_list = []
    label = label.reshape(len(label), 1)
    for num in range(n):
        F_h_h_argmax = np.argmax(F_h_h, axis=1)
        F_h_h_onehot = convert_to_one_hot(F_h_h_argmax, len(label))
        F_h_h = F_h_h - np.multiply(F_h_h, F_h_h_onehot)
        gt_list.append(np.dot(F_h_h_onehot, label))
    gt_ = np.array(gt_list).transpose(2, 1, 0)[0].astype(np.int64)
    count_list = []
    count_list.append([np.argmax(np.bincount(gt_[i])) for i in range(lsd2.shape[0])])
    gt_pre = np.array(count_list)
    return gt_pre.transpose()

def classify(lsd1, lsd2, label):
    F_h_h = np.dot(lsd2, np.transpose(lsd1))
    label = label.reshape(len(label), 1) - 1
    enc = OneHotEncoder()
    a = enc.fit_transform(label)
    label_onehot = a.toarray()
    label_num = np.sum(label_onehot, axis=0)
    F_h_h_sum = np.dot(F_h_h, label_onehot)
    F_h_h_mean = F_h_h_sum / label_num
    label_pre = np.argmax(F_h_h_mean, axis=1) + 1
    return label_pre

def svm_classify(latent_fusion_train, labels_train, latent_fusion_test, labels_test):
    clf = svm.SVC()  # class 
    clf.fit(latent_fusion_train, labels_train)  # training the svc model
    label_pre = clf.predict(latent_fusion_test)
    scores = accuracy_score(labels_test, label_pre)
    precision = precision_score(labels_test, label_pre, average='macro', labels=np.unique(labels_train))
    precision = np.round(precision, 2)
    f_score = f1_score(labels_test, label_pre, average='macro', labels=np.unique(labels_train))
    f_score = np.round(f_score, 2)
    return scores,precision,f_score

def score(labels_test, label_pre):
    return accuracy_score(labels_test, label_pre)

def svm_classify_acc(latent_fusion_train, labels_train, latent_fusion_test, labels_test):
    clf = svm.SVC()  # class 
    clf.fit(latent_fusion_train, labels_train)  # training the svc model
    label_pre = clf.predict(latent_fusion_test)
    scores = accuracy_score(labels_test, label_pre)
    return scores