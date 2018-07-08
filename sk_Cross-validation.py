#!/usr/bin/python
# -*- coding: UTF-8 -*-


#               |    |    |
#              )_)  )_)  )_)
#             )___))___))___)\
#            )____)____)_____)\\
#          _____|____|____|____\\\__
# ---------\                   /---------
# 交叉验证-----超参数确定
# (1)
#训练集-测试集分割
# <editor-fold desc="train_test_split">
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=0)
# </editor-fold>
# (2)
# 交叉验证评价
# <editor-fold desc="交叉验证评价">
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
iris = datasets.load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.4, random_state=0)
# 直接交叉看评价，评价？？怎么定？？
from sklearn.model_selection import cross_val_score
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, iris.data, iris.target, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# 评估函数可以在 metrics 里找找，很全
# 评估相关函数 http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
from sklearn import metrics
scores = cross_val_score(clf, iris.data, iris.target, cv=5, scoring='f1_macro')
# </editor-fold>
# (3)
# 多评价指标交叉验证
# <editor-fold desc="多评价指标交叉验证">
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
# And for multiple metric evaluation, the return value is a dict with the following keys - ['test_<scorer1_name>', 'test_<scorer2_name>', 'test_<scorer...>', 'fit_time', 'score_time']
scoring = ['precision_macro', 'recall_macro']
clf = svm.SVC(kernel='linear', C=1, random_state=0)
scores = cross_validate(clf, iris.data, iris.target, scoring=scoring,
                        cv=5, return_train_score=True)
sorted(scores.keys())
scores['test_recall_macro']
# </editor-fold>
# (4)
# 交叉验证一般假设特征间 独立且一致分布的 （IID），生成在同一时间，且与过去没有依赖,实际很少符合
# <editor-fold desc="KFold 样本分割">
# KFold 样本分割
import numpy as np
from sklearn.model_selection import KFold
X = np.array([[0., 0.], [1., 1.], [-1., -1.], [2., 2.]])
y = np.array([0, 1, 0, 1])
kf = KFold(n_splits=2)
for train, test in kf.split(X):
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
     print("%s %s" % (train, test))

# </editor-fold>
# (5)
#只留一个样本测试 ，LeaveOneOut
# <editor-fold desc="LeaveOneOut">
from sklearn.model_selection import LeaveOneOut
X = [1, 2, 3, 4]
loo = LeaveOneOut()
for train, test in loo.split(X):
    print("%s %s" % (train, test))

# </editor-fold>
# (6)
# 分成抽样K折验证，使得不同类别的样本比例和整体一种
# StratifiedKFold、StratifiedShuffleSplit
# <editor-fold desc="StratifiedKFold">
from sklearn.model_selection import StratifiedKFold
X = np.ones(10)
y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
skf = StratifiedKFold(n_splits=3)
for train, test in skf.split(X, y):
   print("%s %s" % (train, test))
skf
# </editor-fold>
# (7)
# 组交叉验证
# 用于来自不同组的同样特征，例如不同病人身上的一系列特征，在针对不同类别的疾病上
# <editor-fold desc="GroupKFold">
# GroupKFold
from sklearn.model_selection import GroupKFold
X = [0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 8.8, 9, 10]
y = ["a", "b", "b", "b", "c", "c", "c", "d", "d", "d"]
groups = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
gkf = GroupKFold(n_splits=3)
for train, test in gkf.split(X, y, groups=groups):
    print("%s %s" % (train, test))
gkf
# </editor-fold>
# (8)
# 时间序列样本分割
# TimeSeriesSplit
# <editor-fold desc="时间序列样本分割">
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4, 5, 6])
tscv = TimeSeriesSplit(n_splits=3)
print(tscv)
TimeSeriesSplit(max_train_size=None, n_splits=3)
for train, test in tscv.split(X):
    print("%s %s" % (train, test))
# [0 1 2] [3]
# [0 1 2 3] [4]
# [0 1 2 3 4] [5]
# </editor-fold>


# (9)
# 评估函数
precision_recall_curve(y_true, probas_pred)
# Compute precision-recall pairs for different probability thresholds
roc_curve(y_true, y_score[, pos_label, …])
# Compute Receiver operating characteristic (ROC)

# Others also work in the multiclass case:
cohen_kappa_score(y1, y2[, labels, weights, …])
# Cohen’s kappa: a statistic that measures inter-annotator agreement.
confusion_matrix(y_true, y_pred[, labels, …])
# Compute confusion matrix to evaluate the accuracy of a classification
hinge_loss(y_true, pred_decision[, labels, …])
# Average hinge loss (non-regularized)
matthews_corrcoef(y_true, y_pred[, …])
# Compute the Matthews correlation coefficient (MCC)

# Some also work in the multilabel case:
accuracy_score(y_true, y_pred[, normalize, …])
# Accuracy classification score.
classification_report(y_true, y_pred[, …])
# Build a text report showing the main classification metrics
f1_score(y_true, y_pred[, labels, …])
# Compute the F1 score, also known as balanced F-score or F-measure
fbeta_score(y_true, y_pred, beta[, labels, …])
# Compute the F-beta score
hamming_loss(y_true, y_pred[, labels, …])
# Compute the average Hamming loss.
jaccard_similarity_score(y_true, y_pred[, …])
# Jaccard similarity coefficient score
log_loss(y_true, y_pred[, eps, normalize, …])
# Log loss, aka logistic loss or cross-entropy loss.
precision_recall_fscore_support(y_true, y_pred)
# Compute precision, recall, F-measure and support for each class
precision_score(y_true, y_pred[, labels, …])
# Compute the precision
recall_score(y_true, y_pred[, labels, …])
# Compute the recall
zero_one_loss(y_true, y_pred[, normalize, …])
# Zero-one classification loss.

# And some work with binary and multilabel (but not multiclass) problems:
average_precision_score(y_true, y_score[, …])
# Compute average precision (AP) from prediction scores
roc_auc_score(y_true, y_score[, average, …])
# Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.

