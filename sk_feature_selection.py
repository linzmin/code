#!/usr/bin/python
# -*- coding: UTF-8 -*-


#               |    |    |
#              )_)  )_)  )_)
#             )___))___))___)\
#            )____)____)_____)\\
#          _____|____|____|____\\\__
# ---------\                   /---------
# sklearn.feature_selection  特征选择部分-----主要用于降维
# (1)
# 通过统计特征方差，去除低方差特征 var=p(1-p)  p目标出现概率
# <editor-fold desc="通过统计特征方差，去除低方差特征">
from sklearn.feature_selection import VarianceThreshold
X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))   #去除方差小于.8 * (1 - .8)的特征
# </editor-fold>
# {2}
# 单变量统计特征选择chi2 、mutual_info
# <editor-fold desc=" 单变量统计特征选择chi2 、mutual_info">
# SelectKBest 保留K个特征
# SelectPercentile 保留前k%的特征
# 回归: f_regression F值 mutual_info_regression 互信息
# 分类: chi2 卡方（残差和与期望之比） f_classif F值（方差齐性、方差比率，对正态性敏感） mutual_info_classif 互信息（即特征出现对分类正确的影响）
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
For regression: f_regression, mutual_info_regression
For classification: chi2, f_classif, mutual_info_classif
iris = load_iris()
X, y = iris.data, iris.target
X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
# </editor-fold>
# {3}
# 回归特征消除( Recursive feature elimination , RFE)
# <editor-fold desc="回归特征消除">
# 通过feature_importances_属性获得特征重要性。然后，
# 从当前的特性集中删除最不重要的特性。该过程在修剪集上递归地重复，
# 直到需要选择的特性数量为ev

import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE

from sklearn.datasets import make_classification

# Build a classification task using 3 informative features
X, y = make_classification(n_samples=1000, n_features=25, n_informative=3,
                           n_redundant=2, n_repeated=0, n_classes=8,
                           n_clusters_per_class=1, random_state=0)
# Create the RFE object and compute a cross-validated score.
svc = SVC(kernel="linear")
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
              scoring='accuracy')
rfecv.fit(X, y)
print("Optimal number of features : %d" % rfecv.n_features_)
rfecv.ranking_
rfecv.


# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
rfe.fit(X, y)
ranking = rfe.ranking_.reshape(X.shape)

# Plot pixel ranking
plt.matshow(ranking, cmap=plt.cm.Blues)
plt.colorbar()
plt.title("Ranking of pixels with RFE")
plt.show()
# </editor-fold>
# (4)
# SelectFromModel
# 从模型效果上选择特征
# <editor-fold desc="从模型效果上选择特征">
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston


# Load the boston dataset.
boston = load_boston()
X, y = boston['data'], boston['target']

# We use the base estimator LassoCV since the L1 norm promotes sparsity of features.
clf = LassoCV()
# Set a minimum threshold of 0.25
sfm = SelectFromModel(clf, threshold=0.25)
sfm.fit(X, y)
n_features = sfm.transform(X).shape[1]

# 一步步增加阈值，直到特征维数降为2
while n_features > 2:
    sfm.threshold += 0.1
    X_transform = sfm.transform(X)
    n_features = X_transform.shape[1]

# Plot the selected two features from X.
plt.title(
    "Features selected from Boston using SelectFromModel with "
    "threshold %0.3f." % sfm.threshold)
feature1 = X_transform[:, 0]
feature2 = X_transform[:, 1]
plt.plot(feature1, feature2, 'r.')
plt.xlabel("Feature number 1")
plt.ylabel("Feature number 2")
plt.ylim([np.min(feature2), np.max(feature2)])
plt.show()
# </editor-fold>
# (5)
# Linear model， L1正则化
# <editor-fold desc="L1正则化 选择特征">
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
iris = load_iris()
X, y = iris.data, iris.target
clf = ExtraTreesClassifier()
clf = clf.fit(X, y)
clf.feature_importances_
model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X)
# </editor-fold>

# 基于树的特征选择
# <editor-fold desc="基于树的特征选择">
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
iris = load_iris()
X, y = iris.data, iris.target
X.shape
clf = ExtraTreesClassifier()
clf = clf.fit(X, y)
clf.feature_importances_
model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X)
X_new.shape


import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

# Build a classification task using 3 informative features
X, y = make_classification(n_samples=1000,
                           n_features=10,
                           n_informative=3,
                           n_redundant=0,
                           n_repeated=0,
                           n_classes=2,
                           random_state=0,
                           shuffle=False)

# B构建随机森林，并统计特征重要性
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)
forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()
# </editor-fold>








