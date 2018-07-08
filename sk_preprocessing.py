#!/usr/bin/python
# -*- coding: UTF-8 -*-


#               |    |    |
#              )_)  )_)  )_)
#             )___))___))___)\
#            )____)____)_____)\\
#          _____|____|____|____\\\__
# ---------\                   /---------
# sklearn.preprocess 预处理部分

# （1）
# Scaler  线性缩放
# Transformer  非线性缩放
# Normalizer   标准化，是针对样本的，不是针对单个特征的（refers to a per sample transformation instead of a per feature transformation.refers to a per sample transformation instead of a per feature transformation.）
# 将每一行的规整为1阶范数为1的向量，1
# 阶范数即所有值绝对值之和。
# RobustScaler、QuantileTransformer、Normalizer 比较好用
# <editor-fold desc="缩放/标准化">
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm

from sklearn.preprocessing import MinMaxScaler
# MinMaxScaler将每一维特征线性地映射到指定的区间，通常是[0, 1]
# 注意因为零值转换后可能变为非零值，所以即便为稀疏输入，输出也可能为稠密向量。
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
# MaxAbsScaler将每一维的特征变换到[-1, 1]闭区间上，通过除以每一维特征上的最大的绝对值，
# 它不会平移整个分布，也不会破坏原来每一个特征向量的稀疏性。

from sklearn.preprocessing import StandardScaler
# 0均值单位标准差 withStd: 默认为真。将数据标准化到单位标准差。
# withMean: 默认为假。是否变换为0均值。 (此种方法将产出一个稠密输出，所以不适用于稀疏输入。)
from sklearn.preprocessing import RobustScaler
# 对奇异值特别鲁班？？！！
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing.data import QuantileTransformer

from sklearn.datasets import fetch_california_housing
dataset = fetch_california_housing()
X_full, y_full = dataset.data, dataset.target

# Take only 2 features to make visualization easier
# Feature of 0 has a long tail distribution.
# Feature 5 has a few but very large outliers.
X = X_full[:, [0, 5]]
distributions = [
    ('Unscaled data', X),
    ('Data after standard scaling',
        StandardScaler().fit_transform(X)),
    ('Data after min-max scaling',
        MinMaxScaler().fit_transform(X)),
    ('Data after max-abs scaling',
        MaxAbsScaler().fit_transform(X)),
    ('Data after robust scaling',
        RobustScaler(quantile_range=(25, 75)).fit_transform(X)),
    ('Data after quantile transformation (uniform pdf)',
        QuantileTransformer(output_distribution='uniform')
        .fit_transform(X)),
    ('Data after quantile transformation (gaussian pdf)',
        QuantileTransformer(output_distribution='normal')
        .fit_transform(X)),
    ('Data after sample-wise L2 normalizing',
        Normalizer().fit_transform(X))
]

def create_axes(title="", figsize=(16, 6)):
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title)

    # define the axis for the first plot
    left, width = 0.1, 0.22
    bottom, height = 0.1, 0.7
    bottom_h = height + 0.15
    left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.1]
    rect_histy = [left_h, bottom, 0.05, height]

    ax_scatter = plt.axes(rect_scatter)
    ax_histx = plt.axes(rect_histx)
    ax_histy = plt.axes(rect_histy)

    # define the axis for the zoomed-in plot
    left = width + left + 0.2
    left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.1]
    rect_histy = [left_h, bottom, 0.05, height]

    ax_scatter_zoom = plt.axes(rect_scatter)
    ax_histx_zoom = plt.axes(rect_histx)
    ax_histy_zoom = plt.axes(rect_histy)

    # define the axis for the colorbar
    left, width = width + left + 0.13, 0.01

    rect_colorbar = [left, bottom, width, height]
    ax_colorbar = plt.axes(rect_colorbar)

    return ((ax_scatter, ax_histy, ax_histx),
            (ax_scatter_zoom, ax_histy_zoom, ax_histx_zoom),
            ax_colorbar)
def plot_distribution(axes, X, y, hist_nbins=50, title="",
                      x0_label="", x1_label=""):
    ax, hist_X1, hist_X0 = axes

    ax.set_title(title)
    ax.set_xlabel(x0_label)
    ax.set_ylabel(x1_label)

    # The scatter plot
    colors = cm.plasma_r(y)
    ax.scatter(X[:, 0], X[:, 1], alpha=0.5, marker='o', s=5, lw=0, c=colors)

    # Removing the top and the right spine for aesthetics
    # make nice axis layout
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))

    # Histogram for axis X1 (feature 5)
    hist_X1.set_ylim(ax.get_ylim())
    hist_X1.hist(X[:, 1], bins=hist_nbins, orientation='horizontal',
                 color='grey', ec='grey')
    hist_X1.axis('off')

    # Histogram for axis X0 (feature 0)
    hist_X0.set_xlim(ax.get_xlim())
    hist_X0.hist(X[:, 0], bins=hist_nbins, orientation='vertical',
                 color='grey', ec='grey')
    hist_X0.axis('off')

def make_plot(item_idx):
    title, X = distributions[item_idx]
    ax_zoom_out, ax_zoom_in, ax_colorbar = create_axes(title)
    axarr = (ax_zoom_out, ax_zoom_in)
    plot_distribution(axarr[0], X, y, hist_nbins=200,
                      x0_label="Median Income",
                      x1_label="Number of households",
                      title="Full data")

    # zoom-in
    zoom_in_percentile_range = (0, 99)#取99%的数据，排除1%的奇异值
    cutoffs_X0 = np.percentile(X[:, 0], zoom_in_percentile_range)
    cutoffs_X1 = np.percentile(X[:, 1], zoom_in_percentile_range)

    non_outliers_mask = (
        np.all(X > [cutoffs_X0[0], cutoffs_X1[0]], axis=1) &
        np.all(X < [cutoffs_X0[1], cutoffs_X1[1]], axis=1))
    plot_distribution(axarr[1], X[non_outliers_mask], y[non_outliers_mask],
                      hist_nbins=50,
                      x0_label="Median Income",
                      x1_label="Number of households",
                      title="Zoom-in")

    norm = mpl.colors.Normalize(y_full.min(), y_full.max())
    mpl.colorbar.ColorbarBase(ax_colorbar, cmap=cm.plasma_r,
                              norm=norm, orientation='vertical',
                              label='Color mapping for values of y')

# 将输出映射在【0,1】之间
y = minmax_scale(y_full)
make_plot(0) # 说明奇异值影响很大，将y范围固定在0-6之间的效果
make_plot(1) # 说明StandardScaler，去中心化，缩放到标准方差，没有将离群点去除，离群点依然产生影响
make_plot(2) #  说明 MinMaxScaler 将所有样本压缩在0-1之间，离群点没去，所以大量数据集中在很小一个区间
make_plot(3) # 说明 MaxAbsScaler 绝对值压缩在0-1之间

make_plot(4)
# RobustScaler  定心和缩放统计是基于百分位数的，
# 因此不会受到一些非常大的边缘异常值的影响。
# 因此，转换后的特征值的结果范围要比之前的标量大，
# 而且更重要的是，它们是近似的:对于这两个特征，
# 大多数转换后的值都位于一个[- 2,3]的范围，如图缩放后的图所示。
# 注意，异常值本身仍然存在于转换后的数据中

make_plot(5)
# QuantileTransformer (uniform output)均值输出 0-1之间
# 使每个特征的概率密度函数映射成均匀分布。所有的数据将被映射到范围[0,1]
# 对异常值具有强大的鲁棒性

make_plot(6)
# QuantileTransformer (Gaussian output)高斯输出
# 有一个额外的output_distribution参数，允许匹配高斯分布而不是均匀分布。

make_plot(7)
# Normalizer
# 每个样本的矢量重新调整为单位范数，与样本的分布无关。
#  所有的样本都被映射到单位圆上（只有2个特征的情况）
#  衡量一对样本的相似性，那么这个过程将非常有用
# </editor-fold>
# （2）
#    画出 2个特征和y的分布图 很重要！！！！！
# <editor-fold desc="画出 2个特征和y的分布图 很重要！！">
def create_axes(title="", figsize=(16, 6)):
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title)

    # define the axis for the first plot
    left, width = 0.1, 0.22
    bottom, height = 0.1, 0.7
    bottom_h = height + 0.15
    left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.1]
    rect_histy = [left_h, bottom, 0.05, height]

    ax_scatter = plt.axes(rect_scatter)
    ax_histx = plt.axes(rect_histx)
    ax_histy = plt.axes(rect_histy)

    # define the axis for the zoomed-in plot
    left = width + left + 0.2
    left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.1]
    rect_histy = [left_h, bottom, 0.05, height]

    ax_scatter_zoom = plt.axes(rect_scatter)
    ax_histx_zoom = plt.axes(rect_histx)
    ax_histy_zoom = plt.axes(rect_histy)

    # define the axis for the colorbar
    left, width = width + left + 0.13, 0.01

    rect_colorbar = [left, bottom, width, height]
    ax_colorbar = plt.axes(rect_colorbar)

    return ((ax_scatter, ax_histy, ax_histx),
            (ax_scatter_zoom, ax_histy_zoom, ax_histx_zoom),
            ax_colorbar)

def plot_distribution(axes, X, y, hist_nbins=50, title="",
                      x0_label="", x1_label=""):
    ax, hist_X1, hist_X0 = axes

    ax.set_title(title)
    ax.set_xlabel(x0_label)
    ax.set_ylabel(x1_label)

    # The scatter plot
    colors = cm.plasma_r(y)
    ax.scatter(X[:, 0], X[:, 1], alpha=0.5, marker='o', s=5, lw=0, c=colors)

    # Removing the top and the right spine for aesthetics
    # make nice axis layout
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))

    # Histogram for axis X1 (feature 5)
    hist_X1.set_ylim(ax.get_ylim())
    hist_X1.hist(X[:, 1], bins=hist_nbins, orientation='horizontal',
                 color='grey', ec='grey')
    hist_X1.axis('off')

    # Histogram for axis X0 (feature 0)
    hist_X0.set_xlim(ax.get_xlim())
    hist_X0.hist(X[:, 0], bins=hist_nbins, orientation='vertical',
                 color='grey', ec='grey')
    hist_X0.axis('off')
def make_plot_feature2(X,y,f_str1="fea_1",f_str2="fea_2"):
    ax_zoom_out, ax_zoom_in, ax_colorbar = create_axes()
    axarr = (ax_zoom_out, ax_zoom_in)
    plot_distribution(axarr[0], X, y, hist_nbins=200,
                      x0_label=f_str1,
                      x1_label=f_str2,
                      title="")
    # zoom-in
    zoom_in_percentile_range = (0, 99)#取99%的数据，排除1%的奇异值
    cutoffs_X0 = np.percentile(X[:, 0], zoom_in_percentile_range)
    cutoffs_X1 = np.percentile(X[:, 1], zoom_in_percentile_range)

    non_outliers_mask = (
        np.all(X > [cutoffs_X0[0], cutoffs_X1[0]], axis=1) &
        np.all(X < [cutoffs_X0[1], cutoffs_X1[1]], axis=1))
    plot_distribution(axarr[1], X[non_outliers_mask], y[non_outliers_mask],
                      hist_nbins=50,
                      x0_label=f_str1,
                      x1_label=f_str2,
                      title="Zoom-in")
    norm = mpl.colors.Normalize(y_full.min(), y_full.max())
    mpl.colorbar.ColorbarBase(ax_colorbar, cmap=cm.plasma_r,
                              norm=norm, orientation='vertical',
                              label='Color mapping for values of y')
XX = RobustScaler(quantile_range=(25, 75)).fit_transform(X)
make_plot_feature2(XX,y)
# </editor-fold>
# （3）
# 二值化 Binarizer
# <editor-fold desc="二值化 Binarizer">
from sklearn import preprocessing
X = [[ 1., -1.,  2.],
    [ 2.,  0.,  0.],
    [ 0.,  1., -1.]]
binarizer = preprocessing.Binarizer().fit(X)
binarizer.transform(X)
binarizer = preprocessing.Binarizer(threshold=1.1)
binarizer.transform(X)
# </editor-fold>
# （4）
# OneHotEncoder
# 将具有m个可能值的分类特征转换为m个二进制特征，只有一个活动特征。
# <editor-fold desc="将具有m个可能值的分类特征转换为m个二进制特征，只有一个活动特征。">
enc = preprocessing.OneHotEncoder()
enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
enc.transform([[0, 1, 3]]).toarray()
# Out[49]: array([[1., 0.,(训练集只有0,1)     0., 1., 0.,（只有0,1,2）   0., 0., 0., 1.（只有0,1,2,3）]])
enc = preprocessing.OneHotEncoder(n_values=[2, 3, 4])
enc.fit([[1, 2, 3], [0, 2, 0]])
enc.transform([[1, 0, 0]]).toarray()
# array([[ 0.,  1.,（0,1）    1.,  0.,  0.,（0，X1，X2）  1.,  0.,  0.,  0. （0，X1，X2，X3）]])
# </editor-fold>
# （5）
# 缺失值处理
# <editor-fold desc="缺失值处理">
import numpy as np
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit([[1, 2], [np.nan, 3], [7, 6]])
X = [[np.nan, 2], [6, np.nan], [7, 6]]
print(imp.transform(X))

import scipy.sparse as sp
X = sp.csc_matrix([[1, 2], [0, 3], [7, 6]])
imp = Imputer(missing_values=0, strategy='mean', axis=0)
imp.fit(X)
X_test = sp.csc_matrix([[0, 2], [6, 0], [7, 6]])
print(imp.transform(X_test))
# </editor-fold>
# （6）
# 使用均值或中值补全缺失值，中值很强！！   使用Pipeline构建流，并用cross_val_score交叉验证
# <editor-fold desc="使用均值或中值补全缺失值，中值很强！！   使用Pipeline构建流，并用cross_val_score交叉验证">
import numpy as np

from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
# 可以将许多算法模型串联起来，比如将特征提取、归一化、分类组织在一起
# 形成一个典型的机器学习问题工作流
# 可以直接调用fit和predict方法来对pipeline中的所有算法模型进行训练和预测。

from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score

rng = np.random.RandomState(0)

dataset = load_boston()
X_full, y_full = dataset.data, dataset.target
n_samples = X_full.shape[0]
n_features = X_full.shape[1]

# Estimate the score on the entire dataset, with no missing values
estimator = RandomForestRegressor(random_state=0, n_estimators=100)
score = cross_val_score(estimator, X_full, y_full).mean()
print("Score with the entire dataset = %.2f" % score)

# Add missing values in 75% of the lines
missing_rate = 0.75
n_missing_samples = int(np.floor(n_samples * missing_rate))
missing_samples = np.hstack((np.zeros(n_samples - n_missing_samples,
                                      dtype=np.bool),
                             np.ones(n_missing_samples,
                                     dtype=np.bool)))
rng.shuffle(missing_samples)
missing_features = rng.randint(0, n_features, n_missing_samples)

# Estimate the score without the lines containing missing values
X_filtered = X_full[~missing_samples, :]
y_filtered = y_full[~missing_samples]
estimator = RandomForestRegressor(random_state=0, n_estimators=100)
score = cross_val_score(estimator, X_filtered, y_filtered).mean()
print("Score without the samples containing missing values = %.2f" % score)

# Estimate the score after imputation of the missing values
X_missing = X_full.copy()
X_missing[np.where(missing_samples)[0], missing_features] = 0
y_missing = y_full.copy()
estimator = Pipeline([("imputer", Imputer(missing_values=0,
                                          strategy="mean",
                                          axis=0)),
                      ("forest", RandomForestRegressor(random_state=0,
                                                       n_estimators=100))])
score = cross_val_score(estimator, X_missing, y_missing).mean()
print("Score after imputation of the missing values = %.2f" % score)
# </editor-fold>
# （7）
# 使用多项式生成交叉特征值
# <editor-fold desc="使用多项式生成交叉特征值">
from sklearn.preprocessing import PolynomialFeatures
X = np.arange(9).reshape(3, 3)
poly = PolynomialFeatures(degree=3, interaction_only=True)
poly.fit_transform(X)
# (1, X_1, X_2, X_3, X_1X_2, X_1X_3, X_2X_3, X_1X_2X_3).
# </editor-fold>
# （8）
# 构建自定义映射
# <editor-fold desc="构建自定义映射">
from sklearn.preprocessing import FunctionTransformer
transformer = FunctionTransformer(np.log1p)
X = np.array([[0, 1], [2, 3]])
transformer.transform(X)
# </editor-fold>


