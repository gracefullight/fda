from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE

# repository layout: src/week4/main.py and shared assets live in src/assets
# so resolve the assets dir relative to the package src/ directory
ASSET_DIR = Path(__file__).resolve().parent.parent / "assets"
DATAFILE = ASSET_DIR / "imports-85-1.csv"

imports_85 = pd.read_csv(DATAFILE)

print(imports_85.head())

imports_85 = imports_85.replace("?", 0)
# print(imports_85)
print(imports_85.columns)

# pandas hist plot numerical data, we only get 10 histograms
# imports_85.hist(bins=10)
# plt.tight_layout(rect=(0, 0, 1.0, 1.0))
# plt.show()

print(imports_85.dtypes)

# re-code data types of some attributes
# 'normalized-losses' - object because of automatic recognition;
imports_85[["normalized-losses", "bore", "stroke", "horsepower", "peak-rpm", "price"]] = imports_85[
    ["normalized-losses", "bore", "stroke", "horsepower", "peak-rpm", "price"]
].apply(pd.to_numeric)

print(imports_85.dtypes)

# imports_85.hist(bins=10)  # pandas hist plot numerical data, we are getting 6 more histograms.
# plt.tight_layout(rect=(0, 0, 1.0, 1.0))
# plt.show()

# plt.close("all")
# how to plot categorical attributes?
# solution 1: re-code it: replace each category with a number and then plot with the methods for numberical  ----- leave it as a homework:
# solution 2: counting each categories


categorical_features = imports_85.select_dtypes("object").columns
print(categorical_features)
num_cat = len(categorical_features)

# create a wide figure sized by number of categories; keep axes into a flat array
# fig, axes = plt.subplots(1, num_cat, figsize=(4 * max(1, num_cat), 4), squeeze=False)
# axes = axes.ravel()
# for i, categorical_feature in enumerate(categorical_features):
#     imports_85[categorical_feature].value_counts().plot(kind="bar", ax=axes[i])
#     axes[i].set_title(categorical_feature)
# fig.tight_layout(rect=(0, 0, 1.0, 1.0))
# plt.show()

# create a fresh figure and plot boxplots only for numeric columns
# plt.close("all")
numeric_cols = imports_85.select_dtypes(include=[np.number]).columns
# imports_85[numeric_cols].boxplot(rot=90)
# plt.tight_layout(rect=(0, 0, 1.0, 1.0))

print(type(imports_85))

# either one plot for attribute
numerical_features_idx = imports_85.columns.get_indexer(
    imports_85.select_dtypes(["int64", "float64"]).columns
)  # there are 16 in total for this dataset
numerical_features = imports_85.columns[numerical_features_idx]

# plt.close()
# fig, ax = plt.subplots(1, len(numerical_features_idx))
# for i, numerical_feature in enumerate(numerical_features):
#     # print(categorical_feature)
#     imports_85[numerical_feature].plot(kind="box", ax=ax[i], figsize=(60, 5)).set_title(
#         numerical_feature
#     )
# plt.show()

# decipher box plot - with important data points.
# median value
# Q1, Q3 (25th percentile, 75th percentile)
# minimum = Q1 - 1.5*IQR [the length of the box].  {whiskers}
# maximum = Q3 + 1.5*IQR
# outliers - consider if take any data preprocessing steps for them.

# sns.relplot(x="length", y="width", data=imports_85)
# or plot one attributes in terms of groupby the other attributes - consider multiple attributes at once. --- combining categorical and numberical attributes.
# imports_85.boxplot(["length"], by=["body-style"])
# plt.show()

print(imports_85.describe())

# sns.relplot(x="length", y="width", data=imports_85)
# pairplot  --- is quite slow and think about why it is so slow?
#### again plot numerical attributes
# pp = sns.pairplot(
#     imports_85[numerical_features],
#     height=1.8,
#     aspect=1.8,
#     plot_kws=dict(edgecolor="k", linewidth=0.5),
#     diag_kind="kde",
#     diag_kws=dict(fill=True),
# )

# fig = pp.fig
# fig.subplots_adjust(top=0.93, wspace=0.3)
# [think about how to use these plots to assist you do data preprocessing or your classification/prediction tasks.]
# t = fig.suptitle("Import data Pairwise Plots", fontsize=14)

# sns.catplot(x="body-style", y="length", hue="num-of-doors", kind="box", data=imports_85)
# plt.show()


import_embedded = TSNE(
    n_components=2, learning_rate="auto", init="random", perplexity=3
).fit_transform(imports_85[numerical_features])  ### processing numerical data types
print(import_embedded.shape)
print(categorical_features)

df_import_embedded = pd.DataFrame(import_embedded, columns=["Column_A", "Column_B"])
df_import_embedded[categorical_features] = imports_85[categorical_features]

# fig, ax = plt.subplots(1, len(categorical_features_idx))
for i, categorical_feature in enumerate(categorical_features):
    # print(categorical_feature)
    # imports_85[numerical_feature].plot(kind = 'box', ax=ax[i], figsize = (60,5)).set_title(numerical_feature)
    # plt.subplot(1, len(categorical_features_idx), i+1)
    sns.relplot(x="Column_A", y="Column_B", hue=categorical_feature, data=df_import_embedded).set(
        title=categorical_feature
    )
plt.show()

i = 0
print(categorical_features[i])
sns.relplot(x="Column_A", y="Column_B", hue=categorical_features[i], data=df_import_embedded)

print(df_import_embedded.head)
plt.show()
