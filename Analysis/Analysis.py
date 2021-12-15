import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time

from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score, roc_curve, auc, recall_score, roc_auc_score

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

acc = []
precision = []
f1_score = []
auc = []
recall = []
models = []
models = ['Logistic regression', 'KNN',
          'Decision Tree', 'Random Forest', 'SVM']


def clean(dataset, list_cols):
    for col in list_cols:
        temp = [float(x) for x in dataset[col] if x != "?"]
        mean = np.mean(temp)
        std = np.std(temp)
        cut_off = std * 3
        lower, upper = mean - cut_off, mean + cut_off

        outliers = [x for x in temp if x < lower or x > upper]
        for i in dataset[col]:
            if i == "?":
                dataset[col][dataset.index[dataset[col] == i]] = mean
            elif float(i) in outliers:
                dataset.drop(dataset.index[dataset[col] == i])
        data[col] = data[col].astype("float")
    return dataset


def eval_metrics(y_test, y_pred):

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    A = auc(fpr, tpr)
    roc = roc_auc_score(y_test, y_pred)
    #G = (2*roc) - 1
    assert (len(y_test) == len(y_pred))
    all = np.asarray(
        np.c_[y_test, y_pred, np.arange(len(y_test))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(y_test) + 1) / 2.
    gini = giniSum / len(y_test)
    # return giniSum / len(actual)

    return accuracy*100, cm, precision, recall, f1, A, roc, gini


data = pd.read_csv("/content/Thyroid Dataset.csv")

drop_list = ["TSH measured", "T3 measured", "TT4 measured",
             "T4U measured", "FTI measured", "TBG measured", "TBG", "referral source"]
data = data.drop(drop_list, axis=1)
list_cols = ["TSH", "T3", "TT4", "TT4", "T4U", "FTI"]
data = clean(data, list_cols)

columns = [x for x in data.columns if x not in list_cols]

for col in columns:
    print("In ", col, len(data.index[data[col] == "?"]), "rows were deleted")
    data.drop(data.index[data[col] == "?"], inplace=True)
    if col == "sex":
        # Changes the Value type of column 'sex' to String (Before Object)
        data[col] = data[col].astype("string")
    elif col == "binaryClass":
        # Changes the Value type of column 'binaryClass' to int (Before Object)
        data.replace("P", "1", inplace=True)
        data.replace("N", "0", inplace=True)
        data[col] = data[col].astype("int")
    else:
        # Changes the Value type of remaining  column to int (Before Object)
        data.replace("f", "0", inplace=True)
        data.replace("F", "0", inplace=True)
        data.replace("T", "1", inplace=True)
        data.replace("t", "1", inplace=True)
        data[col] = data[col].astype("int")

# Gives the Corelation Heat map for each of our feature variables
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(data.corr(), annot=True, linewidths=.5, ax=ax)

# Correlation with output variable
cor = data.corr()
cor_target = abs(cor['binaryClass'])
# Selecting highly correlated features
relevant_features = cor_target[cor_target > 0.02]

rel_feat = ['on thyroxine', 'on antithyroid medication', 'pregnant',
            'query hypothyroid', 'goitre', 'psych', 'TSH', 'TT4', 'T3', 'FTI', 'binaryClass']
data_list = list(data)
irrel_feat = [item for item in data_list if item not in rel_feat]
data = data.drop(irrel_feat, axis=1)

X = data.iloc[:, :-1:]
y = data.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

lr = LogisticRegression()
lr.fit(x_train, y_train)
pred = lr.predict(x_test)
acc_lr, cm_lr, precision_lr, rec_lr, f1_lr, areaUnderCurve_lr, roc_lr, gini_lr = eval_metrics(
    y_test, pred)

print(
    f"The metrics for Logistic regression are: \n Accuracy: {acc_lr}% \nConfusion Matrix: {cm_lr} \nPrecision: {precision_lr} \nRecall: {rec_lr} \nF1 score: {f1_lr} \nArea Under Curve: {areaUnderCurve_lr}, \n ROC: {roc_lr} \n Gini: {gini_lr}")

# K Nearest Neighbours

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred_knn = knn.predict(x_test)
acc_knn, cm_knn, precision_knn, rec_knn, f1_knn, areaUnderCurve_knn, roc_knn, gini_knn = eval_metrics(
    y_test, y_pred_knn)

print(f"The metrics for KNN are: \n Accuracy: {acc_knn}% \nConfusion Matrix: {cm_knn} \nPrecision: {precision_knn} \nRecall: {rec_knn} \nF1 score: {f1_knn} \nArea Under Curve: {areaUnderCurve_knn}\nROC: {roc_knn} \nGini {gini_knn}")

# Decision Tree

start = time.time()
dt = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=5,
                                 min_samples_leaf=6, max_features='auto', random_state=50)
dt.fit(x_train, y_train)
y_pred_dt = dt.predict(x_test)
end = time.time()
acc_dt, cm_dt, precision_dt, rec_dt, f1_dt, areaUnderCurve_dt, roc_dt, gini_dt = eval_metrics(
    y_test, y_pred_dt)

print(
    f"The metrics for Decision Tree are: \n Accuracy: {acc_dt}% \nConfusion Matrix: {cm_dt} \nPrecision: {precision_dt} \nRecall: {rec_dt} \nF1 score: {f1_dt} \nArea Under Curve: {areaUnderCurve_dt}\nROC: {roc_dt} \n Gini {gini_dt}")

print("Total comuptation time =", (end-start))

# Random  Forest

start = time.time()
rf = RandomForestClassifier(n_estimators=50, criterion='entropy', min_samples_split=5,
                            min_samples_leaf=6, max_features='auto', random_state=50)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)
end = time.time()
acc_rf, cm_rf, precision_rf, rec_rf, f1_rf, areaUnderCurve_rf, roc_rf, gini_rf = eval_metrics(
    y_test, y_pred_rf)

print(
    f"The metrics for Random Forest are: \n Accuracy: {acc_rf}% \nConfusion Matrix: {cm_rf} \nPrecision: {precision_rf} \nRecall: {rec_rf} \nF1 score: {f1_rf} \nArea Under Curve: {areaUnderCurve_rf} \nROC: {roc_rf} \n Gini {gini_rf}")
print("Total comuptation time =", (end-start))

# Support Vector Machine

clf = svm.SVC(kernel='linear')  # Linear Kernel

clf.fit(x_train, y_train)
y_pred_rf = clf.predict(x_test)

acc, cm, precision, rec, f1, areaUnderCurve, roc, gini = eval_metrics(
    y_test, y_pred_rf)

print(
    f"The metrics for SVM (with linear kernel) are: \n Accuracy: {acc}% \nConfusion Matrix: {cm} \nPrecision: {precision} \nRecall: {rec} \nF1 score: {f1} \nArea Under Curve: {areaUnderCurve} \nROC: {roc} \nGini: {gini}")

clf_rbf = svm.SVC(kernel='rbf')  # RBF Kernel
clf_rbf.fit(x_train, y_train)
y_pred_rbf = clf_rbf.predict(x_test)

acc_svm, cm_svm, precision_svm, rec_svm, f1_svm, areaUnderCurve_svm, roc_svm, gini_svm = eval_metrics(
    y_test, y_pred_rbf)

print(
    f"\nThe metrics for SVM (with RBF Kernel) are: \n Accuracy: {acc_svm}% \nConfusion Matrix: {cm_svm} \nPrecision: {precision_svm} \nRecall: {rec_svm} \nF1 score: {f1_svm} \nArea Under Curve: {areaUnderCurve_svm} \nROC: {roc_svm} \nGini: {gini_svm}")

# Creating Plots for Evalutaion Metrics for each model

acc = [acc_lr, acc_knn, acc_dt, acc_rf, acc_svm]
precision = [precision_lr, precision_knn,
             precision_dt, precision_rf, precision_svm]
recall = [rec_lr, rec_knn, rec_dt, rec_rf, rec_svm]
f1_score = [f1_lr, f1_knn, f1_dt, f1_rf, f1_svm]
auc = [areaUnderCurve_lr, areaUnderCurve_knn,
       areaUnderCurve_dt, areaUnderCurve_rf, areaUnderCurve_svm]
x_pos = np.arange(len(models))

fig, axs = plt.subplots(2, 3)
fig.set_size_inches(25, 15)
axs[0, 0].bar(models, acc)
axs[0, 0].set_title('Accuracy')
axs[0, 1].bar(models, precision, color='orange')
axs[0, 1].set_title('Precision')
axs[0, 2].bar(models, f1_score, color='green')
axs[0, 2].set_title('F1 Score')
axs[1, 0].bar(models, recall, color='red')
axs[1, 0].set_title('Recall')
axs[1, 1].bar(models, auc, color='blue')
axs[1, 1].set_title('Area Under Curve')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.xticks(x_pos, models)
    ax.label_outer()

# Saves our decision Tree for future reference in our Website

joblib.dump(dt, 'thyroid.sav')
