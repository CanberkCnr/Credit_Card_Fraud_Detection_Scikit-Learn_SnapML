import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score
import time

#Kaggle download
import opendatasets as od
od.download("https://www.kaggle.com/mlg-ulb/creditcardfraud")

#Read and print DATA
data = pd.read_csv('creditcardfraud/creditcard.csv')
print("There are " + str(len(data)) + " observations in the credit card fraud dataset.")
print("There are " + str(len(data.columns)) + " variables in the dataset.")

data.head()

#Inflate the original dataset
n_replicas = 10

# inflate the original dataset
big_raw_data = pd.DataFrame(np.repeat(data.values, n_replicas, axis=0), columns=data.columns)

print("There are " + str(len(big_raw_data)) + " observations in the inflated credit card fraud dataset.")
print("There are " + str(len(big_raw_data.columns)) + " variables in the dataset.")

big_raw_data.head()
#labels
labels = big_raw_data.Class.unique()
#sizes
sizes = big_raw_data.Class.value_counts().values

#plot sizes and labels
fig, ax = plt.subplots()
ax.pie(sizes,labels = labels, autopct= "%1.3f%%")
ax.set_title("Target Variable Value Counts")
plt.show()

print("----------------------------------------------")
plt.hist(big_raw_data.Amount.values, 6, histtype='bar', facecolor='g')
plt.show()

print("Minimum amount value is ", np.min(big_raw_data.Amount.values))
print("Maximum amount value is ", np.max(big_raw_data.Amount.values))
print("90% of the transactions have an amount less or equal than ", np.percentile(data.Amount.values, 90))

#Dataset Pre-Processing
big_raw_data.iloc[:,1:30] = StandardScaler().fit_transform(big_raw_data.iloc[:,1:30])
data_matrix = big_raw_data.values

X = data_matrix[:,1:30]
Y = data_matrix[:,30]

X = normalize(X,norm = "l1")
print('X shape = ', X.shape, "Y shape = ",Y.shape)

#Train-Test Split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.3, random_state = 42, stratify = Y)
print('X_train.shape=', X_train.shape, 'Y_train.shape=', Y_train.shape)
print('X_test.shape=', X_test.shape, 'Y_test.shape=', Y_test.shape)

#Decision tree Classifier Model with Scikit-Learn
w_train = compute_sample_weight('balanced', Y_train)

from sklearn.tree import DecisionTreeClassifier
sklearn_dt = DecisionTreeClassifier(max_depth=4, random_state=35)

#Train Decision Tree
t0 = time.time()
sklearn_dt.fit(X_train, Y_train, sample_weight=w_train)
sklearn_time = time.time()-t0
print("[Scikit-Learn] Training time (s):  {0:.5f}".format(sklearn_time))

#Decision Tree Model with Snap ML
from snapml import DecisionTreeClassifier
snapml_dt = DecisionTreeClassifier(max_depth=4, random_state=45, n_jobs=4)

t0 = time.time()
snapml_dt.fit(X_train, Y_train, sample_weight=w_train)
snapml_time = time.time()-t0
print("[Snap ML] Training time (s):  {0:.5f}".format(snapml_time))

#Evaluate Decision Tree Models
training_speedup = sklearn_time/snapml_time
print('[Decision Tree Classifier] Snap ML vs. Scikit-Learn speedup : {0:.2f}x '.format(training_speedup))

sklearn_pred = sklearn_dt.predict_proba(X_test)[:,1]

sklearn_roc_auc = roc_auc_score(Y_test, sklearn_pred)
print('[Scikit-Learn] ROC-AUC score : {0:.3f}'.format(sklearn_roc_auc))

snapml_pred = snapml_dt.predict_proba(X_test)[:,1]

snapml_roc_auc = roc_auc_score(Y_test, snapml_pred)   
print('[Snap ML] ROC-AUC score : {0:.3f}'.format(snapml_roc_auc))


#Support Vector Machine Model with Scikit-Learn
from sklearn.svm import LinearSVC
sklearn_svm = LinearSVC(class_weight='balanced', random_state=31, loss="hinge", fit_intercept=False)

t0 = time.time()
sklearn_svm.fit(X_train, Y_train)
sklearn_time = time.time() - t0
print("[Scikit-Learn] Training time (s):  {0:.2f}".format(sklearn_time))

#Support Vector Machine Model with Snap ML
from snapml import SupportVectorMachine
snapml_svm = SupportVectorMachine(class_weight='balanced', random_state=25, n_jobs=4, fit_intercept=False)

t0 = time.time()
model = snapml_svm.fit(X_train, Y_train)
snapml_time = time.time() - t0
print("[Snap ML] Training time (s):  {0:.2f}".format(snapml_time))

#Evaluate SVM Models
training_speedup = sklearn_time/snapml_time
print('[Support Vector Machine] Snap ML vs. Scikit-Learn training speedup : {0:.2f}x '.format(training_speedup))

sklearn_pred = sklearn_svm.decision_function(X_test)

acc_sklearn  = roc_auc_score(Y_test, sklearn_pred)
print("[Scikit-Learn] ROC-AUC score:   {0:.3f}".format(acc_sklearn))

snapml_pred = snapml_svm.decision_function(X_test)

acc_snapml  = roc_auc_score(Y_test, snapml_pred)
print("[Snap ML] ROC-AUC score:   {0:.3f}".format(acc_snapml))

