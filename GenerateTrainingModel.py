import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
import warnings
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree  import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import BernoulliNB
from lightgbm import LGBMClassifier
from sklearn.feature_selection import RFE
import itertools
from xgboost import XGBClassifier
from tabulate import tabulate
import pickle
import optuna
from sklearn.metrics import confusion_matrix, classification_report, f1_score

# Read the training data
train = pd.read_csv('Train_data.csv')
test = pd.read_csv('Test_data.csv')

# Display summary statistics for categorical variables
train.describe(include='object')

# Check the shape of the training data
print("Shape of training data:", train.shape)

# Check for missing values in the training data
print("Missing values in training data:")
print(train.isnull().sum())

# Calculate the percentage of missing values for each column
total = train.shape[0]
missing_columns = [col for col in train.columns if train[col].isnull().sum() > 0]
for col in missing_columns:
    null_count = train[col].isnull().sum()
    per = (null_count/total) * 100
    print(f"{col}: {null_count} ({round(per, 3)}%)")

# Check for duplicate rows in the training data
print(f"Number of duplicate rows: {train.duplicated().sum()}")

# Plot the distribution of the target variable 'class'
sns.countplot(x=train['class'])
plt.show()  # Display the plot

print('Class distribution Training set:')
print(train['class'].value_counts())

def le(df):
    for col in df.columns:
        if df[col].dtype == 'object':
                label_encoder = LabelEncoder()
                df[col] = label_encoder.fit_transform(df[col])

le(train)
le(test)

train.drop(['num_outbound_cmds'], axis=1, inplace=True)
test.drop(['num_outbound_cmds'], axis=1, inplace=True)

print(train.head())

X_train = train.drop(['class'], axis=1)
Y_train = train['class']

rfc = RandomForestClassifier()

rfe = RFE(rfc, n_features_to_select=10)
rfe = rfe.fit(X_train, Y_train)

feature_map = [(i, v) for i, v in itertools.zip_longest(rfe.get_support(), X_train.columns)]
selected_features = [v for i, v in feature_map if i==True]

print(selected_features)

X_train = X_train[selected_features]
scale = StandardScaler()
X_train = scale.fit_transform(X_train)
test = scale.fit_transform(test)

x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, train_size=0.70, random_state=2)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

import time

# Logistic Regression Model
clfl = LogisticRegression(max_iter=1200000)
start_time = time.time()
clfl.fit(x_train, y_train.values.ravel())
end_time = time.time()
print("Training time (Logistic Regression): ", end_time-start_time)

# Save Logistic Regression model as a .pkl file
with open('LogisticRegression.pkl', 'wb') as f:
    pickle.dump(clfl, f)

start_time = time.time()
y_test_pred = clfl.predict(x_train)
end_time = time.time()
print("Testing time (Logistic Regression): ", end_time-start_time)

lg_model = LogisticRegression(random_state=42)
lg_model.fit(x_train, y_train)

lg_train, lg_test = lg_model.score(x_train, y_train), lg_model.score(x_test, y_test)

print(f"Training Score (Logistic Regression): {lg_train}")
print(f"Test Score (Logistic Regression): {lg_test}")

def objective(trial):
    n_neighbors = trial.suggest_int('KNN_n_neighbors', 2, 16, log=False)
    classifier_obj = KNeighborsClassifier(n_neighbors=n_neighbors)
    classifier_obj.fit(x_train, y_train)
    accuracy = classifier_obj.score(x_test, y_test)
    return accuracy

study_KNN = optuna.create_study(direction='maximize')
study_KNN.optimize(objective, n_trials=1)
print(study_KNN.best_trial)

KNN_model = KNeighborsClassifier(n_neighbors=study_KNN.best_trial.params['KNN_n_neighbors'])
KNN_model.fit(x_train, y_train)

KNN_train, KNN_test = KNN_model.score(x_train, y_train), KNN_model.score(x_test, y_test)

print(f"Train Score (KNN): {KNN_train}")
print(f"Test Score (KNN): {KNN_test}")

# Save KNN model as a .pkl file
with open('KNeighborsClassifier.pkl', 'wb') as f:
    pickle.dump(KNN_model, f)

clfd = DecisionTreeClassifier(criterion="entropy", max_depth=4)
start_time = time.time()
clfd.fit(x_train, y_train.values.ravel())
end_time = time.time()
print("Training time (Decision Tree): ", end_time-start_time)

start_time = time.time()
y_test_pred = clfd.predict(x_train)
end_time = time.time()
print("Testing time (Decision Tree): ", end_time-start_time)

def objective(trial):
    dt_max_depth = trial.suggest_int('dt_max_depth', 2, 32, log=False)
    dt_max_features = trial.suggest_int('dt_max_features', 2, 10, log=False)
    classifier_obj = DecisionTreeClassifier(max_features=dt_max_features, max_depth=dt_max_depth)
    classifier_obj.fit(x_train, y_train)
    accuracy = classifier_obj.score(x_test, y_test)
    return accuracy

study_dt = optuna.create_study(direction='maximize')
study_dt.optimize(objective, n_trials=30)
print(study_dt.best_trial)

dt = DecisionTreeClassifier(max_features=study_dt.best_trial.params['dt_max_features'], max_depth=study_dt.best_trial.params['dt_max_depth'])
dt.fit(x_train, y_train)

dt_train, dt_test = dt.score(x_train, y_train), dt.score(x_test, y_test)

print(f"Train Score (Decision Tree): {dt_train}")
print(f"Test Score (Decision Tree): {dt_test}")

# Save Decision Tree model as a .pkl file
with open('DecisionTreeClassifier.pkl', 'wb') as f:
    pickle.dump(dt, f)

data = [["KNN", KNN_train, KNN_test], 
        ["Logistic Regression", lg_train, lg_test],
        ["Decision Tree", dt_train, dt_test]]

col_names = ["Model", "Train Score", "Test Score"]
print(tabulate(data, headers=col_names, tablefmt="fancy_grid"))

# Model Validation
SEED = 42

# Decision Tree Model
dtc = DecisionTreeClassifier()

# KNN
knn = KNeighborsClassifier()

# LOGISTIC REGRESSION MODEL
lr = LogisticRegression()

models = {}
models['KNeighborsClassifier'] = knn
models['LogisticRegression'] = lr
models['DecisionTreeClassifier'] = dtc

scores = {}
for name in models:
    scores[name] = {}
    for scorer in ['precision', 'recall']:
        scores[name][scorer] = cross_val_score(models[name], x_train, y_train, cv=10, scoring=scorer)

def line(name):
    return '*'*(25-len(name)//2)

for name in models:
    print(line(name), name, 'Model Validation', line(name))

    for scorer in ['precision', 'recall']:
        mean = round(np.mean(scores[name][scorer])*100, 2)
        stdev = round(np.std(scores[name][scorer])*100, 2)
        print("Mean {}:".format(scorer), "\n", mean, "%", "+-", stdev)
        print()

for name in models:
    for scorer in ['precision', 'recall']:
        scores[name][scorer] = scores[name][scorer].mean()
scores = pd.DataFrame(scores).swapaxes("index", "columns")*100
scores.plot(kind="bar", ylim=[80, 100], figsize=(24, 6), rot=0)
plt.show()

preds = {}
for name in models:
    models[name].fit(x_train, y_train)
    preds[name] = models[name].predict(x_test)
print("Predictions complete.")

def line(name, sym="*"):
    return sym*(25-len(name)//2)

target_names = ["normal", "anomaly"]
for name in models:
    print(line(name), name, 'Model Testing', line(name))
    print(confusion_matrix(y_test, preds[name]))
    print(line(name, '-'))
    print(classification_report(y_test, preds[name], target_names=target_names))

f1s = {}
for name in models:
    f1s[name] = f1_score(y_test, preds[name])
f1s = pd.DataFrame(f1s.values(), index=f1s.keys(), columns=["F1-score"])*100
f1s.plot(kind="bar", ylim=[80, 100], figsize=(10, 6), rot=0)
plt.show()
