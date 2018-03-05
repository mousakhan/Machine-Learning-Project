import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn.utils import shuffle
from sklearn.metrics import precision_score, recall_score
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from gplearn.genetic import SymbolicRegressor
from sklearn.model_selection import GridSearchCV

random_state = 99

# Read in data
df = pd.read_csv('adult.csv')

# Shuffle all the rows randomly
df = shuffle(df, random_state=random_state)

# Remove education column since education.num already exists as a column
df = df.drop('education', 1)

# Relabel <=50k to 0 for conveniance
df = df.replace('<=50K', 0)

# Relabel >50k to 1 for conveniance
df = df.replace('>50K', 1)

# Replace any '?' missing data with NaN for conveniance
df = df.replace('?', np.NaN)

# Drop all rows with any missing data, which drops almost 2400 rows
df = df.dropna(axis=0, how='any')

# Drop duplicates, which drops 24 rows
df = df.drop_duplicates()

# Convert all categorical data using one-hot encoding (apply label encoding
df = df.apply(preprocessing.LabelEncoder().fit_transform)

# Get data
X = df.loc[:, df.columns != 'income']

# Get labels
y = df.iloc[:,-1].values

# Split the data 70/30
X_tr, X_ts, Y_tr, Y_ts = train_test_split(X, y, 
                                           train_size = 0.7, 
                                           random_state=random_state)


# Use SMOTE to deal with imbalanced data, which creates synthetic data points
sm = SMOTE(random_state=random_state)
X_train, Y_train = sm.fit_sample(X_tr, Y_tr)

# Creating LR model
#model = LogisticRegression(class_weight='balanced')
#model.fit(X_train, Y_train)

# Use k-means 5 folds and calculate accuracy
#cv = np.mean(cross_val_score(model, X_train, Y_train, cv=5))

# Calculate results on testing set
#results = model.predict(X_ts)


param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]
grid = GridSearchCV(SVC(), param_grid)
grid.fit(X_train, Y_train)

print('The parameters combination that would give best accuracy is : ')
print(grid.best_params_)
print('The best accuracy achieved after parameter tuning via grid search is : ', grid.best_score_)


#print("The recall is " + str(recall_score(Y_ts, results, average='macro') ))
#print("The precision is " + str(precision_score(Y_ts, results, average='macro')) )
#print("The cross validation accuracy is " + str(cv * 100.0) + "%")
#print("The accuracy on testing set is " + str((results == Y_ts).sum()/float(Y_ts.size) * 100.0) + "%")

