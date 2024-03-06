import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import copy

# Load the dataset
cols  = ['Age', 'Sex', 'Job', 'Housing', 'Saving accounts', 
        'Checking account',	'Credit amount', 'Duration', 'Purpose',	'Risk']
g_cred = pd.read_csv('german_credit_data.csv', usecols=cols)
g_cred_raw = g_cred.copy()

# Preprocess the data
categ_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose', 'Risk']
le_dict = {}

for col in categ_cols:
    le = LabelEncoder()
    g_cred[col] = le.fit_transform(g_cred[col])
    le_dict[col] = le

g_cred.reset_index(drop=True, inplace=True)
# print(g_cred.head())

# # One hot encoding
# g_cred2 = pd.read_csv('german_credit_data.csv')
# g_cred2['Risk'] = g_cred2['Risk'].map({'good': 1, 'bad': 0})
# g_cred2 = pd.get_dummies(g_cred2, columns=categ_col)
# print(g_cred2.head())

# Utility function
def X_y_split(df):
    X = df.drop('Risk', axis=1)
    y = df['Risk']
    return X, y

# Utility function
def save_to_csv(df, categ_cols, le_dict, filename):
    df_cpy = df.copy()
    for col in categ_cols:
        df_cpy[col] = le_dict[col].inverse_transform(df_cpy[col])
    df_cpy.to_csv(filename)

# Utility function
def build_predicate(df, index, attributes):
    predicate = ''
    n_attr = len(attributes) - 1
    for i, attr in enumerate(attributes):
        attr_val  = df.loc[index, attr]
        predicate += f'`{attr}` == {attr_val}'
        if i < n_attr:
            predicate += ' and '
    return predicate

# Utility function
def all_subsets(S, subset_list, curr=[], maxlen=None):
    if maxlen is None:
        maxlen = len(S)

    if len(S) == 0 or len(curr) == maxlen:
        if len(curr) > 0:
            subset_list.append(copy.copy(curr))
        return

    curr.append(S[0])
    all_subsets(S[1:], subset_list, curr, maxlen)

    curr.pop()
    all_subsets(S[1:], subset_list, curr, maxlen)

# Utility function
def model_reproducibility(base, new):
    diff = np.abs(base - new)
    return 1.0 - np.average(diff)


# Split the data into training and testing sets
train_set, test_set = train_test_split(g_cred, test_size=0.2, random_state=42)

# For manual analysis
save_to_csv(df=train_set, categ_cols=categ_cols, le_dict=le_dict, filename='g_credit_train.csv')
save_to_csv(df=test_set, categ_cols=categ_cols, le_dict=le_dict, filename='g_credit_test.csv')

# Train a Random Forest Classifier
X_train, y_train = X_y_split(train_set)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions and evaluate the model
X_test, y_test = X_y_split(test_set)
y_pred = clf.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))
correct = y_pred == y_test
np.savetxt('res.txt', correct, fmt="%s", delimiter='\n')

# Produce all combinations (upto a maxlen) of attributes for building predicates
attributes = ['Purpose', 'Duration', 'Job', 'Housing', 'Saving accounts', 'Checking account']
combinations = []
all_subsets(S=attributes, subset_list=combinations, maxlen=3)
# all_subsets(S=attributes, subset_list=combinations)
# print(combinations)


# _________________________For single query row_________________________
# Misclaffied
test_index = 740
# test_index = 174
# test_index = 235
# test_index = 687
# test_index = 578
# test_index = 289
# test_index = 307
# test_index = 959
# test_index = 559
# test_index = 583
# test_index = 429
# test_index = 309
# test_index = 595
# test_index = 649
# test_index = 208

# Correct
# test_index = 737
# test_index = 76
# test_index = 54
# test_index = 292
# test_index = 67
# test_index = 985
# test_index = 120

max_size = 10
reproducibility_threshold = 0.9

print('Inspecting:\n', g_cred_raw.loc[[test_index]])
test_index_pred = clf.predict(X_test.loc[[test_index]])

for attributes in combinations:
    predicate = build_predicate(df=X_test, index=test_index, attributes=attributes)
    
    temp_train_set = train_set.query(f'not({predicate})')
                
    X_train, y_train = X_y_split(temp_train_set)
    temp_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    temp_clf.fit(X_train, y_train)

    new_test_index_pred = temp_clf.predict(X_test.loc[[test_index]])
    new_y_pred = temp_clf.predict(X_test)

    reproducibility = model_reproducibility(base=y_pred, new=new_y_pred)

    if new_test_index_pred != test_index_pred and reproducibility > reproducibility_threshold:
        drop_subset = train_set.query(predicate)
        # if True:
        if len(drop_subset) < max_size:
            print('Index:', test_index)
            print("Predicate:", predicate)
            print("Reproducibility:", reproducibility)
            print("Drop subset size:", len(drop_subset))
            print("Drop subset:\n", g_cred_raw.loc[drop_subset.index])






# # _________________________For entire test set_________________________
# # testing_for_correct = True          # Approx 12 mins
# testing_for_correct = False           # Approx 3.5 mins

# if testing_for_correct:
#     log_file = 'log_correct.txt'
#     checklist = 'checklist_correct.txt'
# else:
#     log_file = 'log_missed.txt'
#     checklist = 'checklist_missed.txt'

# file = open(log_file,'w')
# file = open(checklist,'w')

# max_size = 10
# reproducibility_threshold = 0.9

# for i, (index, value) in enumerate(y_test.items()):
#     test_index_pred = clf.predict(X_test.loc[[index]])
#     if correct.iloc[i] == testing_for_correct:    
#         for attributes in combinations:
#             predicate = build_predicate(df=X_test, index=index, attributes=attributes)
            
#             temp_train_set = train_set.query(f'not({predicate})')
                        
#             X_train, y_train = X_y_split(temp_train_set)
#             temp_clf = RandomForestClassifier(n_estimators=100, random_state=42)
#             temp_clf.fit(X_train, y_train)

#             new_test_index_pred = temp_clf.predict(X_test.loc[[index]])
#             new_y_pred = temp_clf.predict(X_test)

#             reproducibility = model_reproducibility(base=y_pred, new=new_y_pred)

#             if new_test_index_pred != test_index_pred and reproducibility > reproducibility_threshold:
#                 print('Index:', index, file=open(log_file,'a+'))
#                 print("Predicate:", predicate, file=open(log_file,'a+'))
#                 print("Reproducibility:", reproducibility, file=open(log_file,'a+'))
                
#                 drop_subset = train_set.query(predicate)
#                 print("Drop subset size:", len(drop_subset), file=open(log_file,'a+'))
#                 print("Drop subset:", drop_subset.index, file=open(log_file,'a+'))

#                 if len(drop_subset) < max_size:
#                     print('Index:', index, file=open(checklist,'a+'))
#                     print("Predicate:", predicate, file=open(checklist,'a+'))
#                     print("Reproducibility:", reproducibility,  file=open(checklist,'a+'))
#                     print("Drop subset size:", len(drop_subset), file=open(checklist,'a+'))
#                     print("Drop subset:", drop_subset.index, file=open(checklist,'a+'))