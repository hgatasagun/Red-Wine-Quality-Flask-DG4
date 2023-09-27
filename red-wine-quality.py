##################################################################
# Red Wine Quality Prediction Model
#################################################################

# Importing libraries
#######################################################
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
import pickle

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

import warnings
warnings.simplefilter("ignore")

################################################
# EXPLORATORY DATA ANALYSIS (EDA)
################################################

df = pd.read_csv('Data Glacier/winequality-red.csv')
df.head()
df.columns = [col.lower() for col in df.columns]

# Check DataFrame Information
#######################################################
def check_df(dataframe, head=5):
    print('################# Columns ################# ')
    print(dataframe.columns)
    print('################# Types  ################# ')
    print(dataframe.dtypes)
    print('##################  Head ################# ')
    print(dataframe.head(head))
    print('#################  Shape ################# ')
    print(dataframe.shape)
    print('#################  NA ################# ')
    print(dataframe.isnull().sum())
    print('#################  Quantiles ################# ')
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99]).T)


check_df(df)


# Capture Numerical and Categorical Variables
#######################################################
def grab_col_names(dataframe, cat_th=10, car_th=20):

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# Explore the Distribution of Numerical and Categorical Variables
##################################################################

# Categorical variables:
########################
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

# Dropping Highly Imbalanced Categorical Variables
##################################################
for col in cat_cols:
    if df[col].dtypes == 'bool':
        df[col] = df[col].astype(int)
        cat_summary(df, col, plot=True)

    else:
        cat_summary(df, col, plot=True)


# Numerical variables:
######################
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot = True)


# Analyze Numerical Variables in Relation to the Target Variable
###################################################################
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: 'mean'}), end='\n\n\n')

for col in num_cols:
    target_summary_with_num(df, 'quality', col)


################################################
# FEATURE ENGINEERING
################################################

# Analyze Outliers
#######################################################
df.plot(kind='box', subplots=True, layout=(20, 5), sharex=False, sharey=False, figsize=(20, 40))

# Define a function to check for outliers in a column
#######################################################
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df, col))


# Define a function to identify and display outliers
#######################################################
def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    outlier_df = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))]

    if outlier_df.shape[0] > 10:
        print(outlier_df.head())
    else:
        print(outlier_df)

    if index:
        outlier_index = outlier_df.index
        return outlier_index

    return outlier_df.shape[0]

for col in num_cols:
    print(col, grab_outliers(df, col))


# Local Outlier Factor Method
#######################################################
clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)

df_scores = clf.negative_outlier_factor_
df_scores[0:5]
np.sort(df_scores)[0:5] # en kotu 5 gozlem

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-')
plt.show()

th = np.sort(df_scores)[4]

df[df_scores < th].shape

df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T

rows_to_drop = df[df_scores < th].index
df.drop(axis=0, index=rows_to_drop, inplace=True)
df.shape


# Analyze Missing Values
#######################################################
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)


# Correlation Analysis
#######################################################
def high_correlated_cols(dataframe, plot = False, corr_th = 0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool_))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > 0.90)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap='RdBu')
        plt.show(block=True)
    return drop_list

high_correlated_cols(df, plot = True)


# Rare Analyser
#######################################################
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "quality", cat_cols)


# Merging Groups
#######################################################
df["quality"] = np.where(df["quality"] > 5, 1, 0)
df["quality"].head()

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# Standardization
#######################################################
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
df.head()


#######################################
# MODELLING
#######################################

# Logistic regression
#################################
y = df["quality"]
X = df.drop(["quality"], axis=1)

model = LogisticRegression(max_iter=1000, random_state=17).fit(X, y)

cv_results = cross_validate(model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

y_preds = cross_val_predict(model, X, y, cv=5)

report_log = classification_report(y, y_preds, target_names=['Class 0', 'Class 1'])
print(report_log)
# Accuracy - 0.74
# F1 - 0.75

print("Mean: %0.3f and Standard deviation: %0.3f"
      % (cv_results['test_roc_auc'].mean(), cv_results['test_roc_auc'].std()))
# Mean: 0.811 and Standard deviation: 0.032


# Save the model
################################
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)


# Load the model
################################
with open('model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)






