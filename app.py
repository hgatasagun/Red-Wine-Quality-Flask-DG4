from flask import Flask, request, render_template
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.neighbors import LocalOutlierFactor


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


def preprocess_input(dataframe):
    dataframe.columns = [col.lower() for col in dataframe.columns]

    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe, cat_th=5, car_th=20)

    for col in num_cols:
        print(col, check_outlier(dataframe, col))

    clf = LocalOutlierFactor(n_neighbors=20)
    clf.fit_predict(dataframe)

    df_scores = clf.negative_outlier_factor_
    th = np.sort(df_scores)[4]

    rows_to_drop = dataframe[df_scores < th].index
    dataframe.drop(axis=0, index=rows_to_drop, inplace=True)

    dataframe["quality"] = np.where(dataframe["quality"] > 5, 1, 0)

    scaler = StandardScaler()
    dataframe[num_cols] = scaler.fit_transform(dataframe[num_cols])

    y = dataframe["quality"]
    X = dataframe.drop(["quality"], axis=1)

    return X, y


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    fixed_acidity = float(request.form['fixed_acidity'])
    volatile_acidity = float(request.form['volatile_acidity'])
    citric_acid = float(request.form['citric_acid'])
    residual_sugar = float(request.form['residual_sugar'])
    chlorides = float(request.form['chlorides'])
    free_sulfur_dioxide = float(request.form['free_sulfur_dioxide'])
    total_sulfur_dioxide = float(request.form['total_sulfur_dioxide'])
    density = float(request.form['density'])
    pH = float(request.form['ph'])
    sulphates = float(request.form['sulphates'])
    alcohol = float(request.form['alcohol'])

    input_features = [fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                      free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]

    prediction = model.predict([input_features])

    result = "Bad" if prediction[0] == 0 else "Good"

    prediction_text = f'Predicted Red Wine Quality: {result}'

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
