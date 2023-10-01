from flask import Flask, request, render_template
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

def preprocess_input(input_data):
    input_df = pd.read_csv(input_data)
    input_df.columns = [col.lower() for col in input_df.columns]

    clf = LocalOutlierFactor(n_neighbors=20)
    outliers = clf.fit_predict(input_df)
    input_df = input_df[outliers == 1]

    scaler = StandardScaler()
    num_cols = [col for col in input_df.columns if input_df[col].dtypes != "O"]
    input_df[num_cols] = scaler.fit_transform(input_df[num_cols])

    X = input_df.drop(["quality"], axis=1)
    y = np.where(input_df["quality"] > 5, 1, 0)

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
