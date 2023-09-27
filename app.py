from flask import Flask, request, render_template
import pickle

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
