import os
import pickle
from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__, template_folder='templates')
MODEL_PATH = 'insurance_model.pkl'

# Features expected: age, sex, bmi, children, smoker, region
def preprocess_input(data):
    # Simple encoding for demonstration
    sex = 1 if data['sex'].lower() == 'male' else 0
    smoker = 1 if data['smoker'].lower() == 'yes' else 0
    region_map = {'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3}
    region = region_map.get(data['region'].lower(), 0)
    return [[data['age'], sex, data['bmi'], data['children'], smoker, region]]

# Train a simple model if not present
def train_dummy_model():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'insurance_new.csv'))
    df['sex'] = df['sex'].map({'male': 1, 'female': 0})
    df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
    df['region'] = df['region'].map({'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3})
    X = df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
    y = df['charges']
    model = LinearRegression()
    model.fit(X, y)
    with open(MODEL_PATH, 'wb'):
        pickle.dump(model, open(MODEL_PATH, 'wb'))
    return model

# Load or train model
if os.path.exists(MODEL_PATH):
    model = pickle.load(open(MODEL_PATH, 'rb'))
else:
    model = train_dummy_model()


# Home page with form
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# Form submission endpoint
@app.route('/predict_form', methods=['POST'])
def predict_form():
    data = {
        'age': int(request.form['age']),
        'sex': request.form['sex'],
        'bmi': float(request.form['bmi']),
        'children': int(request.form['children']),
        'smoker': request.form['smoker'],
        'region': request.form['region']
    }
    features = preprocess_input(data)
    prediction = model.predict(features)[0]
    return render_template('index.html', result=round(prediction, 2))

# JSON API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = preprocess_input(data)
    prediction = model.predict(features)[0]
    return jsonify({'predicted_charge': prediction})

if __name__ == '__main__':
    app.run(debug=True)
