from joblib import dump, load
from flask import Flask
from flask import request
import pandas as pd
import time
import json

app = Flask(__name__)


@app.route('/predict_single/')
def predict_single():
    """Predict an element"""
    before_time = time.time()
    pixels = request.args.get('pixels')
    sample_test = pd.DataFrame([pixels.split(',')])
    y_pred = clf_load.predict(sample_test)
    print('\nPrediction is ' + y_pred[0] + ', found in ' + str(round(time.time() - before_time, 3)) + ' seconds')
    return ('Prediction is ' + y_pred[0] + ', found in ' + str(round(time.time() - before_time, 3)) + ' seconds')


@app.route('/predict/', methods=['POST'])
def predict():
    """Predict an element"""
    json_data = request.get_json(force=True)
    dict_sample = json.loads(json_data)
    df_samples = pd.DataFrame.from_dict(dict_sample, orient="index")
    y_pred = clf_load.predict(df_samples)
    return str(y_pred)


if __name__ == '__main__':
    clf_load = load('model.joblib')
    app.run()
