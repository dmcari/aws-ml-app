#!/usr/bin/env python3
import warnings
warnings.filterwarnings('ignore')

import numpy as np

from flask import Flask, request, json, jsonify

import json
import os
import pickle

import predictions as pr

app = Flask(__name__)

@app.route("/")
def hello():
    return jsonify(message=f"ML Classifier {os.environ.get('ENVIRONMENT')}")


@app.route('/classify', methods=['GET'])
def classify():
    """
        Classification in 4 example attributes.
    """
    # Get the url params.
    attribute1 = request.args.get('attribute1')
    attribute2 = request.args.get('attribute2')
    attribute3 = request.args.get('attribute3')
    attribute4 = request.args.get('attribute4')
    sample = [attribute1, attribute2, attribute3, attribute4]

    # Validations
    if None in sample:
        return jsonify(message='Missing some measure'), 404

    for measure in sample:
        try:
            float(measure)
        except ValueError:
            return jsonify(message='Some measure is not a float'), 404

    return jsonify(pr.predict(sample))


if __name__ == '__main__':    
    app.run(host='0.0.0.0', debug=True) 