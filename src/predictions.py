#!/usr/bin/env python3
import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from config import *
import aws_utils as au

def fetch_pickle(bucket_name, folder, file_name):
    print(f'Loading {file_name} from local')    
    with open(os.path.join(folder, file_name), 'rb') as f:
        fetched_object = pickle.load(f)
    return fetched_object    

def get_model_and_encoders():
    print('Loading models')
    normalizer = fetch_pickle(BUCKET_NAME, FOLDER, 'normalizer.pkl')
    encoder = fetch_pickle(BUCKET_NAME, FOLDER, 'encoder.pkl')
    model = fetch_pickle(BUCKET_NAME, FOLDER, 'model.pkl')
    return normalizer, encoder, model

def predict(sample: list) -> dict:
    """
        'sample': List of four floats
    """
    normalizer, encoder, model = get_model_and_encoders()
    test_data = normalizer.transform(np.array(sample).reshape(1, -1))
    y_probabilities = model.predict(test_data)
    print(f'y_probabilities: {y_probabilities}')
    y_class = y_probabilities.argmax(axis=-1)
    print(f'y_class: {y_class}')
    y_class_decoded = encoder.inverse_transform(y_class[0])
    print(f'y_class_decoded: {y_class_decoded}')

    predicted_class = {
        'sample': sample,
        'class': y_class_decoded,
        'confidence': round(y_probabilities.flatten()[y_class[0]] * 100, 2)
    }

    return predicted_class