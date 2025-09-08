import os
import sys
import pickle

from src.exception import CustomException
from sklearn.metrics import r2_score

def save_object(file_path, obj):

    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for i in range(len(models)):
            model = list(models.values())[i]
            model.fit(X_train, y_train)  # Train the model
            y_train_pred = model.predict(X_train)  # Predict on train data
            y_test_pred = model.predict(X_test)  # Predict on test data

            train_model_score = r2_score(y_train, y_train_pred)  # Calculate r2 score for train data
            test_model_score = r2_score(y_test, y_test_pred)  # Calculate r2 score for test data

            report[list(models.keys())[i]] = test_model_score  

        return report
    
    except Exception as e:
        raise CustomException(e, sys)
