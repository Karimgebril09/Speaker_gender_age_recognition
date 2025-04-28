from joblib import load 
import numpy as np

class Predictors:
    def __init__(self, age_directory, gender_directory):
        # self.age_model = load(age_directory)
        # self.gender_model = load(gender_directory)

        self.mapper_array = np.array([
            [1, 3],
            [0, 2],
        ])

    def predict_gender(self, data):
        
        return np.array(self.gender_model.predict(data), dtype=np.int8)
    
    def predict_age(self, data):
        return np.array(self.age_model.predict(data), dtype=np.int8)
    
    def infer(self, gender_data, age_data):
        age_results = self.predict_age(age_data)   # 0 for twenties, 1 for fifties
        gender_results = self.predict_gender(gender_data)  # 0 for female, 1 for male

        final_results = self.mapper_array[gender_results, age_results]

        return final_results
