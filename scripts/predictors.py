from joblib import load 
import numpy as np

class Predictors:
    def __init__(self, model_dir,scaler_dir,reducer_dir):
        self.model = load(model_dir )
        self.scaler = load(scaler_dir)
        self.reducer = load(reducer_dir)

    def predict(self, data):
        return np.array(self.model.predict(data), dtype=np.int8)
    
    def infer(self, data):
        print("Predicting...")
        return np.array([1,2,3,4,5,6,7,8,9,10], dtype=np.int8) # for test
        # Scale the data
        scaled_data = self.scaler.transform(data)
        # Reduce the data
        reduced_data = self.reducer.transform(scaled_data)
        # Predict 
        results = self.predict(reduced_data)
        return results

