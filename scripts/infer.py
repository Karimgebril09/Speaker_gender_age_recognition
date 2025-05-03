from data_processor import DataProcessor
from predictors import Predictors

def infer(data_dir):
    print("initializing predictors and data_processor")
    data_processor = DataProcessor(data_dir)
    predictors=Predictors("Models/stacking_model.pkl","Models/scaler.pkl","Models/reducer.pkl")
    

    print("loading data")
    features = data_processor.load_all_data()

    print("predicting")
    result=predictors.infer(features)
    
    with open("scripts/result.txt", "w") as f:
        for i in range(len(result)):
            f.write(f"{result[i]}\n")


if __name__ == "__main__":
    import sys
    print("hi from infer.py")
    data_dir = sys.argv[1]
    infer(data_dir)


