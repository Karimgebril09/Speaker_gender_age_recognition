from data_processor import DataProcessor
from predictors import Predictors

def infer(data_dir):
    ## init
    print("importing processor")
    data_processor = DataProcessor(data_dir)
    print("loading_model")
    predictors=Predictors("../Models/svm_model.pkl","../Models/scaler.pkl","../Models/pca_reducer.pkl")
    ## load data and extract features
    print("loading data")
    features = data_processor.load_all_data()
    print("finished feature extraction")
    ## predict
    result=predictors.infer(features)
    print("finished prediction")
    ## save result
    with open("result.txt", "w") as f:
        for i in range(len(result)):
            f.write(f"{result[i]}\n")


if __name__ == "__main__":
    import sys
    print("hi from infer.py")
    data_dir = sys.argv[1]
    infer(data_dir)


