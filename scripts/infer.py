from data_processor import DataProcessor
from predictors import Predictors

def infer(data_dir):
    ## init
    print("loading")
    data_processor = DataProcessor(data_dir)
    predictors=Predictors("../Models/age_model.joblib","../Models/gender_model.joblib")
    ## load data and extract features
    # age_features,gender_features = data_processor.load_all_data()
    print("finished feature extraction")
    ## predict
    # result=predictors.infer(gender_features,age_features)
    print("finished prediction")
    result=[1,2,3,4,5,6,7,8,9,10] ## for test
    ## save result
    with open("result.txt", "w") as f:
        for i in range(len(result)):
            f.write(f"{result[i]}\n")


if __name__ == "__main__":
    import sys
    print("hi from infer.py")
    data_dir = sys.argv[1]
    infer(data_dir)


