import time
import joblib
import csv
from datetime import datetime
import pandas as pd
import logging
from typing import Iterable, Callable, Dict

import sklearn.pipeline
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def save_pipeline(pipeline: sklearn.pipeline.Pipeline, ts: datetime = None) -> str:
    model_path = f"data/pipeline-{ts}.pkl".replace(" ","-")
    joblib.dump(pipeline, model_path)
    return model_path


def record_model(pipeline: sklearn.pipeline.Pipeline, score: float, parameters: dict) -> None:
    model = str(pipeline)
    ts = datetime.now()
    model_path = save_pipeline(pipeline, ts)
    f = open('metadata_store.csv', 'a')
    csv_writer = csv.writer(f)
    csv_writer.writerow([ts, model, parameters, score, model_path])
    f.close()
    
def new_feature(feature_dict: Dict) -> Dict:
    return feature_dict["feature_1"] ** 2

class SuperSimpleFeatureStore:
    
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe
        self.feature_dict = user_data.set_index("user_id").T.to_dict()
            
    def register_feature(self, name: str, feature_definition: Callable) -> None:
        for key in self.feature_dict:
            self.feature_dict[key][name] = feature_definition(self.feature_dict[key]) 
            
    def get_user_feature(self, user_id: str) -> Dict:
        return self.feature_dict.get(user_id)
    
    def get_all_data(self) -> pd.DataFrame:
        return pd.DataFrame(self.feature_dict).T
        
if __name__ == "__main__":
    while True:
        user_data = pd.read_csv("data/user_data.csv", index_col=0)

        feature_store = SuperSimpleFeatureStore(user_data)
        feature_store.register_feature("feature_3", new_feature)

        parameters = {
            "penalty":"l2",
            "C":1.0,
            "max_iter": 100
        }

        scaler = StandardScaler()
        logistic_regression = LogisticRegression(
            penalty=parameters["penalty"],
            C=parameters["C"],
            max_iter=parameters["max_iter"],
        )
        pipeline = make_pipeline(scaler, logistic_regression)
        data = feature_store.get_all_data()
        X_train, X_test, y_train, y_test = train_test_split(data[["feature_1","feature_2","feature_3"]], data["target"])

        pipeline.fit(X_train, y_train)
        score = pipeline.score(X_test, y_test)
        record_model(pipeline, score, parameters)
        logging.info("Success")
        time.sleep(60)