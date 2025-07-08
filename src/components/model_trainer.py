import os
import sys
from dataclasses import dataclass

from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                'RandomForestRegressor': RandomForestRegressor(),
                'GradientBoostingRegressor': GradientBoostingRegressor(),
                'AdaBoostRegressor': AdaBoostRegressor(),
                'LinearRegression': LinearRegression(),
                'KNeighborsRegressor': KNeighborsRegressor(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'CatBoostRegressor': CatBoostRegressor(verbose=0),
                'XGBRegressor': XGBRegressor(eval_metric='rmse')
            }
            
            #match the above models with their respective hyperparameters
            params = {
                'RandomForestRegressor': {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5]
                },
                'GradientBoostingRegressor': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5]
                },
                'AdaBoostRegressor': {
                    'n_estimators': [50, 100],
                    'learning_rate': [1.0, 0.1]
                },
                'LinearRegression': {},
                'KNeighborsRegressor': {
                    'n_neighbors': [3, 5, 7]
                },
                'DecisionTreeRegressor': {
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5]
                },
                'CatBoostRegressor': {
                    'iterations': [100, 200],
                    'depth': [6, 8]
                },
                'XGBRegressor': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1]
                }
            }   

            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)  

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException(
                    f"No best model found with sufficient score, best score: {best_model_score}",
                    sys
                )
            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")

            logging.info("Saving the best model")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info("Best model saved successfully")

            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test, predicted)
            return r2

        except Exception as e:
            raise CustomException(e, sys) from e