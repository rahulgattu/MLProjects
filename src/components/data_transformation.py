import os
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
# from src.components.data_ingestion import DataIngestion
from src.utils import save_object



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
    def get_data_transformer_object(self):
        """This function is responsible for the data transformer based on the different types of data

        Raises:
            CustomException: Exception following the CustomException class

        Returns:
            _type_: _description_
        """
        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]
            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),
                    ("scalar",StandardScaler())
                ]
            )
            
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scalar",StandardScaler(with_mean=False))
                ]
            )
            
            logging.info("Categorical Columns: {}".format(categorical_columns))
            logging.info("Numerical Columns: {}".format(numerical_columns))
            
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )
            
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path, encoding='utf8', encoding_errors='ignore')
            test_df = pd.read_csv(test_path, encoding='utf8', encoding_errors='ignore')
            
            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")
            
            preprocessing_object=self.get_data_transformer_object()
            
            target_column_name="math_score"
            numerical_column=['writing_score','reading_score']
            
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df= train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            
            input_feature_train_arr = preprocessing_object.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_object.transform(input_feature_test_df)
            
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info(f"Saved Preprocessing obj.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_object  
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            ) 
        except Exception as e:
            raise CustomException(e, sys)
            
        
    

# class DataTransformation:
#     def __init__(self):
#         self.data_ingestion = DataIngestion()
#         self.train_data_path, self.test_data_path = self.data_ingestion.initiate_data_ingestion()
#         logging.info("Train data path and test data path is captured")

        
#     def data_loading(self):
#         logging.info("Data Loading started")
#         logging.info("Training data loaded")
#         logging.info("Test Data loaded")
        
#         print(train_df.head())
        
        
# if __name__ == '__main__':
    # obj = DataTransformation()
    # obj.data_loading()
    # pass
    
        