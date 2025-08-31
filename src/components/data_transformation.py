from sklearn.impute import SimpleImputer #handle missing values
from sklearn.preprocessing import StandardScaler #handling feature scaling
from sklearn.preprocessing import OrdinalEncoder # ordinal Encoding
##piplines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import sys,os
from dataclasses import dataclass

import numpy as np
import pandas as pd


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

#data transgormation config
@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')
    








##data ingestionconfig class

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationconfig()
        
    def get_data_transformation_object(self):
        try:
                logging.info("data tranformation initiated")
                #define which coulmns should be ordinal and which should be scaled 
                cat_cols = ['cut', 'color','clarity']
                num_cols = ['carat', 'depth','table', 'x', 'y', 'z']
                
                # define the custom ranking for each ordinla varibale
                cut_cat=['Fair','Good','Very Good','Premium','Ideal']
                color_cat=['D','E','F','G','H','I','J']
                clarity_cat=['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
                
                logging.info("pipline initiated")
                ## numerical pipeline
                num_pipeline=Pipeline(
                    steps=[
                        ('imputer',SimpleImputer(strategy='median')),
                        ('scaler',StandardScaler())
                    ]
                )


                # categorical pipeline
                cat_pipeline=Pipeline(
                    steps=[
                        ('imputer',SimpleImputer(strategy='most_frequent')),
                        ('ordinalencoder',OrdinalEncoder(categories=[cut_cat, color_cat, clarity_cat])),
                        ('scaler', StandardScaler())
                    ]
                )


                preprocessor=ColumnTransformer([
                    ('num_pipeline',num_pipeline,num_cols),
                    ('cat_pipeline',cat_pipeline,cat_cols)
                ]
                )
                
                return preprocessor
         
                logging.info("pipeline completed")
             
             
        except Exception as e:
            logging.info("error in data transformation")
            raise CustomException(e,sys)
    
                    
         
         
    def initiate_data_transformation(self,train_path,test_path):
        try:
            #reading train and test data 
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("read train and tets data completed")
            logging.info(f'train Dataframe head : \n{train_df.head().to_string()}')
            logging.info(f'test Dataframe head : \n{test_df.head().to_string()}')
            
            logging.info("obtaining preprocessing object")
            
            
            preprocessing_obj=self.get_data_transformation_object()
            
            target_column_name="price"
            drop_columns=[target_column_name,'id']
            
            ## features into independent and dependent features
            
            input_feature_train_df=train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            
            input_feature_test_df=test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            
            ##apply the transformation
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            logging.info("applying preprocessing obk=ject on training and testing datasets")
            
            
            train_arr=np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr, np.array(target_feature_test_df)] 
            
            save_object(
                
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            
            logging.info('preprocessor pickle is created and save')
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
                
            )
        
        except Exception as e:
            logging.info("exception ocuured in the initiate_datatransformation")
            
            raise CustomException(e,sys)