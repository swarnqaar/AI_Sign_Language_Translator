from AI_signlanguage.exception.exception import SignLanguageException
from AI_signlanguage.logging.logger import logging


## configuration of the Data Ingestion Config

from AI_signlanguage.entity.config_entity import DataIngestionConfig
from AI_signlanguage.entity.artifact_entity import DataIngestionArtifact
import os
import sys
import numpy as np
import pandas as pd
import pymongo
from typing import List
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL=os.getenv("MONGO_DB_URL")


class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config=data_ingestion_config
        except Exception as e:
            raise SignLanguageException(e,sys)
        
    def export_collection_as_dataframe(self) -> pd.DataFrame:
        try:
            # ── Try MongoDB first ──────────────────────────────────────────
            try:
                import certifi
                ca = certifi.where()
                self.mongo_client = pymongo.MongoClient(
                    MONGO_DB_URL,
                    tlsCAFile=ca,
                    serverSelectionTimeoutMS=5000,  # 5 second timeout
                )
                self.mongo_client.server_info()  # test connection

                database_name   = self.data_ingestion_config.database_name
                collection_name = self.data_ingestion_config.collection_name
                collection      = self.mongo_client[database_name][collection_name]

                df = pd.DataFrame(list(collection.find()))
                if "_id" in df.columns:
                    df = df.drop(columns=["_id"])
                df.replace({"na": np.nan}, inplace=True)

                logging.info(f"Loaded {len(df)} rows from MongoDB")
                return df

            except Exception as mongo_error:
                logging.warning(f"MongoDB failed: {mongo_error}")
                logging.info("Falling back to local CSV file...")

            # ── Fallback: load from local CSV ─────────────────────────────
            local_csv_paths = [
                "Sign_Data/sign_language_data.csv",
                os.path.join("Sign_Data", "sign_language_data.csv"),
            ]

            for csv_path in local_csv_paths:
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path, low_memory=False)

                    # Clean dirty values
                    dirty = ["nan","NaN","Nan","Knan","knan","NULL",
                            "null","None","none","NA","na","N/A",""]
                    df.replace(dirty, np.nan, inplace=True)

                    # Force numeric columns
                    for col in df.columns:
                        if col != "label":
                            df[col] = pd.to_numeric(df[col], errors="coerce")

                    logging.info(f"Loaded {len(df)} rows from local CSV: {csv_path}")
                    print(f"Loaded data from local CSV: {csv_path} ({len(df)} rows)")
                    return df

            # ── No data found ─────────────────────────────────────────────
            raise FileNotFoundError(
                "No data found! Neither MongoDB nor local CSV available.\n"
                "Please run: python collect_data.py --gesture A --samples 300"
            )

        except Exception as e:
            raise SignLanguageException(e, sys)
        
    def export_data_into_feature_store(self,dataframe: pd.DataFrame):
        try:
            feature_store_file_path=self.data_ingestion_config.feature_store_file_path
            #creating folder
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            dataframe.to_csv(feature_store_file_path,index=False,header=True)
            return dataframe
            
        except Exception as e:
            raise SignLanguageException(e,sys)
        
    def split_data_as_train_test(self,dataframe: pd.DataFrame):
        try:
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )
            logging.info("Performed train test split on the dataframe")

            logging.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )
            
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            
            os.makedirs(dir_path, exist_ok=True)
            
            logging.info(f"Exporting train and test file path.")
            
            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )

            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )
            logging.info(f"Exported train and test file path.")

            
        except Exception as e:
            raise SignLanguageException(e,sys)
        
        
    def initiate_data_ingestion(self):
        try:
            dataframe=self.export_collection_as_dataframe()
            dataframe=self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)
            dataingestionartifact=DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,
                                                        test_file_path=self.data_ingestion_config.testing_file_path)
            return dataingestionartifact

        except Exception as e:
            raise SignLanguageException(e, sys)