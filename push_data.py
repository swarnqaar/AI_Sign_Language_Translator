
import os
import sys
import json

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL=os.getenv("MONGO_DB_URL")
print(MONGO_DB_URL)

import certifi
import pandas as pd
import numpy as np
import pymongo
import ssl
from AI_signlanguage.exception.exception import SignLanguageException
from AI_signlanguage.logging.logger import logging
class SignLanguageDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise SignLanguageException(e,sys)
        
    def csv_to_json_convertor(self,file_path):
        try:
            data=pd.read_csv(file_path)
            data.reset_index(drop=True,inplace=True)
            records=list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise SignLanguageException(e,sys)
    def insert_data_mongodb(self,records,database,collection):
        try:
            self.database=database
            self.collection=collection
            self.records=records

            ca = certifi.where()
            self.mongo_client=pymongo.MongoClient(MONGO_DB_URL, tls=True, tlsCAFile=ca)
            self.database = self.mongo_client[self.database]
            
            self.collection=self.database[self.collection]
            self.collection.insert_many(self.records)
            return(len(self.records))
        except Exception as e:
            raise SignLanguageException(e,sys)
        
if __name__=='__main__':
    FILE_PATH="sign_data/sign_language_data.csv" 
    DATABASE="SHUBHAM_AI"
    Collection="SignLanguageData"
    sign_language_obj=SignLanguageDataExtract()
    records=sign_language_obj.csv_to_json_convertor(file_path=FILE_PATH)
    print(records)
    no_of_records=sign_language_obj.insert_data_mongodb(records,DATABASE,Collection)
    print(no_of_records)


'''



import pymongo
import certifi
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

print("="*50)
print("URL loaded:", MONGO_DB_URL)  # Print FULL URL to verify
print("="*50)

try:
    ca = certifi.where()
    client = pymongo.MongoClient(MONGO_DB_URL, tls=True, tlsCAFile=ca)
    # Force actual connection
    client.admin.command('ping')
    print("✅ Connection SUCCESSFUL!")
except Exception as e:
    print(f"❌ Connection FAILED: {e}")


'''