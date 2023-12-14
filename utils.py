import os
import pymongo
from dotenv import load_dotenv

load_dotenv()

mongo_connection_string = os.environ['DATABASE_URL']
mongo_db_name = os.environ['MONGO_DB']

def get_mongo_client():
    """
    Function to get the mongo client.
    """
    mongo_client = pymongo.MongoClient(mongo_connection_string)
    return mongo_client

def get_mongo_collection(collection_name):
    """
    Function to get the mongo collection.
    """
    mongo_client = get_mongo_client()
    db = mongo_client[mongo_db_name]
    collection = db[collection_name]
    return collection