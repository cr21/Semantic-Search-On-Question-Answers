# import json
# import time
import sys
from elasticsearch import Elasticsearch
# from elasticsearch.helpers import bulk
import csv
import tensorflow as tf
import tensorflow_hub as hub


#Connect to Elastic Search instance
es = Elasticsearch([{"host": "localhost", "port": 9200}])


print("[TESTING] Connection to ElasticSearch Server")
if es.ping():
    print("[INFO] Connected to Elastic Search!")
else:
    print("[ERROR] Could not connect to Elastic Search")
    sys.exit(500)

print("+" * 100)

# Create an index for Questions in Elastic Search

# Define Structure of Elastic Search index

# Mapping " Structure of index"
# Property/Field Name and Type

body = {
    "mappings": {
        "properties": {
            "title": {
                "type": "text"
            },
            "title_vector": {
                "type": "dense_vector",
                "dims": 512
            }
        }
    }
}

# 400 if IndexAlreadyExistsException ignore it
ret_index = es.indices.create(index='questions-index', ignore=400, body=body)

# To check it go to browser and run localhost:9200/questions-index

print("[INFO] GENERATE EMBEDDINGS AND CREATE INDEX FOR EVERY QUESTIONS")

# universal sentence encoder model
use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# we have more than millon question It will consume lots of time and memory so we will limited
# our search to 200000 questions

NUM_QUESTION_INDEXED = 200000

# Questions.csv HEADER : Id,OwnerUserId,CreationDate,ClosedDate,Score,Title,Body
count = 0

with open('./data/Questions.csv', encoding='latin1') as ques_csv:
    reader = csv.reader(ques_csv, delimiter=',')
    next(reader, None)  # Skip header
    for row in reader:
        # Id Field
        doc_id = row[0]
        # Title field
        title = row[5]
        vector = tf.make_ndarray(tf.make_tensor_proto(use_model([title]))).tolist()[0]
        # Body should follow structure define for index
        body = {
            "title": title,
            "title_vector": vector
        }

        # create index for question
        response = es.index('questions-index', id=doc_id, body=body)
        count += 1
        if count % 100 == 0:
            print("[INFO] {} number of questions processed".format( count))
        if count == NUM_QUESTION_INDEXED:
            break
    print("[INFO] {} number of question indexed ".format(NUM_QUESTION_INDEXED))
    print("___" * 25)
