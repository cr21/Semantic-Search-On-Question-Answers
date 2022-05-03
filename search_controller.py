# import json
import time
import sys
from elasticsearch import Elasticsearch
# from elasticsearch.helpers import bulk
from flask import Flask
import tensorflow as tf
import tensorflow_hub as hub


def connectToES(host="localhost", port=9200):
    """
    :param host: Host Address where ElasticSearch is installed
    :type host:  String
    :param port: Port Address for Elastic Search
    :type port: int
    :return: Elastic Search instance
    :rtype: Elasticsearch
    """

    es_instance = Elasticsearch([{"host": host, "port": port}])

    print("[TESTING] Connection to ElasticSearch Server")
    if es_instance.ping():
        print("[INFO] Connected to Elastic Search!")
        return es_instance
    else:
        print("[ERROR] Could not connect to Elastic Search")
        sys.exit(500)


def lexicalSearch(es_instance, q):
    """

    :param es_instance: Elastic Search instance
    :type es_instance: Elasticsearch
    :param q: Query Question
    :type q: String
    :return: All the question matches with query questions
    :rtype: List[Questions] Returned by es_instance.search method
    """

    # Searching based on Question title
    search_criteria = {
        'query': {
            "match": {
                'title': q
            }
        }
    }

    res = es_instance.search(index='questions-index', body=search_criteria)
    return res


def semantic_search_by_vector_similarity(es_instance, q, tfmodel):
    """


    :param es_instance: Elastic Search instance
    :type es_instance: Elasticsearch
    :param q: Query Question
    :type q: String
    :param tfmodel: Tensorflow pretrained Model
    :type tfmodel: tf.model
    :return: All the question matches semantically based on Vector representation
    :rtype: List<Questions> returned by es_instance.search method
    """

    query_vector = tf.make_ndarray(tf.make_tensor_proto(tfmodel([q]))).tolist()[0]
    b = {
        "query": {
            "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'title_vector') + 1.0",
                    "params": {"query_vector": query_vector}
                }
            }
        }
    }

    # print(json.dumps(b,indent=4))
    res = es_instance.search(index='questions-index', body=b)

    return res


app = Flask(__name__)
es = connectToES("localhost", 9200)
# model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
model = hub.load("./data/USE4/")


@app.route("/search/<query>")
def search(query):
    # we can not use space in search stirng when we run from browser, + is placeholder for " "
    q = query.replace("+", " ")

    lexical_result = lexicalSearch(es, q)
    semantic_result = semantic_search_by_vector_similarity(es, q, model)

    out = ""
    for hit in lexical_result['hits']['hits']:
        out += (" KeyWord [Lexical] Search : " + str(hit['_score']) + "\t" + hit["_source"]['title'] + "\n")
    out += "*" * 50
    out += "\n"
    for hit in semantic_result['hits']['hits']:
        out += (" Semantic Search : " + str(hit['_score']) + "\t" + hit["_source"]['title'] + "\n")

    return out


if __name__ == "__main__":
    es = connectToES("localhost", 9200)
    # model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    model = hub.load("./data/USE4/")

    while 100:
        input_query = input("Enter Query Question : ")
        start = time.time()

        if input_query == "END":
            break
        print("Query : ".format(input_query))
        lexicalSearch(es, input_query)
        semantic_search_by_vector_similarity(es, input_query, model)

        end = time.time()
        print("Total Time taken {} ".format(end - start))
