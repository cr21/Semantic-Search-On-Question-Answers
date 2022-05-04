# Machine Learning Design :  Searching in Question Answer Dataset

## Problem:
Give a Query Efficiently retrieve Similar question from question answer repository.

This can be solved by building text similarity algorithms. Text Similarity can be usefule in many different
areas.
- **Question Answer Forum**  : Given a collection of question, find questions that are most similar to given queries
- **Image search** : Given text description, retrieve all images matches with text description.



## Objective:
* Given a query text retrieve most similar question efficiently
* Faster response time
* Lower service cost
* better utilization of in house knowledge base.

## First Cut Solution Approach :  Keyword based - Lexical Search
In this technique we would rank the results based on how many words they share with query text.
In keyword based search we represent text as vector which is assign to one dimensions for each word in corpus.

Vector for entire query is based on numer of times each term in vocab appears. This is also known as "Bag Of Words" representation. 
TF-IDF and inverted index based approach also comes under this type of search.
    
                        W1           W2             W3          w4
        Q1 :           How           to            install      pip?
                        D1           D2             D4           D3
                        D2           D4             D3           D5
                        .             .              .             .
                        .             .              .             .
                        .             .              .             .
                        D3           D3             D8........   D13

Each Column have entries of all the documents where Word at column header appears.
D3 Document  contains all the word mentioned in query so it is more similar to query.


### Problem With Keyword based Search:
A lexical search approach  would be to rank documents based on how many words they share with the query.
But Document may contain different words in different orders, yet it can be similar semantically. Other thing bag of words based 
representation will not hold word ordering into account, for natural query understanding and to understand context word ordering is important.


Let say we have the following questions

    Q1. How to install new packages via python?
    Q2. use new library via pip?

Question **Q1 and Q2 are worded differently, but they semantically mean the same** think, how to install new package using some software

    Q3. install elasticsearch
    Q4. setup Lucene and Apache Solr
    Q5. Install fulltext search engine software

Similarly, Q3, Q4, and Q5 are worded differently, but they all related to same kind of software.
**Apache Solr, Elasticsearch and Lucene all are full text search engine.** 

If we had used keyword based search, these questions could not be similar.

### How to Improve, What next?

## Semantic Search : 

We would like to build query representation in way that will capture linguistic content of the query text. We will call it as embedding which
is dense numeric vector representation of given text.

These vector captures semantic meaning of the words, closely related vectors should be closed to gather.
Synonyms words should be in near distance in vector space.

### What's the benefits of Embeddings

* The main problem with "Bag of words" vectors are that, they are very high dimensional vectors (d = length of vocab), and they are sparse.
* Text Embeddings vectors are dense and lower dimensional vector, which contains semantic meaning of information in text.
* The problem with "Bag of words" is that it fails to capture word ordering which is very important in understanding large context.


### How to use Embeddings for Similarity Search

Let's say we had collection of questions and answers. A User ask a questions, and we want to retrieve most similar questions from corpus.

For faster search we need data structure for fast retrieval, indexing is solution for this.
* We will first train Query Embedding models to generate dense lower dimensional vector for queries, or we can use pretrained model already trained on large dataset like wikipedia or common crawl.
* We will then create an index for query embeddings.
* at run time user query is passed through query embedding model to get query vector, then we will compare query vector to all the questions vector in dataset using cosine similarity to get top k results.

### Implementation on StackOverflow Questions answers dataset

[StackSample](https://www.kaggle.com/datasets/stackoverflow/stacksample) is Dataset with the text of 10% of questions and answers from the Stack Overflow programming Q&A website
I have used **200000 questions from this dataset** ( nearly 20% of this dataset.)

I created a script to download dataset, and created embedding model in tensorflow. I used Google's **Universal sentence encoder**
We only used pretrained model, we haven't fine tune this model.

We created following elasticsearch index:
```yaml

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
    }`
```

here "dense_vector" dimension is 512, so we want to make sure that our embedding model will generate 512 dimensional vector.

To index questions, we will pass question through model and generate 512 dimensional vector,and then it is added to "title_vector" field.
```yaml
{
  "script_score": {
    "query": {"match_all": {}},
    "script": {
      "source": "cosineSimilarity(params.query_vector, 'title_vector') + 1.0",
      "params": {"query_vector": query_vector}
    }
  }
}

```

We used cosineSimilarity for similarity search. 

### How to set up project

1. download and set up docker in system
2. Set up Elasticsearch
```yaml
   # Download Elasticsearch v 7.7.0 image from docker
   docker pull docker.elastic.co/elasticsearch/elasticsearch:7.7.0
   docker image ls
   # Docker with 6Gb of Ram, running on Port 9200 --name is <name_of_elastic_instance>
   docker run -m 6G -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" --name my_elastic_stack docker.elastic.co/elasticsearch/elasticsearch:7.7.0
   docker ps
   docker stats
   # to start docker app run follwing
   docker exec -it my_elastic_stack bash
  ```
3. Set up directories
```yaml
        pwd
        cd /usr/share/elasticsearch/
        mkdir searchqa
        cd searchqa
   ```
4. Install packages
```yaml
    # to know linux release version
    # If Fedora or red hat used yml 
    # if ubuntu use sudo apt-get
    
    cat /etc/*-release
    
    yum -y update
    yum install -y python3
    yum install -y vim
    yum -y install wget
    Yum install –y tar
    yum clean all
    
    pip3.6 install --upgrade pip
    pip3.6 --version
    pip3.6 install elasticsearch
    pip3.6 install pandas
    # https://tfhub.dev/google/universal-sentence-encoder/4

    pip3.6 install --upgrade --no-cache-dir tensorflow
    pip3.6 install --upgrade tensorflow-hub
   ```
5. Download Universal sentence encoder model from Tfhub and copy to Docker location
```yaml
    Download dataset from : https://www.kaggle.com/datasets/stackoverflow/stacksample
    # Copy datazip file from local to docker
    docker cp stack_sampels.zip my_elastic_stack:/usr/share/elasticsearch/searchqa/
    
    mkdir data
    # move all csv file to data folder after unzip
    mv *.csv ./data/
    rm-rf stack_sampels.zip
    
    # test if elasticsearch is running 
    https:://localhost:9200/
    
```
6. Set up Tf Model
```yaml
    Download model from : https://tfhub.dev/google/universal-sentence-encoder/4
    # Copy zip file from local to docker
    docker cp universal-sentence-encoder_4.tar.gz my_elastic_stack:/usr/share/elasticsearch/searchqa/data/
    yum install –y tar
    tar -xvzf universal-sentence-encoder_4.tar.gz -C ./USE4/
```
7. Create ElasticSearch Index
```yaml
    python index.py

    # to check how many indexes is created
    curl -X GET "localhost:9200/questions-index/_stats?pretty"
    
    # Search by Id
    http://localhost:9200/questions-index/_doc/80
    

```
7. Create A Flask API Search
```yaml
    pip3.6 install flask
    # set updated locale, if this gives error Google It.
    LC_ALL=en_US
    export LC_ALL
    export FLASK_APP=search_controller.py
    python3.6 -m flask run
    
    

```
8. Test Python API
```yaml
  time curl http://127.0.0.1:5000/search/how+to+install+python 
```

### Project Diagram


### Results 

## Other Extended Applications:
    
* Case Management in Customer Service Portal
* Discussion Forum
* Given Query Find answer from Video or Audio


# Semantic-Search-On-Question-Answers
High Level Design Of Semantic Search On Stack Overflow question answers dataset.
