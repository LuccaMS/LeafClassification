import os

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from helpers.ImageEmbeddings import ImageEmbedding
from dotenv import load_dotenv

load_dotenv()


class ElasticSearch:
    def __init__(self, index, images_path):
        self.images_path = images_path
        self.host = os.getenv("ELASTICSEARCH_HOST")
        self.es_client = Elasticsearch(
            hosts=[self.host],
            sniff_on_connection_fail=True,
            sniff_on_start=True,
            min_delay_between_sniffing=600,
            request_timeout=600,
            sniff_timeout=300,
            max_retries=5,
            retry_on_timeout=True,
        )
        self.emb_client = ImageEmbedding(self.images_path)
        self.index = index

    def upload_item(self, item_name):
        item_to_upload = self.build_item_to_upload(item_name)
        try:
            bulk(self.es_client, [item_to_upload])
        except Exception as e:
            print(f"Error while uploading file {item_name} - {e}")

    def upload_all_items(self):
        all_items = []
        for items_name in os.listdir(self.images_path):
            all_items.append(self.build_item_to_upload(items_name))
        try:
            bulk(self.es_client, all_items)
        except Exception as e:
            print(f"Error while uploading files in path {self.images_path} - {e}")

    def build_item_to_upload(self, item_name):
        emb = self.get_embedding_list(item_name)
        no = item_name.split(".")[0]

        source = {"no": no, "name": item_name, "embedding": emb.tolist()}

        return {"_index": self.index, "_source": source}

    def get_embedding_list(self, item_name):
        self.emb_client.generate_embedding(item_name)
        return self.emb_client.embeddings[item_name]

    def search(self, item_name):
        return self.es_client.search(
            index=self.index,
            body={
                "size": 1,
                "query": {"match": {"name": item_name}},
                "_source": {"includes": ["text"]},
            },
        )

    def get_n_similar(self, item_name, n):
        embedding = self.get_embedding_list(item_name)
        return self.es_client.search(
            index=self.index,
            body={
                "size": n,
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity("
                            "params.queryVector, "
                            "'embedding'"
                            ") + 1.0",
                            "params": {"queryVector": embedding},
                        },
                    }
                },
            },
        )

    def get_index_mapping(self):
        # Use the indices.get() method to retrieve the index mapping
        index_mapping = self.es_client.indices.get(index=self.index)

        # Return the mapping for the index
        return index_mapping

    def get_all_indexes(self):
        # Use the indices.get_alias() method to retrieve a list of all indexes
        indexes = list(self.es_client.indices.get_alias().keys())

        # Return the list of indexes
        return indexes

    def refresh_index(self):
        self.es_client.indices.refresh(index=self.index)
