import os
from img2vec_pytorch import Img2Vec
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from alive_progress import alive_bar
from decorators.Timer import count_time  # noqa: F401
import numpy as np


# https://github.com/christiansafka/img2vec
class ImageEmbedding:
    def __init__(self, path):
        self.converter = Img2Vec()
        self.embeddings = {}
        self.base_path = path
        self.files_path = os.listdir(self.base_path) or 0
        self.total_files_length = len(self.files_path)
        self.similarities = {}

    def generate_embeddings(self):
        with alive_bar(self.total_files_length) as bar:
            for file in self.files_path:
                self.generate_embedding(file)
                bar()

    def generate_embedding(self, file):
        filename = os.fsdecode(file)
        img_path = os.path.join(self.base_path, filename)
        img = Image.open(img_path).convert("RGB")
        vec = self.converter.get_vec(img)
        self.embeddings[filename] = vec

    # @count_time
    def calculate_similarities(self, img_name):
        self.similarities[img_name] = {}
        for key in list(self.embeddings.keys()):
            if key == img_name:
                continue
            self.similarities[img_name][key] = cosine_similarity(
                self.embeddings[img_name].reshape((1, -1)),
                self.embeddings[key].reshape((1, -1)),
            )[0][0]

    def get_n_most_similar(self, img_name, n):
        if img_name not in self.similarities.keys():
            self.calculate_similarities(img_name)

        similar = self.similarities[img_name]

        d_view = [(v, k) for k, v in similar.items()]
        d_view.sort(reverse=True)
        return {k: v for v, k in d_view[:n]}

    def save(self):
        np.save("data/embeddings/emb.npy", self.embeddings)

    def load(self):
        self.embeddings = np.load("data/embeddings/emb.npy", allow_pickle=True).item()
