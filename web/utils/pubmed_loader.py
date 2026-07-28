## 化合物所在文献部分加载pkl
import pickle

INDEX_PATH = "/data2/jiangsiyu/ATNP_Database/pubmed_index.pkl"

with open(INDEX_PATH, "rb") as f:
    PUBMED_INDEX = pickle.load(f)