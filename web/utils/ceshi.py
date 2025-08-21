# check_index_pickle.py
import hnswlib
import pickle
import numpy as np
from spec2vec import SpectrumDocument
from spec2vec.vector_operations import calc_vector
from matchms.filtering import normalize_intensities

# 配置
PICKLE_PATH = "/data2/jiangsiyu/ATNP_Database/model/herbs_spectra_pos.pickle"    # 你的 pickle 文件
INDEX_PATH = "/data2/jiangsiyu/ATNP_Database/model/herbs_index_pos.bin"          # 你的 HNSW 索引文件
VECTOR_DIM = 300                             # spec2vec 向量维度

# 1️⃣ 加载 pickle
with open(PICKLE_PATH, "rb") as f:
    refs = pickle.load(f)

print(f"✅ Pickle loaded: {len(refs)} spectra")
print("Example metadata of first spectrum:", refs[0].metadata)

# 2️⃣ 加载 HNSW 索引
p = hnswlib.Index(space="cosine", dim=VECTOR_DIM)
p.load_index(INDEX_PATH)
print(f"✅ HNSW index loaded: {p.get_current_count()} vectors")

# 3️⃣ 检查数量是否一致
if len(refs) != p.get_current_count():
    print(f"⚠ Warning: Number of spectra ({len(refs)}) != index vectors ({p.get_current_count()})")
else:
    print("✅ Pickle and HNSW index counts match")

# 4️⃣ 测试查询第一个谱图
spec = refs[0]
spec_norm = normalize_intensities(spec)
doc = SpectrumDocument(spec_norm, n_decimals=2)
# 假设你有 Word2Vec 模型文件
# model = ... # 这里填你的 Word2Vec 模型路径加载
# query_vector = calc_vector(model, doc, allowed_missing_percentage=100)

# 这里先只验证索引能运行 knn_query
# 生成随机向量作为模拟测试
query_vector = np.random.rand(VECTOR_DIM).astype('float32')
query_vector /= np.linalg.norm(query_vector)
labels, distances = p.knn_query(query_vector, k=1)
print(f"✅ Example query works, nearest neighbor index: {labels[0][0]}, distance: {distances[0][0]}")
