import json
import pickle

index = {}

def build_index(file_path, tag):
    with open(file_path, "r", encoding="utf-8") as f:
        while True:
            pos = f.tell()  # 📍记录当前位置（字节）
            line = f.readline()
            if not line:
                break

            try:
                data = json.loads(line)
            except:
                continue

            name = data.get("compound_name", "").lower().strip()

            if not name:
                continue

            # 👉 支持一个化合物在两个文件中出现
            index.setdefault(name, []).append({
                "file": tag,
                "offset": pos
            })

# 两个文件
build_index("/data2/jiangsiyu/ATNP_Database/crawler/5_N_all_clean_pubmed_results.jsonl", "N")
build_index("/data2/jiangsiyu/ATNP_Database/crawler/5_P_all_clean_pubmed_results.jsonl", "P")

# 保存
with open("/data2/jiangsiyu/ATNP_Database/pubmed_index.pkl", "wb") as f:
    pickle.dump(index, f)

print("索引构建完成，共：", len(index), "个化合物")