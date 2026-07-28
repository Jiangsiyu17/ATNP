import json
import pickle
import re

index = {}

def normalize_name(name):
    if not name:
        return ""

    name = name.lower().strip()

    # 去掉 (E)- (Z)- 等前缀
    name = re.sub(r"^\(.*?\)-", "", name)

    # 多空格压缩
    name = " ".join(name.split())

    return name


def build_index(file_path, tag):
    with open(file_path, "r", encoding="utf-8") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break

            try:
                data = json.loads(line)
            except:
                continue

            raw_name = data.get("compound_name", "")
            name = normalize_name(raw_name)

            if not name:
                continue

            index.setdefault(name, []).append({
                "file": tag,
                "offset": pos
            })


# 构建索引
build_index("/data2/jiangsiyu/ATNP_Database/crawler/5_N_all_clean_pubmed_results.jsonl", "N")
build_index("/data2/jiangsiyu/ATNP_Database/crawler/5_P_all_clean_pubmed_results.jsonl", "P")

# 保存
with open("pubmed_index.pkl", "wb") as f:
    pickle.dump(index, f)

print("索引构建完成，共：", len(index), "个化合物")