import pickle
from pathlib import Path

pkl_path = Path(__file__).resolve().parents[2] / "model_copy" / "herbs_spectra_neg.pickle"

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

print("整体类型:", type(data))
print("第一条数据类型:", type(data[0]))
print("第一条 spectrum 字段类型:", type(data[0]["spectrum"]))
print("第一条 vector 字段类型:", type(data[0]["vector"]))
print("第一条数据",data[0])
print("第一条 spectrum",data[0]["spectrum"])
print("第一条 vector",data[0]["vector"])
