import sys
import os

BASE_DIR = "/data2/jiangsiyu/ATNP_Database"
sys.path.append(BASE_DIR)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ATNP.settings")

import django
django.setup()

from matchms.importing import load_from_mgf
import base64
from io import BytesIO

# -------------------------------
# 读取 MGF 文件，只取第一个谱图
# -------------------------------
def load_single_spectrum_from_mgf(mgf_path):
    spectra = list(load_from_mgf(mgf_path))
    if len(spectra) == 0:
        raise ValueError(f"No spectrum found in MGF file: {mgf_path}")
    return spectra[0]


# -------------------------------
# 文件版：两个 MGF → 镜像图 base64
# -------------------------------
def plot_2_spectrum_from_file(file1, file2, loss=False):
    from web.utils.plot_tools import plot_2_spectrum   # 用你现成的函数

    spec1 = load_single_spectrum_from_mgf(file1)
    spec2 = load_single_spectrum_from_mgf(file2)

    return plot_2_spectrum(spec1, spec2, loss=loss)

import base64

img_base64 = plot_2_spectrum_from_file("/data2/jiangsiyu/ATNP_Database/web/utils/text_result/specA.mgf", "/data2/jiangsiyu/ATNP_Database/web/utils/text_result/specB.mgf")

with open("/data2/jiangsiyu/ATNP_Database/web/utils/text_result/mirror_plot.png", "wb") as f:
    f.write(base64.b64decode(img_base64))

print("完成！mirror_plot.png 已生成。")
