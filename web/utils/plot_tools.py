import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matchms import filtering as msfilters
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdFMCS
from itertools import chain
import io
import base64


def plot_2_spectrum(spectrum, reference, loss=False):
    """
    Mirror plot: 上=样品红，下=参考蓝；两谱都各自归一化到 1
    """
    mz, intensity = spectrum.peaks.mz, spectrum.peaks.intensities
    mz_ref, inten_ref = reference.peaks.mz, reference.peaks.intensities

    # 可选：画 neutral loss
    if loss:
        try:
            spectrum  = msfilters.add_parent_mass(spectrum)
            spectrum  = msfilters.add_losses(spectrum, 10.0, 2000.0)
            reference = msfilters.add_parent_mass(reference)
            reference = msfilters.add_losses(reference, 10.0, 2000.0)
            mz,  intensity  = spectrum.losses.mz,   spectrum.losses.intensities
            mz_ref, inten_ref = reference.losses.mz, reference.losses.intensities
        except Exception:
            pass

    # ----------- 归一化：各自最大峰 = 1 -----------
    if intensity.size and intensity.max() > 0:
        intensity = intensity / intensity.max()
    if inten_ref.size and inten_ref.max() > 0:
        inten_ref = inten_ref / inten_ref.max()

    fig = Figure(figsize=(5, 3), dpi=100)
    ax  = fig.add_subplot(111)
    fig.subplots_adjust(top=0.95, bottom=0.3, left=0.18, right=0.95)

    ax.vlines(mz,      0,  intensity,  color="r", lw=0.5)
    ax.vlines(mz_ref,  0, -inten_ref,  color="b", lw=0.5)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("m/z", fontsize=8)
    ax.set_ylabel("relative abundance", fontsize=8)
    ax.set_ylim(-1.05, 1.05)

    buf = io.BytesIO()
    FigureCanvas(fig).print_png(buf)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def plot_single_spectrum(spectrum):
    import matplotlib.pyplot as plt
    import io
    import base64
    import numpy as np

    # 归一化强度
    intensities = np.array(spectrum.intensities)
    max_intensity = intensities.max() if intensities.size > 0 else 1.0
    intensities = intensities / max_intensity * 100  # 归一化为百分比

    mz_values = np.array(spectrum.mz)

    # 画图
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.vlines(mz_values, 0, intensities, color='blue', linewidth=1)
    ax.set_xlabel('m/z')
    ax.set_ylabel('Normalized Intensity (%)')
    # 不设置标题
    ax.set_xlim(left=max(0, mz_values.min() - 50), right=mz_values.max() + 50)
    ax.set_ylim(0, 110)
    plt.tight_layout()

    # 输出为 base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_b64

def plot_ref_mol(smi_ref):
    """
    将 SMILES 转为分子结构图并以 base64 字符串返回，供 HTML 显示使用
    """
    try:
        if not smi_ref:
            return None
        mol_ref = Chem.MolFromSmiles(smi_ref)
        img = Draw.MolToImage(mol_ref, wedgeBonds=False)

        # 转为 base64 编码字符串
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return img_base64
    except Exception as e:
        print(f"plot_ref_mol error: {e}")
        return None
