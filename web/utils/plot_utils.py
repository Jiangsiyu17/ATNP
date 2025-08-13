# 可视化化合物结构（SMILES）和谱图比对结果（镜像图），并将生成的图像 转换为 Base64 编码

import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from matchms import Spectrum
import base64
import io

def plot_ref_mol(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        img = Draw.MolToImage(mol, size=(300, 300))
        import io, base64
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        return encoded
    except Exception as e:
        print(f"Mol image plot error: {e}")
        return None

def plot_two_spectra(spectrum1, spectrum2, output_path):
    """Draw mirror plot of two spectra and save to image file."""
    fig, ax = plt.subplots(figsize=(10, 6))
    # mirror_plot(spectrum1, spectrum2, ax=ax)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_spectra(query_spectrum: Spectrum, reference_spectrum: Spectrum) -> str:
    fig, ax = plt.subplots(figsize=(10, 6))
    # mirror_plot(query_spectrum, reference_spectrum, ax=ax)
    ax.set_title("Mirror Plot of Query vs Reference")
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_base64