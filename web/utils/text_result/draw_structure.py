import os
from rdkit import Chem
from rdkit.Chem import Draw

def plot_ref_mol_png(smi_ref, output_path):
    if not smi_ref:
        return False
    mol_ref = Chem.MolFromSmiles(smi_ref)
    if mol_ref is None:
        return False
    img = Draw.MolToImage(mol_ref, wedgeBonds=False)
    img.save(output_path, format='PNG')
    return True

def parse_mgf_and_plot(mgf_file, output_dir="mgf_structures"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(mgf_file, "r", encoding="utf-8") as f:
        content = f.read()

    spectra_blocks = content.strip().split("BEGIN IONS")[1:]  # 分割每条谱图
    total = 0
    for block in spectra_blocks:
        lines = block.strip().splitlines()
        title = f"spectrum_{total}"
        smiles = None
        for line in lines:
            if line.startswith("TITLE="):
                title = line.split("=",1)[1].strip()
            elif line.startswith("SMILES="):
                smiles = line.split("=",1)[1].strip()
        
        if not smiles:
            print(f"[Warning] Spectrum '{title}' has no SMILES, skipped.")
            total += 1
            continue

        safe_title = "".join([c if c.isalnum() else "_" for c in title])
        img_path = os.path.join(output_dir, f"{safe_title}.png")
        if plot_ref_mol_png(smiles, img_path):
            print(f"Saved structure image: {img_path}")
        else:
            print(f"[Error] Failed to generate image for '{title}'")
        total += 1

    print(f"Total {total} spectra processed.")

if __name__ == "__main__":
    mgf_file = "/data2/jiangsiyu/ATNP_Database/web/utils/text_result/specB.mgf"
    parse_mgf_and_plot(mgf_file)
