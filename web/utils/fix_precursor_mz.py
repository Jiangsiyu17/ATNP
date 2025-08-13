import pickle

def safe_parse_pepmass(pepmass):
    try:
        if isinstance(pepmass, (list, tuple)) and len(pepmass) > 0:
            return float(pepmass[0])
        elif isinstance(pepmass, str):
            parts = pepmass.strip().split()
            if len(parts) >= 1:
                return float(parts[0])
        elif isinstance(pepmass, float) or isinstance(pepmass, int):
            return float(pepmass)
    except Exception as e:
        print(f"[⚠️] pepmass 解析失败: {pepmass} -> {e}")
    return None

def fix_precursor_mz_in_pickle(pickle_path: str):
    with open(pickle_path, "rb") as f:
        spectra = pickle.load(f)

    fixed = 0
    for spec in spectra:
        if spec.get("precursor_mz") is None:
            pepmass = spec.get("pepmass")
            precursor_mz = safe_parse_pepmass(pepmass)
            if precursor_mz is not None:
                spec.set("precursor_mz", precursor_mz)
                fixed += 1

    with open(pickle_path, "wb") as f:
        pickle.dump(spectra, f)

    print(f"✅ 修复完成：已添加 precursor_mz 至 {fixed} 个谱图。")

if __name__ == "__main__":
    fix_precursor_mz_in_pickle("/data2/jiangsiyu/ATNP_Database/model/herbs_spectra_pos.pickle")
    fix_precursor_mz_in_pickle("/data2/jiangsiyu/ATNP_Database/model/herbs_spectra_neg.pickle")
