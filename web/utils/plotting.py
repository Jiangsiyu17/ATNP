from django.core.cache import cache
from web.models import CompoundLibrary
from .plot_tools import plot_2_spectrum, plot_single_spectrum
from rdkit import Chem
from rdkit.Chem import Draw
import io, base64
from django.utils.text import slugify

def get_precursor_mz(obj):
    """
    优先尝试返回浮点型 precursor_mz 字段，
    否则尝试解析 pepmass 字符串字段。
    """
    if obj.precursor_mz is not None:
        try:
            return float(obj.precursor_mz)
        except Exception:
            pass
    val = obj.pepmass
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        try:
            return float(val[0])
        except Exception:
            return None
    if isinstance(val, str):
        try:
            return float(val.strip().split()[0])
        except Exception:
            return None
    try:
        return float(val)
    except Exception:
        return None

def mz_within(mz1, mz2, ppm_tol=20):
    """
    判断mz1和mz2是否在ppm_tol容差范围内
    """
    if mz1 is None or mz2 is None:
        return False
    return abs(mz1 - mz2) / mz1 * 1e6 <= ppm_tol

def is_valid_spectrum(spec):
    if spec is None:
        return False
    if not hasattr(spec, "peaks") or spec.peaks is None:
        return False
    mzs = getattr(spec.peaks, "mz", None)
    ints = getattr(spec.peaks, "intensities", None)
    if mzs is None or ints is None:
        return False
    if len(mzs) == 0 or len(ints) == 0:
        return False
    return True

def match_std_name(std_name, ref_name):
    if not std_name or not ref_name:
        return False
    return slugify(std_name) in slugify(ref_name) or slugify(ref_name) in slugify(std_name)

def make_cache_key(*args):
    """
    简单生成缓存 key
    """
    # 拼接参数并替换空格/特殊字符
    key = "_".join(str(a) for a in args)
    key = key.replace(" ", "_")
    return key

def get_plot_base64(sample_obj: CompoundLibrary, candidates=None, only_nist=False) -> str | None:
    sample_spec = sample_obj.get_spectrum()
    if not is_valid_spectrum(sample_spec):
        print(f"[Invalid] Sample spectrum is invalid or empty for {sample_obj.title}")
        return None

    try:
        # 如果要求只显示 NIST 库 → 单谱
        if only_nist:
            print(f"[Only-NIST] Drawing single spectrum for {sample_obj.title}")
            return plot_single_spectrum(sample_spec)

        # 如果传入了 candidates（通过 matched_spectrum_id 查找的）
        if candidates:
            print(f"[Match] Sample '{sample_obj.title}' → using {len(candidates)} candidates directly")
            for ref in candidates:
                reference_spec = ref.get_spectrum()
                if is_valid_spectrum(reference_spec):
                    print(f"[Compare] Plotting {sample_obj.title} vs {ref.title}")
                    return plot_2_spectrum(sample_spec, reference_spec)
                else:
                    print(f"[Invalid Ref] Reference spectrum invalid for {ref.title}")
            # 如果 candidates 都无效，就退回单谱
            return plot_single_spectrum(sample_spec)

        # 没有 candidates → 单谱
        print(f"[No Match] Drawing single for {sample_obj.title}")
        return plot_single_spectrum(sample_spec)

    except Exception as e:
        print("[Error] plot_2_spectrum:", e)
        return None

def get_cached_spectrum_plot(sample_obj: CompoundLibrary, candidates=None, only_nist=False) -> str | None:
    cache_key = f"spectrum_plot_{sample_obj.id}_{sample_obj.matched_spectrum_id}_{sample_obj.ionmode}_{only_nist}"

    cached_image = cache.get(cache_key)

    if cached_image is None:
        print(f"[Cache Miss] Generating plot for sample id={sample_obj.id}")
        cached_image = get_plot_base64(sample_obj, candidates, only_nist)
        if cached_image:
            cache.set(cache_key, cached_image, timeout=3600)
    else:
        print(f"[Cache Hit] Found cached plot for sample id={sample_obj.id}")

    return cached_image

def generate_spectrum_comparison(entries, only_nist=False, min_score=0.0, standards=None):
    """
    entries: 样品库 CompoundLibrary 对象列表（含 spectrum_blob）
    standards: 标品 CompoundLibrary 列表（可为空）
    """
    import pickle

    results = []

    for sample in entries:
        # ---------- 取匹配 ID / score ----------
        matched_id_raw = getattr(sample, "matched_spectrum_id", None)
        matched_id = str(matched_id_raw).strip() if matched_id_raw else None
        ionmode = (sample.ionmode or "").lower()
        sample_score = getattr(sample, "score", 0.0) or 0.0

        if not matched_id or sample_score < min_score:
            continue

        # ---------- 取 Spectrum 对象 ----------
        spectrum = getattr(sample, "spectrum", None)
        if spectrum is None:
            # 从 pickle 恢复 matchms Spectrum
            spectrum = pickle.loads(sample.spectrum_blob)
            spectrum.metadata["db_id"] = sample.id
            spectrum.db_id = sample.id
            sample.spectrum = spectrum

        # ---------- 从标品列表中找候选 ----------
        if standards is not None:
            candidates = [
                s for s in standards
                if str(s.standard_id or "").strip() == str(matched_id or "").strip()
                and (s.ionmode or "").lower() == (ionmode or "").lower()
            ]
            if candidates:
                candidates = [candidates[0]]
        else:
            candidates = []

        # ---------- 绘制图像 ----------
        img = get_cached_spectrum_plot(
            sample, 
            candidates=candidates, 
            only_nist=only_nist
        )

        if img:
            results.append({
                "sample": sample,
                "image": img,
                "score": sample_score,
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def plot_ref_mol(smi_ref):
    try:
        if not smi_ref:
            return None

        smi_ref = smi_ref.strip().replace('"', '')
        mol_ref = Chem.MolFromSmiles(smi_ref)
        if mol_ref is None:
            print(f"[plot_ref_mol] Invalid SMILES: {smi_ref}")
            return None

        img = Draw.MolToImage(mol_ref, size=(300, 300))
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return img_base64

    except Exception as e:
        print(f"[ERROR] plot_ref_mol failed: {e}")
        return None
    

def format_latin_name(name: str) -> str:
    """
    规范化拉丁学名:
      - 属名首字母大写
      - 其余部分全部小写
    例：
      "panax ginseng" -> "Panax ginseng"
      "GLYCYRRHIZA URALENSIS" -> "Glycyrrhiza uralensis"
      "camellia SINENSIS" -> "Camellia sinensis"
    """
    if not name:
        return ""
    parts = name.split()
    if not parts:
        return ""
    genus = parts[0].capitalize()
    rest = [p.lower() for p in parts[1:]]
    return " ".join([genus] + rest)
