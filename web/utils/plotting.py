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

def get_plot_base64(sample_obj: CompoundLibrary, candidates=None, only_nist=False) -> str | None:
    sample_spec = sample_obj.get_spectrum()
    if not is_valid_spectrum(sample_spec):
        print(f"[Invalid] Sample spectrum is invalid or empty for {sample_obj.title}")
        return None

    try:
        if only_nist:
            print(f"[Only-NIST] Drawing single spectrum for {sample_obj.title}")
            return plot_single_spectrum(sample_spec)

        std_name = (sample_obj.standard or "").strip()
        mz_query = get_precursor_mz(sample_obj)

        if not std_name or mz_query is None:
            print(f"[Fallback] No std_name or mz for {sample_obj.title}, drawing single")
            return plot_single_spectrum(sample_spec)

        matched_refs = []
        for r in candidates or []:
            r_std = (r.standard or "").strip()
            r_mz = get_precursor_mz(r)
            if not r_std or r_mz is None:
                continue

            name_match = match_std_name(std_name, r_std)
            mz_match = mz_within(r_mz, mz_query)

            if name_match and mz_match:
                matched_refs.append(r)

        print(f"[Match] Sample '{sample_obj.title}' → matched {len(matched_refs)} references")

        if matched_refs:
            if all("nist" in (r.database or "").lower() for r in matched_refs):
                return plot_single_spectrum(sample_spec)
            else:
                for ref in matched_refs:
                    if "nist" not in (ref.database or "").lower():
                        reference_spec = ref.get_spectrum()
                        if is_valid_spectrum(reference_spec):
                            print(f"[Compare] Plotting {sample_obj.title} vs {ref.title}")
                            return plot_2_spectrum(sample_spec, reference_spec)
                        else:
                            print(f"[Invalid Ref] Reference spectrum invalid for {ref.title}")
                return plot_single_spectrum(sample_spec)

        print(f"[No Match] Drawing single for {sample_obj.title}")
        return plot_single_spectrum(sample_spec)

    except Exception as e:
        print("[Error] plot_2_spectrum:", e)
        return None

def get_cached_spectrum_plot(sample_obj: CompoundLibrary, candidates=None, only_nist=False) -> str | None:
    cache_key = f"spectrum_plot_{sample_obj.id}_{only_nist}"
    cached_image = cache.get(cache_key)

    if cached_image is None:
        print(f"[Cache Miss] Generating plot for sample id={sample_obj.id}")
        cached_image = get_plot_base64(sample_obj, candidates, only_nist)
        if cached_image:
            cache.set(cache_key, cached_image, timeout=3600)
    else:
        print(f"[Cache Hit] Found cached plot for sample id={sample_obj.id}")

    return cached_image

def generate_spectrum_comparison(entries, only_nist=False, ppm_tol=20):
    results = []

    for sample in entries:
        mz_query = get_precursor_mz(sample)
        if mz_query is None:
            continue

        sample_std = (sample.standard or "").strip().lower()
        print(f"[DEBUG] Sample '{sample.title}' std='{sample_std}' pepmass={mz_query} ppm_tol={ppm_tol}")

        if not sample_std:
            img = get_cached_spectrum_plot(sample, candidates=None, only_nist=only_nist)
            if img:
                results.append({"sample": sample, "image": img})
            continue

        # 精确匹配 precursor_mz，ppm_tol转化为绝对值范围，ppm_tol越小越严格
        mz_delta = mz_query * ppm_tol * 1e-6

        candidates = CompoundLibrary.objects.filter(
            spectrum_type="standard",
            precursor_mz__gte=mz_query - mz_delta,
            precursor_mz__lte=mz_query + mz_delta,
            standard__icontains=sample_std,
        )
        print(f"[DEBUG] Candidates found: {candidates.count()}")

        img = get_cached_spectrum_plot(sample, candidates=candidates, only_nist=only_nist)
        if img:
            results.append({"sample": sample, "image": img})

    return results

def plot_ref_mol(smi_ref):
    try:
        if not smi_ref:
            return None
        mol_ref = Chem.MolFromSmiles(smi_ref)
        img = Draw.MolToImage(mol_ref, wedgeBonds=False)

        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return img_base64
    except Exception as e:
        print(f"plot_ref_mol error: {e}")
        return None
