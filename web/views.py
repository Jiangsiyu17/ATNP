from django.shortcuts import render, get_object_or_404, redirect
from django.core.cache import cache
from .models import CompoundLibrary
from django.db.models.functions import Lower
from django.db.models import Q 
from django.core.paginator import Paginator
from web.utils.plotting import plot_ref_mol, generate_spectrum_comparison
import re
import itertools
from urllib.parse import unquote
from django.utils.text import slugify
from collections import defaultdict
import unicodedata
from web.utils.plot_tools import plot_ref_mol
import hashlib


def _fallback_mz(pepmass):
    """从 PEPMASS 字段提取 m/z（返回 float 或 None）"""
    if pepmass in (None, ""):
        return None
    if isinstance(pepmass, (list, tuple)):
        pepmass = pepmass[0]
    # 匹配第一个数字（支持科学计数法）
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(pepmass))
    return float(m.group()) if m else None

def _canon(s: str | None) -> str:
    """大小写、空白、全半角统一后的 key"""
    if not s:
        return ""
    # 全形 -> 半形
    s = unicodedata.normalize("NFKC", s)
    return s.strip().casefold()          # casefold 比 lower 更“彻底”

def compound_list(request):
    query = request.GET.get("query", "")
    field = request.GET.get("field", "standard")

    # ① 优化查询：只筛选 spectrum_type = "sample" 并在数据库层面处理
    qs = CompoundLibrary.objects.filter(spectrum_type="sample")

    # ---------- ② 查询过滤 ----------
    if query:
        lookups = {
            "standard": Q(standard__icontains=query),
            "precursor_mz": Q(precursor_mz__icontains=query),
            "database": Q(database__icontains=query),
            "smiles": Q(smiles__icontains=query),
        }
        qs = qs.filter(lookups.get(field, Q()))

    # ---------- ③ 优化：去除 unnecessary `annotate`，简化查询 ----------
    # 直接获取需要的字段，不要做多余的 `annotate`
    raw = qs.values("id", "standard", "precursor_mz", "database", "smiles", "pepmass") \
             .order_by(Lower("standard"))

    # ---------- ④ 优化：直接在数据库层做去重分组，避免在内存中处理 ----------
    # 利用 defaultdict 存储 unique 数据
    rows_dict = defaultdict(lambda: {
        "standard": None,
        "first_id": None,
        "precursor_mz": None,
        "smiles": None,
        "databases": set(),
    })

    for item in raw:
        canon_key = _canon(item["standard"])  # 规范化处理
        r = rows_dict[canon_key]

        # 记录首次出现时的字段
        if r["standard"] is None:
            r["standard"] = item["standard"] or "(unknown)"
            r["first_id"] = item["id"]
            r["precursor_mz"] = item["precursor_mz"] or _fallback_mz(item["pepmass"])
            r["smiles"] = item["smiles"]

        # 合并数据库名称，避免重复
        if item["database"]:
            normalized_db = _canon(item["database"])
            if normalized_db in {"nist20", "nist"}:
                r["databases"].add("NIST")
            else:
                r["databases"].add(normalized_db.upper())

    # ---------- ⑤ dict → list，并整理数据库列 ----------
    rows = [{
        "standard": r["standard"],
        "first_id": r["first_id"],
        "precursor_mz": f"{r['precursor_mz']:.4f}" if r["precursor_mz"] else "-",
        "database": ", ".join(sorted(db.upper() for db in r["databases"])) or "-",
        "smiles": r["smiles"],
    } for r in rows_dict.values()]

    # 按标准化名称再次排序
    rows.sort(key=lambda x: _canon(x["standard"]))

    # ---------- ⑥ 分页 ----------
    paginator = Paginator(rows, 20)  # 使用分页器，避免加载全部数据
    page_obj = paginator.get_page(request.GET.get("page"))

    return render(request, "web/compound_list.html", {
        "compounds": page_obj.object_list,
        "page_obj": page_obj,
        "query": query,
        "field": field,
        "no_result": query and paginator.count == 0,
    })


def herb_list(request):
    query        = request.GET.get("query", "")
    search_field = request.GET.get("field", "latin_name")

    base = (CompoundLibrary.objects
            .exclude(latin_name__isnull=True)      
            .exclude(latin_name=""))

    if query:
        lookups = {
            "latin_name":   Q(latin_name__icontains=query),
            "chinese_name": Q(chinese_name__icontains=query),
            "pinyin":       Q(herb_pinyin__icontains=query),
        }
        base = base.filter(lookups.get(search_field, Q()))

    # 把所有需要的字段都拿出来
    qs = (base
          .annotate(latin_lower=Lower("latin_name"))
          .values("latin_lower", "latin_name", "chinese_name", "tissue")
          .order_by("latin_lower", "tissue"))

    rows = []
    for latin, group in itertools.groupby(qs, key=lambda x: x["latin_lower"]):
        group_list = list(group)
        first      = group_list[0]
        tissues    = sorted({g["tissue"] for g in group_list if g["tissue"]})
        rows.append({
            "latin_lower": latin,                        # ★ 模板用来反向解析
            "latin_name":  first["latin_name"],          # 备用，如需展示
            "chinese_name": first["chinese_name"],
            "tissue":      ", ".join(tissues) or "-",    # ★ 模板列名保持一致
        })

    paginator = Paginator(rows, 20)
    page_obj  = paginator.get_page(request.GET.get("page"))

    return render(request, "web/herb_list.html", {
        "page_obj": page_obj,
        "herbs":    page_obj.object_list,
        "query":    query,
        "field":    search_field,
        "no_result": query and paginator.count == 0,
    })

def herb_detail(request, latin_name):
    # 获取与当前 latin_name 对应的化合物
    compounds = CompoundLibrary.objects.filter(latin_name__iexact=latin_name)

    # 使用 defaultdict 合并相同 standard 的记录
    rows_dict = defaultdict(lambda: {
        "standard": None,
        "first_id": None,
        "precursor_mz": None,
        "smiles": None,
        "databases": set(),
        "pepmass": None,  # 添加 pepmass 字段
    })

    for compound in compounds:
        canon_key = _canon(compound.standard)  # 标准化处理
        r = rows_dict[canon_key]

        if r["standard"] is None:
            r["standard"] = compound.standard or "(unknown)"
            r["first_id"] = compound.id  # 确保是有效的 ID
            r["precursor_mz"] = compound.precursor_mz or _fallback_mz(compound.pepmass)
            r["smiles"] = compound.smiles
            r["pepmass"] = compound.pepmass  # 确保传递 pepmass 字段

        if compound.database:
            normalized_db = _canon(compound.database).replace("nist20", "nist")
            r["databases"].add(normalized_db)

    # 生成最终的行数据
    rows = [{
        "standard": r["standard"],
        "first_id": r["first_id"],
        "precursor_mz": f"{r['precursor_mz']:.4f}" if r["precursor_mz"] else "-",
        "database": ", ".join(sorted(r["databases"])) or "-",
        "smiles": r["smiles"],
        "pepmass": r["pepmass"],  # 添加 pepmass 字段
    } for r in rows_dict.values()]

    rows.sort(key=lambda x: _canon(x["standard"]))

    # 确保传递给模板的数据没有问题
    return render(request, 'web/herb_detail.html', {
        'latin_name': latin_name,
        'compounds': rows,
    })

def home(request):
    return render(request, 'web/home.html')

def search(request):
    query = request.GET.get('query', '')
    category = request.GET.get('category', 'compounds')

    if category == 'herbs':
        return redirect(f'/herbs/?query={query}&field=latin_name')
    else:
        return redirect(f'/compound/list?query={query}')


def build_matchms_spectrum(peaks, metadata=None):
    import numpy as np
    from matchms import Spectrum
    if not peaks:
        return None
    if isinstance(peaks[0], dict):
        mz = np.array([float(p['mz']) for p in peaks], dtype=float)  # 改这里
        intensities = np.array([float(p['int']) for p in peaks], dtype=float)  # 改这里
    else:
        mz, intensities = zip(*peaks)
        mz = np.array(mz, dtype=float)  # 改这里
        intensities = np.array(intensities, dtype=float)  # 改这里
    print(f"mz dtype: {mz.dtype}, intensities dtype: {intensities.dtype}")
    return Spectrum(mz=mz, intensities=intensities, metadata=metadata or {})


def get_similar_sample_sources(compound, top_k=50):
    import pickle, hnswlib, json
    import numpy as np
    from gensim.models import KeyedVectors
    from spec2vec import SpectrumDocument
    from spec2vec.vector_operations import calc_vector
    from matchms.filtering import normalize_intensities
    from web.models import CompoundLibrary
    from django.conf import settings
    import os

    MODEL_DIR = os.path.join(settings.BASE_DIR, "model")
    POS_MODEL_PATH = os.path.join(MODEL_DIR, "Ms2Vec_allGNPSpositive.hdf5")
    NEG_MODEL_PATH = os.path.join(MODEL_DIR, "Ms2Vec_allGNPSnegative.hdf5")
    POS_INDEX_PATH = os.path.join(MODEL_DIR, "herbs_index_pos.bin")
    NEG_INDEX_PATH = os.path.join(MODEL_DIR, "herbs_index_neg.bin")
    POS_SPEC_PATH = os.path.join(MODEL_DIR, "herbs_spectra_pos.pickle")
    NEG_SPEC_PATH = os.path.join(MODEL_DIR, "herbs_spectra_neg.pickle")
    
    ionmode = (compound.ionmode or "positive").lower()
    if ionmode == "negative":
        model_path = NEG_MODEL_PATH
        index_path = NEG_INDEX_PATH
        spec_path = NEG_SPEC_PATH
    else:
        model_path = POS_MODEL_PATH
        index_path = POS_INDEX_PATH
        spec_path = POS_SPEC_PATH

    try:
        model = KeyedVectors.load(model_path)
        with open(spec_path, "rb") as f:
            sample_spectra = pickle.load(f)
        index = hnswlib.Index(space='l2', dim=300)
        index.load_index(index_path)
        index.set_ef(200)
    except Exception as e:
        print(f"[ERROR] 加载模型失败: {e}")
        return []

    peaks = compound.peaks
    if isinstance(peaks, str):
        peaks = json.loads(peaks)

    spectrum = build_matchms_spectrum(peaks, metadata={"compound_name": compound.standard})
    spectrum = normalize_intensities(spectrum)
    query_vec = calc_vector(model, SpectrumDocument(spectrum, n_decimals=2), allowed_missing_percentage=100)
    if query_vec is None:
        return []

    query_vec /= np.linalg.norm(query_vec)

    idx, distances = index.knn_query(query_vec, k=top_k)
    similar_samples = []
    for i, d in zip(idx[0], distances[0]):
        score = 1 - d / 4.0  # spec2vec 距离转为相似度
        metadata = sample_spectra[i].metadata
        similar_samples.append({
            "latin_name": metadata.get("latin_name"),
            "chinese_name": metadata.get("chinese_name"),
            "tissue": metadata.get("tissue"),
            "score": round(score, 4),
        })
    return similar_samples

def compound_detail(request, pk):
    compound = get_object_or_404(CompoundLibrary, pk=pk)
    mol_img = plot_ref_mol(compound.smiles) if compound.smiles else None

    # 查询草药来源信息
    plant_sources_raw = (
        CompoundLibrary.objects
        .filter(standard=compound.standard)
        .exclude(latin_name__isnull=True)
        .exclude(latin_name__exact='')
        .values("chinese_name", "latin_name")
        .distinct()
        .order_by("latin_name")
    )
    plant_sources = [
        {
            "chinese_name": ps["chinese_name"],
            "latin_name": ps["latin_name"],
            "latin_slug": slugify(ps["latin_name"]),
            "compound_raw": compound.standard,
        }
        for ps in plant_sources_raw
    ]

    # 获取与该化合物谱图相似的植物样本（sample库）
    similar_samples = get_similar_sample_sources(compound)

    context = {
        "compound": compound,
        "mol_img": mol_img,
        "plant_sources": plant_sources,
        "similar_samples": similar_samples,  # 新增相似植物样本列表
    }

    return render(request, "web/compound_detail.html", context)


def make_cache_key(prefix, latin_name, compound):
    raw_key = f"{latin_name}_{compound}"
    key_hash = hashlib.md5(raw_key.encode('utf-8')).hexdigest()
    return f"{prefix}_{key_hash}"

def herb_compound_detail(request, latin_name, compound):
    latin_name = unquote(latin_name)
    compound = unquote(compound)

    real_latin_name = None
    for name in CompoundLibrary.objects.values_list("latin_name", flat=True).distinct():
        if not name:
            continue
        if slugify(name) == latin_name:
            real_latin_name = name
            break

    if real_latin_name is None:
        entries = CompoundLibrary.objects.none()
        only_nist = False
    else:
        entries = CompoundLibrary.objects.filter(
            latin_name=real_latin_name, standard=compound
        ).order_by("tissue")

        databases = set(
            CompoundLibrary.objects.filter(standard=compound)
            .values_list("database", flat=True)
            .distinct()
        )
        databases = {db.lower() for db in databases if db}
        only_nist = all("nist" in db for db in databases) if databases else False

    comparison_list = []
    if entries.exists():
        cache_key = make_cache_key("spectrum_comparison", real_latin_name, compound)
        comparison_list = cache.get(cache_key)

        if comparison_list is None:
            comparison_list = generate_spectrum_comparison(entries, only_nist=only_nist)
            cache.set(cache_key, comparison_list, timeout=60 * 15)

    matched_ids = {item["sample"].id for item in comparison_list}

    return render(request, "web/herb_compound_detail.html", {
        "latin_name": real_latin_name or latin_name,
        "compound": compound,
        "entries": entries,
        "comparison_list": comparison_list,
        "only_nist": only_nist,
        "matched_ids": matched_ids,
    })