from django.shortcuts import render, get_object_or_404, redirect
from django.core.cache import cache
from .models import CompoundLibrary
from django.db.models.functions import Lower
from django.db.models import Q 
from django.core.paginator import Paginator
from web.utils.plotting import plot_ref_mol, generate_spectrum_comparison, format_latin_name
import re
import itertools
from urllib.parse import unquote
from django.utils.text import slugify
from collections import defaultdict
import unicodedata
from web.utils.plot_tools import plot_ref_mol
import hashlib
from web.utils.identify import identify_spectrums
from urllib.parse import quote
from matchms.exporting import save_as_mgf
import logging
logger = logging.getLogger(__name__)
from django.urls import reverse
import logging
from matchms import Spectrum
import numpy as np
import pickle
from django.http import HttpResponse
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import warnings


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

    # 只显示有 PLANTS 字段的标品谱图
    qs = CompoundLibrary.objects.filter(
        spectrum_type__iexact="standard"
    ).filter(
        ~Q(plants__isnull=True) & ~Q(plants__regex=r'^\s*$')
    )

    # 支持搜索
    if query:
        lookups = {
            "standard": Q(standard__icontains=query),
            "precursor_mz": Q(precursor_mz__icontains=query),
            "database": Q(database__icontains=query),
            "smiles": Q(smiles__icontains=query),
        }
        qs = qs.filter(lookups.get(field, Q()))

    # 提取字段
    raw = qs.values("id", "standard", "precursor_mz", "database", "smiles", "pepmass", "plants", "ionmode") \
            .order_by(Lower("standard"))

    # 用 (standard, ionmode) 作为去重 key
    rows_dict = defaultdict(lambda: {
        "standard": None,
        "ionmode": None,
        "first_id": None,
        "precursor_mz": None,
        "smiles": None,
        "databases": set(),
        "plants": None,
    })

    for item in raw:
        canon_key = (_canon(item["standard"]), item.get("ionmode"))
        r = rows_dict[canon_key]

        if r["standard"] is None:
            r["standard"] = item["standard"] or "(unknown)"
            r["ionmode"] = item.get("ionmode")
            r["first_id"] = item["id"]
            r["precursor_mz"] = item["precursor_mz"] or _fallback_mz(item["pepmass"])
            r["smiles"] = item["smiles"]
            r["plants"] = item.get("plants")

        if item["database"]:
            normalized_db = _canon(item["database"])
            if normalized_db in {"nist20", "nist"}:
                r["databases"].add("NIST")
            else:
                r["databases"].add(normalized_db.upper())

    # 转换为列表
    rows = [{
        "standard": r["standard"],
        "ionmode": r["ionmode"],
        "first_id": r["first_id"],
        "precursor_mz": f"{r['precursor_mz']:.4f}" if r["precursor_mz"] else "-",
        "database": ", ".join(sorted(r["databases"])) or "-",
        "smiles": r["smiles"],
        "plants": r["plants"],
    } for r in rows_dict.values()]

    rows.sort(key=lambda x: _canon(x["standard"]))

    # 分页
    paginator = Paginator(rows, 20)
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
            "latin_name":  format_latin_name(first["latin_name"]),          # 备用，如需展示
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
    compounds = CompoundLibrary.objects.filter(
        latin_name__iexact=latin_name
    ).order_by(Lower("standard"))

    rows = []
    for c in compounds:

        # ========= 处理 precursor_mz，与 compound_list 完全一致 =========
        if c.precursor_mz:
            precursor = f"{c.precursor_mz:.4f}"
        elif c.pepmass:
            try:
                mz_value = float(c.pepmass.split()[0])
                precursor = f"{mz_value:.4f}"
            except:
                precursor = c.pepmass
        else:
            precursor = "-"

        # ========= database 标准化 =========
        if c.database:
            db = c.database.lower()
            db = db.replace("nist20", "nist")
            db = db.lower()
        else:
            db = "-"

        rows.append({
            "id": c.id,
            "standard": c.standard or "(unknown)",
            "precursor_mz": precursor,
            "database": db,
            "ionmode": c.ionmode or "-",
            "smiles": c.smiles or "-",
        })

    return render(request, 'web/herb_detail.html', {
        'latin_name': latin_name,
        'compounds': rows,
    })


def home(request):
    return render(request, 'web/home.html')

def search(request):
    query = request.GET.get("q", "").strip()

    if not query:
        return render(request, "search_not_found.html", {"query": query})

    # 1. 搜化合物（用 standard/title/smiles）
    compound = (CompoundLibrary.objects.filter(standard__icontains=query).first() or
                CompoundLibrary.objects.filter(title__icontains=query).first() or
                CompoundLibrary.objects.filter(smiles__icontains=query).first())

    if compound:
        return redirect(reverse("compound_detail", args=[compound.pk]))

    # 2. 搜植物（用 latin_name 或 chinese_name）
    herb = (CompoundLibrary.objects.filter(latin_name__icontains=query).first() or
            CompoundLibrary.objects.filter(chinese_name__icontains=query).first())

    if herb:
        # 注意：这里跳转 herb_detail，用 latin_name 作为参数
        return redirect(reverse("herb_detail", args=[herb.latin_name]))

    # 3. 都没找到
    return render(request, "search_not_found.html", {"query": query})


def parse_plants_field(plants_field):
    """
    解析 PLANTS=[P1:Chinese_name=卷叶欧芹;Latin_name=Petroselinum crispum var. crispum;Tissue=Root 2;matched_spectrum_id=3717];[P2:Chinese_name=虎耳草;Latin_name=Saxifraga stolonifera;Tissue=Whole plant;matched_spectrum_id=3717]
    返回列表：
    [
        {"chinese_name": "卷叶欧芹", "latin_name": "Petroselinum crispum var. crispum", "tissue": "Root 2", "matched_spectrum_id": "3717"},
        ...
    ]
    """
    pattern = r"Chinese_name=(.*?);Latin_name=(.*?);Tissue=(.*?);matched_spectrum_id=(\d+)"
    results = []
    for match in re.findall(pattern, plants_field):
        results.append({
            "chinese_name": match[0],
            "latin_name": match[1],
            "latin_slug": slugify(match[1]),
            "tissue": match[2],
            "matched_spectrum_id": match[3],
        })
    return results

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, QED
import rdkit.Chem.AllChem as AllChem
from web.utils import sascorer

def compound_detail(request, pk):
    logger = logging.getLogger(__name__)
    logger.info(f"→ Enter compound_detail, pk={pk}")

    # 获取化合物对象
    compound = get_object_or_404(CompoundLibrary, pk=pk)
    mol_img = plot_ref_mol(compound.smiles) if compound.smiles else None

    # ===== RDKit 计算性质 =====
    rdkit_props = None

    if compound.smiles:
        mol = Chem.MolFromSmiles(compound.smiles)

        if mol:
            # 先计算不报错的性质
            rdkit_props = {
                "mol_weight": round(Descriptors.MolWt(mol), 2),
                "num_rings": Lipinski.RingCount(mol),
                "num_aromatic_rings": Lipinski.NumAromaticRings(mol),
                "hbd": Lipinski.NumHDonors(mol),
                "hba": Lipinski.NumHAcceptors(mol),
                "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
                "logp": round(Crippen.MolLogP(mol), 2),
                "qed": round(float(QED.qed(mol)), 2),
            }

            # 再计算 SA score（单独 try，不影响整个 rdkit_props）
            try:
                sa = sascorer.calculateScore(mol)
                rdkit_props["sa_score"] = round(float(sa), 2)
            except Exception as e:
                print("SA score error:", e)
                rdkit_props["sa_score"] = "N/A"

            print("RDKit props computed:", rdkit_props)

    # ===== 表格 A：来源植物（解析 PLANTS 字段） =====
    plant_sources = []

    if compound.plants:
        # 匹配每个 [P1: ... ] 片段
        entries = re.findall(r"\[(P\d+:[^\]]+)\]", compound.plants)

        for entry in entries:
            chinese_name = re.search(r"Chinese_name=([^;]+)", entry)
            latin_name = re.search(r"Latin_name=([^;]+)", entry)
            tissue = re.search(r"Tissue=([^;]+)", entry)
            matched_id = re.search(r"matched_spectrum_id=([0-9]+)", entry)

            ps = {
                "chinese_name": chinese_name.group(1).strip() if chinese_name else "-",
                "latin_name": format_latin_name(latin_name.group(1).strip()) if latin_name else "-",
                "tissue": tissue.group(1).strip().capitalize() if tissue else "-",
                "matched_id": matched_id.group(1).strip() if matched_id else None,
                "latin_slug": slugify(latin_name.group(1)) if latin_name else "",
                "compound_raw": compound.standard,
                "compound_url": quote(compound.standard, safe=''),
            }
            plant_sources.append(ps)

    logger.info(f"→ Parsed {len(plant_sources)} plant sources from PLANTS field")

    
    # ✅ 去重：同一个中文名 + 拉丁名 只保留一条
    plant_sources_dict = {}
    for ps in plant_sources:
        key = (ps["chinese_name"], ps["latin_name"])
        if key not in plant_sources_dict:
            plant_sources_dict[key] = ps
    plant_sources = list(plant_sources_dict.values())

    logger.info(f"→ Plant_sources count: {len(plant_sources)}")

    # ===== 表格 B：谱图相似来源 =====
    cache_key = f"compound_identify_{compound.id}"
    results = None
    # results = cache.get(cache_key)
    logger.info(f"→ Cache hit: {results is not None}")

    similar_samples = []

    spectrum = compound.get_spectrum()
    if spectrum:
        logger.info(f"✔ Got spectrum, peaks count: {len(spectrum.peaks)}")

        # ----- 归一化峰强度 -----
        intensities = spectrum.intensities
        if intensities.max() > 0:
            intensities = intensities / intensities.max()
        spectrum = Spectrum(mz=spectrum.mz, intensities=intensities, metadata=spectrum.metadata)

        # 调用 identify_spectrums 获取实际匹配结果
        raw_results = identify_spectrums([spectrum])
        logger.info(f"✔ identify_spectrums returned {len(raw_results)} results")

        # 1️⃣ 过滤 score > 0.6
        filtered_results = [r for r in raw_results if r.get("score", 0) > 0.6]

        # 2️⃣ 去重：同一 latin_name + tissue + precursor_mz，只保留 score 最大
        best_results = {}
        for r in filtered_results:
            key = (
                r.get("latin_name"),
                r.get("tissue"),
                r.get("tissue"),
                round(r.get("precursor_mz", 0), 4)  # 母离子质荷比
            )
            if key not in best_results or r["score"] > best_results[key]["score"]:
                best_results[key] = r

        # 3️⃣ 按 score 降序排列，保留所有符合条件的谱图
        similar_samples_sorted = sorted(best_results.values(), key=lambda x: x["score"], reverse=True)

        # 4️⃣ 构建前端显示
        similar_samples = [
            {
                "latin_name": format_latin_name(r.get("latin_name")),
                "chinese_name": r.get("chinese_name"),
                "tissue": (r.get("tissue") or "").capitalize(),
                "score": r.get("score"),
                "latin_slug": slugify(r.get("latin_name") or ""),
                "precursor_mz": round(r.get("precursor_mz", 0), 4),
                "spectrum_idx": r.get("spectrum_index"), 
            }
            for r in similar_samples_sorted
        ]

    else:
        logger.warning("⚠ No spectrum found for this compound")


    logger.info(f"✔ Similar_samples count: {len(similar_samples)}")

    # ===== 模板上下文 =====
    context = {
        "compound": compound,
        "mol_img": mol_img,
        "rdkit": rdkit_props,
        "plant_sources": plant_sources,        # 表格 A
        "similar_samples": similar_samples,    # 表格 B
        "debug_info": {                        # 调试信息
            "plant_sources_count": len(plant_sources),
            "similar_samples_count": len(similar_samples),
            "spectrum_exists": bool(spectrum)
        }
    }

    return render(request, "web/compound_detail.html", context)


def make_cache_key(prefix, latin_name, compound):
    raw_key = f"{latin_name}_{compound}"
    key_hash = hashlib.md5(raw_key.encode('utf-8')).hexdigest()
    return f"{prefix}_{key_hash}"

def herb_compound_detail(request, latin_name, compound):
    from urllib.parse import unquote
    from django.utils.text import slugify
    from web.models import CompoundLibrary
    from web.utils.plotting import generate_spectrum_comparison

    import pickle

    def normalize_name(name):
        if not name:
            return ""
        # 把特殊字符 × 去掉
        name = name.replace("×", " ")
        # 标准化多个空格
        return " ".join(name.split()).strip().lower()

    latin_name = unquote(latin_name)
    compound = unquote(compound)
    matched_id = request.GET.get("matched_id")

    # ---- Step 1: 找真实拉丁名（normalize + slugify）----
    real_latin_name = None
    all_names = CompoundLibrary.objects.values_list("latin_name", flat=True).distinct()

    for name in all_names:
        if slugify(normalize_name(name)) == latin_name:
            real_latin_name = name
            break

    # ---- Step 2: 查询 normalize 后完全一致的样品 ----
    norm_real = normalize_name(real_latin_name)

    all_entries = CompoundLibrary.objects.filter(
        matched_spectrum_id=matched_id,
        spectrum_type="sample"
    )

    entries = []
    for e in all_entries:
        if normalize_name(e.latin_name) == norm_real:
            spectrum = pickle.loads(e.spectrum_blob)
            spectrum.metadata["db_id"] = e.id
            spectrum.db_id = e.id
            e.spectrum = spectrum
            entries.append(e)

    # ---- Step 3: 查标品 ----
    standard = CompoundLibrary.objects.filter(
        standard_id=matched_id,
        spectrum_type="standard"
    ).first()

    # ---- Step 4: 图像比对 ----
    comparison_list = []
    matched_ids = []

    if entries and standard:
        std_db = (standard.database or "").lower()
        is_nist = "nist" in std_db

        if is_nist:
            comparison_list = generate_spectrum_comparison(entries, standards=None)
            matched_ids = [e.id for e in entries]
        else:
            comparison_list = generate_spectrum_comparison(entries, standards=[standard])
            matched_ids = [c["sample"].id for c in comparison_list]

    # ---- Debug ----
    print(f"[DEBUG] latin_name={latin_name}, real_latin_name={real_latin_name}, matched_id={matched_id}")
    print(f"[DEBUG] sample count={len(entries)}")
    print(f"[DEBUG] standard found={bool(standard)}")

    return render(request, "web/herb_compound_detail.html", {
        "latin_name": real_latin_name or latin_name,
        "compound": compound,
        "entries": entries,
        "comparison_list": comparison_list,
        "matched_ids": matched_ids,
    })



# 植物谱图pickle路径
HERB_SPECTRA_POS = "/data2/jiangsiyu/ATNP_Database/model_copy/herbs_spectra_pos.pickle"
HERB_SPECTRA_NEG = "/data2/jiangsiyu/ATNP_Database/model_copy/herbs_spectra_neg.pickle"

# 缓存pickle避免重复加载
_herb_spectra_cache = {"pos": None, "neg": None}

def load_herb_spectra(ionmode="positive"):
    global _herb_spectra_cache
    key = "pos" if ionmode == "positive" else "neg"
    if _herb_spectra_cache[key] is None:
        path = HERB_SPECTRA_POS if key == "pos" else HERB_SPECTRA_NEG
        with open(path, "rb") as f:
            _herb_spectra_cache[key] = pickle.load(f)
    return _herb_spectra_cache[key]


from web.utils.identify import load_models_and_indexes
from web.utils.plot_tools import plot_2_spectrum

def similar_compare(request, compound_id, spectrum_idx):
    compound_obj = get_object_or_404(CompoundLibrary, pk=compound_id)

    # === 基准谱图 ===
    ref_spectrum = compound_obj.get_spectrum()
    if ref_spectrum is None:
        return HttpResponse("❌ No spectrum found for this compound", status=404)

    # === 加载模型和谱图库 ===
    load_models_and_indexes()
    ionmode = (compound_obj.ionmode or "positive").lower()
    mode = "pos" if ionmode.startswith("pos") else "neg"

    from web.utils.identify import _refs
    all_spectra = _refs.get(mode, [])

    # === 取植物谱图 ===
    try:
        sample_entry = all_spectra[spectrum_idx]
    except IndexError:
        return HttpResponse("❌ Invalid spectrum index", status=404)
    
    from web.utils.identify import dict_to_spectrum

    # ✅ 兼容结构（dict or Spectrum）
    if isinstance(sample_entry, dict):
        sample_spectrum = sample_entry.get("spectrum", sample_entry)
    else:
        sample_spectrum = sample_entry

    # 如果 spectrum 是字符串或非法结构，也安全转换
    if isinstance(sample_spectrum, str) or not hasattr(sample_spectrum, "peaks"):
        sample_spectrum = dict_to_spectrum(sample_entry)

    # === 生成对比图 ===
    comparison_plot = None
    try:
        from web.utils.plot_tools import plot_2_spectrum
        comparison_plot = plot_2_spectrum(ref_spectrum, sample_spectrum)
    except Exception as e:
        print("⚠ Plotting error:", e)
        return HttpResponse(f"⚠ Error while plotting: {e}", status=500)

    # === 相似度参数 ===
    similarity = request.GET.get("score", "0")
    try:
        similarity = float(similarity)
    except ValueError:
        similarity = 0.0

    # === 提取植物信息 ===
    meta = getattr(sample_spectrum, "metadata", {})
    if not meta and isinstance(sample_entry, dict):
        meta = sample_entry.get("metadata", {})

    sample_info = {
        "chinese_name": meta.get("chinese_name", ""),
        "latin_name": meta.get("latin_name", ""),
        "tissue": meta.get("tissue", ""),
        "similarity": similarity,
    }

    return render(request, "web/similar_compare.html", {
        "compound": compound_obj,
        "sample": sample_info,
        "comparison_plot": comparison_plot,
    })


def structure_query(request):
    return render(request, "web/structure_query.html")


def structure_search(request):
    if request.method == "POST":
        smiles = request.POST.get("smiles", "").strip()

        if not smiles:
            return render(request, "web/structure_query.html", {"error": "No structure provided."})

        mol_query = Chem.MolFromSmiles(smiles)
        if mol_query is None:
            return render(request, "web/structure_query.html", {"error": "Invalid SMILES."})

        # 统一 canonical SMILES
        canonical_smiles = Chem.MolToSmiles(mol_query, canonical=True)
        mol_query = Chem.MolFromSmiles(canonical_smiles)

        # 计算查询指纹
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fp_query = AllChem.GetMorganFingerprintAsBitVect(
                mol_query, radius=2, nBits=2048
            )

        # 只取有指纹的标品谱图 + 有 PLANTS 字段
        qs = CompoundLibrary.objects.filter(
            spectrum_type__iexact="standard"
        ).filter(
            ~Q(plants__isnull=True) & ~Q(plants__regex=r'^\s*$')
        ).exclude(
            morgan_fp__isnull=True
        )

        # ---------- 相似度筛选 ----------
        tmp_hits = []
        for c in qs:
            fp = c.get_fingerprint()
            if fp is None:
                continue

            sim = DataStructs.TanimotoSimilarity(fp_query, fp)
            if sim >= 0.75:
                tmp_hits.append((sim, c))

        # 按相似度排序
        tmp_hits.sort(key=lambda x: x[0], reverse=True)

        # ---------- 去重逻辑（与 compound_list 完全一致） ----------
        rows_dict = defaultdict(lambda: {
            "standard": None,
            "ionmode": None,
            "first_id": None,
            "precursor_mz": None,
            "smiles": None,
            "databases": set(),
            "plants": None,
            "similarity": 0.0,
        })

        for sim, obj in tmp_hits:
            item = {
                "id": obj.id,
                "standard": obj.standard,
                "precursor_mz": obj.precursor_mz,
                "database": obj.database,
                "smiles": obj.smiles,
                "pepmass": obj.pepmass,
                "plants": obj.plants,
                "ionmode": obj.ionmode,
            }

            canon_key = (_canon(item["standard"]), item.get("ionmode"))
            r = rows_dict[canon_key]

            # 只在首次写入
            if r["standard"] is None:
                r["standard"] = item["standard"] or "(unknown)"
                r["ionmode"] = item.get("ionmode")
                r["first_id"] = item["id"]
                r["precursor_mz"] = item["precursor_mz"] or _fallback_mz(item["pepmass"])
                r["smiles"] = item["smiles"]
                r["plants"] = item["plants"]
                r["similarity"] = sim

            # 规范化 database
            if item["database"]:
                normalized_db = _canon(item["database"])
                if normalized_db in {"nist20", "nist"}:
                    r["databases"].add("NIST")
                else:
                    r["databases"].add(normalized_db.upper())

        # ---------- 转换为列表 ----------
        rows = [{
            "standard": r["standard"],
            "ionmode": r["ionmode"],
            "first_id": r["first_id"],
            "precursor_mz": f"{r['precursor_mz']:.4f}" if r["precursor_mz"] else "-",
            "database": ", ".join(sorted(r["databases"])) or "-",
            "smiles": r["smiles"],
            "plants": r["plants"],
            "similarity": f"{r['similarity']:.3f}",
        } for r in rows_dict.values()]

        rows.sort(key=lambda x: _canon(x["standard"]))

        return render(request, "web/structure_results.html", {
            "results": rows,
            "query_smiles": canonical_smiles,
        })

    return render(request, "web/structure_query.html")
