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

    # 只按 standard 去重
    rows_dict = defaultdict(lambda: {
        "standard": None,
        "first_id": None,
        "precursor_mz": None,
        "smiles": None,
        "databases": set(),
        "plants": None,
    })

    for item in raw:
        key = (_canon(item["standard"]))
        r = rows_dict[key]

        if r["standard"] is None:
            r["standard"] = item["standard"] or "(unknown)"
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
        "first_id": r["first_id"],
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

    # ===== 基本信息 =====
    compound = get_object_or_404(CompoundLibrary, pk=pk)
    mol_img = plot_ref_mol(compound.smiles) if compound.smiles else None

    # ===== RDKit 计算性质 =====
    rdkit_props = None
    if compound.smiles:
        mol = Chem.MolFromSmiles(compound.smiles)
        if mol:
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
            try:
                rdkit_props["sa_score"] = round(float(sascorer.calculateScore(mol)), 2)
            except Exception:
                rdkit_props["sa_score"] = "N/A"

    # =====================================================
    # 表格 A：来源植物（从「样品库」反查真实谱图）
    # =====================================================
    plant_sources = []

    if compound.plants:
        entries = re.findall(r"\[(P\d+:[^\]]+)\]", compound.plants)

        for entry in entries:
            chinese_name = re.search(r"Chinese_name=([^;]+)", entry)
            latin_name = re.search(r"Latin_name=([^;]+)", entry)
            tissue = re.search(r"Tissue=([^;]+)", entry)
            matched_id = re.search(r"matched_spectrum_id=([0-9]+)", entry)

            chinese_name = chinese_name.group(1).strip() if chinese_name else "-"
            latin_name = latin_name.group(1).strip() if latin_name else "-"
            tissue = tissue.group(1).strip() if tissue else "-"
            matched_id = matched_id.group(1) if matched_id else None

            # 默认值
            precursor_mz = "-"
            ionmode = "-"

            # 🔥 核心：去「样品库」找谱图（不是标品库）
            if matched_id:
                sample_qs = CompoundLibrary.objects.filter(
                    spectrum_type="sample",
                    matched_spectrum_id=matched_id
                )

                if latin_name != "-":
                    sample_qs = sample_qs.filter(latin_name__iexact=latin_name)
                if tissue != "-":
                    sample_qs = sample_qs.filter(tissue__iexact=tissue)

                # ⚠ 不用 .get()，允许 POS / NEG 各一条
                samples = sample_qs.all()

                for sample in samples:
                    mz = sample.precursor_mz or sample.pepmass
                    if mz:
                        precursor_mz = round(float(mz), 4)

                    ionmode = sample.ionmode or "-"

                    plant_sources.append({
                        "chinese_name": chinese_name,
                        "latin_name": format_latin_name(latin_name),
                        "tissue": tissue.capitalize() if tissue != "-" else "-",
                        "matched_id": matched_id,
                        "latin_slug": slugify(latin_name),
                        "compound_raw": compound.standard,
                        "compound_url": quote(compound.standard, safe=""),
                        "precursor_mz": precursor_mz,
                        "ionmode": ionmode,
                    })

            else:
                # 没 matched_id 的兜底展示
                plant_sources.append({
                    "chinese_name": chinese_name,
                    "latin_name": format_latin_name(latin_name),
                    "tissue": tissue.capitalize() if tissue != "-" else "-",
                    "matched_id": None,
                    "latin_slug": slugify(latin_name),
                    "compound_raw": compound.standard,
                    "compound_url": quote(compound.standard, safe=""),
                    "precursor_mz": "-",
                    "ionmode": "-",
                })

    # ===== 去重：中文名 + 拉丁名 + ionmode =====
    uniq = {}
    for ps in plant_sources:
        key = (ps["chinese_name"], ps["latin_name"], ps["ionmode"])
        uniq[key] = ps
    plant_sources = list(uniq.values())

    logger.info(f"✔ plant_sources final count = {len(plant_sources)}")

    # =====================================================
    # 表格 B：谱图相似来源（你原来的逻辑，未动）
    # =====================================================
    similar_samples = []

    spectrum = compound.get_spectrum()
    if spectrum:
        intensities = spectrum.intensities
        if intensities.max() > 0:
            intensities = intensities / intensities.max()
        spectrum = Spectrum(
            mz=spectrum.mz,
            intensities=intensities,
            metadata=spectrum.metadata
        )

        raw_results = identify_spectrums([spectrum])
        filtered = [r for r in raw_results if r.get("score", 0) > 0.6]

        best = {}
        for r in filtered:
            key = (
                r.get("latin_name"),
                r.get("tissue"),
                round(r.get("precursor_mz", 0), 4),
                r.get("ionmode")
            )
            if key not in best or r["score"] > best[key]["score"]:
                best[key] = r

        similar_samples = [
            {
                "latin_name": format_latin_name(r.get("latin_name")),
                "chinese_name": r.get("chinese_name"),
                "tissue": (r.get("tissue") or "").capitalize(),
                "score": r.get("score"),
                "latin_slug": slugify(r.get("latin_name") or ""),
                "precursor_mz": round(r.get("precursor_mz", 0), 4),
                "ionmode": r.get("ionmode", "-"),
                "spectrum_idx": r.get("spectrum_index"),
            }
            for r in sorted(best.values(), key=lambda x: x["score"], reverse=True)
        ]

    # ===== 渲染 =====
    return render(request, "web/compound_detail.html", {
        "compound": compound,
        "mol_img": mol_img,
        "rdkit": rdkit_props,
        "plant_sources": plant_sources,      # 表格 A（双离子模式）
        "similar_samples": similar_samples,  # 表格 B
    })


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
    print("[DEBUG] candidates len =", len([standard] if standard else []))

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

from web.utils.compound_aggregate import aggregate_by_inchikey, smiles_to_inchikey

def structure_search(request):
    if request.method == "POST":
        smiles = request.POST.get("smiles", "").strip()

        if not smiles:
            return render(
                request,
                "web/structure_query.html",
                {"error": "No structure provided."}
            )

        # =========================================================
        # 1️⃣ 解析并规范化查询结构
        # =========================================================
        mol_query = Chem.MolFromSmiles(smiles)
        if mol_query is None:
            return render(
                request,
                "web/structure_query.html",
                {"error": "Invalid SMILES."}
            )

        canonical_smiles = Chem.MolToSmiles(mol_query, canonical=True)
        mol_query = Chem.MolFromSmiles(canonical_smiles)

        # 查询 InChIKey（用于精确匹配）
        try:
            query_inchikey = Chem.inchi.MolToInchiKey(mol_query)
        except Exception:
            query_inchikey = None

        # =========================================================
        # 2️⃣ 计算查询指纹（用于相似度搜索）
        # =========================================================
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fp_query = AllChem.GetMorganFingerprintAsBitVect(
                mol_query,
                radius=2,
                nBits=2048
            )

        # =========================================================
        # 3️⃣ 查询候选标品
        # =========================================================
        qs = CompoundLibrary.objects.filter(
            spectrum_type__iexact="standard"
        ).filter(
            ~Q(plants__isnull=True) & ~Q(plants__regex=r'^\s*$')
        ).exclude(
            smiles__isnull=True
        ).exclude(
            smiles=""
        )

        # =========================================================
        # 4️⃣ InChIKey 精确匹配（最高优先级）
        # =========================================================
        exact_hits = []
        if query_inchikey:
            for obj in qs:
                try:
                    if smiles_to_inchikey(obj.smiles) == query_inchikey:
                        exact_hits.append(obj)
                except Exception:
                    continue

        if exact_hits:
            results = aggregate_by_inchikey(exact_hits)
            for r in results:
                r["similarity"] = "1.000"

            return render(
                request,
                "web/structure_results.html",
                {
                    "results": results,
                    "query_smiles": canonical_smiles,
                }
            )

        # =========================================================
        # 5️⃣ 指纹相似度搜索（兜底方案）
        # =========================================================
        SIM_THRESHOLD = 0.3   # ✅ 推荐：0.3–0.5，0.75 太严格

        matched_objects = []
        similarity_map = {}   # obj.id → max similarity

        for obj in qs:
            fp = obj.get_fingerprint()
            if fp is None:
                continue

            sim = DataStructs.TanimotoSimilarity(fp_query, fp)
            if sim >= SIM_THRESHOLD:
                matched_objects.append(obj)
                similarity_map[obj.id] = max(
                    similarity_map.get(obj.id, 0),
                    sim
                )

        if not matched_objects:
            return render(
                request,
                "web/structure_results.html",
                {
                    "results": [],
                    "query_smiles": canonical_smiles,
                }
            )

        # =========================================================
        # 6️⃣ InChIKey 统一聚合 + 相似度排序
        # =========================================================
        results = aggregate_by_inchikey(matched_objects)

        for r in results:
            r["similarity"] = f"{similarity_map.get(r['first_id'], 0):.3f}"

        results.sort(
            key=lambda x: float(x.get("similarity", 0)),
            reverse=True
        )

        return render(
            request,
            "web/structure_results.html",
            {
                "results": results,
                "query_smiles": canonical_smiles,
            }
        )

    return render(request, "web/structure_query.html")

def molecular_weight_query(request):
    """
    仅显示 MW 搜索表单，不显示任何查询结果
    """
    return render(request, 'web/molecular_weight_query.html')

from rdkit import Chem
from rdkit.Chem import inchi


from collections import defaultdict
from django.db.models import Q
from django.db.models.functions import Lower


def molecular_weight_search(request):
    if request.method == "POST":
        min_mw = request.POST.get("min_mw", "").strip()
        max_mw = request.POST.get("max_mw", "").strip()

        try:
            min_mw = float(min_mw) if min_mw else None
            max_mw = float(max_mw) if max_mw else None
        except ValueError:
            return render(request, "web/molecular_weight_query.html", {
                "error": "Invalid number input."
            })

        qs = CompoundLibrary.objects.filter(
            spectrum_type__iexact="standard"
        ).filter(
            ~Q(plants__isnull=True) & ~Q(plants__regex=r'^\s*$')
        ).exclude(
            smiles__isnull=True
        ).exclude(
            smiles=""
        )

        if min_mw is not None:
            qs = qs.filter(precursor_mz__gte=min_mw)
        if max_mw is not None:
            qs = qs.filter(precursor_mz__lte=max_mw)

        results = aggregate_by_inchikey(qs)

        return render(request, "web/molecular_weight_results.html", {
            "results": results,
            "min_mw": min_mw,
            "max_mw": max_mw,
        })

    return render(request, "web/molecular_weight_query.html")


def msms_search(request):
    """显示搜索页面"""
    return render(request, 'web/msms_search.html')


from django.shortcuts import render
from django.db.models import Q
import numpy as np
from matchms import Spectrum
from matchms.filtering import (
    normalize_intensities,
    select_by_mz,
    require_minimum_number_of_peaks
)

def msms_result(request):
    results = []
    error = None

    if not request.GET:
        return render(request, "web/msms_result.html", {
            "results": [],
            "error": None
        })

    # ======================================================
    # 1️⃣ 解析输入 MS/MS
    # ======================================================
    try:
        msms_input = request.GET.get("msms_spectrum", "").strip()
        parent_mz  = request.GET.get("parent_mz")
        ion_mode   = request.GET.get("ion_mode", "").lower().strip()

        peaks = []
        for line in msms_input.splitlines():
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            mz, intensity = map(float, parts)
            peaks.append((mz, intensity))

        if len(peaks) < 3:
            raise ValueError("At least 3 valid MS/MS peaks required.")

        mzs = np.array([p[0] for p in peaks], dtype=float)
        intensities = np.array([p[1] for p in peaks], dtype=float)

        metadata = {"ionmode": ion_mode}
        if parent_mz:
            metadata["precursor_mz"] = float(parent_mz)

        spectrum = Spectrum(mz=mzs, intensities=intensities, metadata=metadata)

        spectrum = normalize_intensities(spectrum)
        spectrum = select_by_mz(spectrum, mz_from=50, mz_to=2000)
        spectrum = require_minimum_number_of_peaks(spectrum, n_required=3)

        if spectrum is None:
            raise ValueError("Spectrum discarded after preprocessing.")

    except Exception as e:
        return render(request, "web/msms_result.html", {
            "error": f"MS/MS spectrum parsing error: {e}"
        })

    # ======================================================
    # 2️⃣ 谱图搜索（植物样本库）
    # ======================================================
    raw_results = identify_spectrums([spectrum])
    filtered = [r for r in raw_results if r.get("score", 0) >= 0.6]

    if not filtered:
        return render(request, "web/msms_result.html", {
            "results": [],
            "error": "No matched plant spectra found."
        })

    # ======================================================
    # 3️⃣ ORM 顺序必须和建模一致（sample）
    # ======================================================
    sample_qs = list(
        CompoundLibrary.objects.filter(
            spectrum_type="sample"
        ).order_by("id")
    )

    if not sample_qs:
        return render(request, "web/msms_result.html", {
            "results": [],
            "error": "Plant spectrum database is empty."
        })

    # ======================================================
    # 4️⃣ 聚合为「植物」维度
    # ======================================================
    herb_map = {}
    for r in filtered:
        idx = r.get("spectrum_index")
        score = r.get("score", 0)

        if idx is None or idx < 0 or idx >= len(sample_qs):
            continue

        obj = sample_qs[idx]
        key = (obj.latin_name, obj.chinese_name, obj.tissue)

        if key not in herb_map:
            herb_map[key] = {
                "latin_name": obj.latin_name,
                "chinese_name": obj.chinese_name,
                "tissue": obj.tissue,
                "best_score": score
            }
        else:
            herb_map[key]["best_score"] = max(
                herb_map[key]["best_score"], score
            )

    results = sorted(
        herb_map.values(),
        key=lambda x: x["best_score"],
        reverse=True
    )

    for r in results:
        r["best_score"] = f"{r['best_score']:.4f}"

    return render(request, "web/msms_result.html", {
        "results": results,
        "error": None
    })
