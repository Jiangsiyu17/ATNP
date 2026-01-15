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
from web.utils.plot_tools import plot_ref_mol, plot_single_spectrum
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
    query = request.GET.get("query", "").strip()

    # 1️⃣ 只取【标品库 + 有 plants 的记录】
    qs = (
        CompoundLibrary.objects
        .filter(spectrum_type="standard")
        .exclude(plants__isnull=True)
        .exclude(plants=[])
        .values("plants")
    )

    # 2️⃣ 聚合：latin_name → tissues
    herb_map = {}  # latin_name -> {"chinese_name": ..., "tissues": set()}

    for row in qs:
        for p in row["plants"]:
            latin = p.get("latin_name")
            chinese = p.get("chinese_name")
            tissue = p.get("tissue")

            if not latin:
                continue

            # 搜索（只在植物名层面）
            if query and query.lower() not in latin.lower():
                continue

            if latin not in herb_map:
                herb_map[latin] = {
                    "latin_name": latin,
                    "chinese_name": chinese,
                    "tissues": set(),
                }

            if tissue:
                herb_map[latin]["tissues"].add(tissue)

    # 3️⃣ 转为模板可用列表
    rows = []
    for herb in herb_map.values():
        rows.append({
            "latin_name": herb["latin_name"],
            "latin_lower": herb["latin_name"].lower(),
            "chinese_name": herb["chinese_name"] or "-",
            "tissue": ", ".join(sorted(herb["tissues"])) or "-",
        })

    # 4️⃣ 排序
    rows.sort(key=lambda x: x["latin_lower"])

    # 5️⃣ 分页
    paginator = Paginator(rows, 20)
    page_obj = paginator.get_page(request.GET.get("page"))

    return render(request, "web/herb_list.html", {
        "page_obj": page_obj,
        "herbs": page_obj.object_list,
        "query": query,
        "no_result": query and paginator.count == 0,
    })


def herb_detail(request, latin_name):
    """
    展示所有 plants 中包含该植物的化合物
    """
    qs = (
        CompoundLibrary.objects
        .filter(spectrum_type="standard")
        .exclude(plants__isnull=True)
        .exclude(plants=[])
        .order_by(Lower("standard"))
    )

    rows = []

    for c in qs:
        matched = False
        for p in c.plants:
            if p.get("latin_name", "").lower() == latin_name.lower():
                matched = True
                break

        if not matched:
            continue

        # ===== precursor_mz =====
        if c.precursor_mz:
            precursor = f"{c.precursor_mz:.4f}"
        elif c.pepmass:
            try:
                mz_value = float(c.pepmass.split()[0])
                precursor = f"{mz_value:.4f}"
            except Exception:
                precursor = c.pepmass
        else:
            precursor = "-"

        # ===== database =====
        if c.database:
            db = c.database.lower().replace("nist20", "nist")
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

    # ✅ 分页：每页 20 条
    paginator = Paginator(rows, 20)
    page_obj = paginator.get_page(request.GET.get("page"))

    return render(request, "web/herb_detail.html", {
        "latin_name": latin_name,
        "compounds": page_obj.object_list,
        "page_obj": page_obj,
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

from django.core.paginator import Paginator

def compound_detail(request, pk):
    logger = logging.getLogger(__name__)
    logger.info(f"→ Enter compound_detail, pk={pk}")

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

    # ===== 表格 A：plant_sources =====
    plant_sources = []
    plants = compound.plants or []
    if isinstance(plants, dict):
        plants = list(plants.values())
    for p in plants:
        pid = p.get("pid")
        chinese_name = p.get("chinese_name", "-")
        latin_name = p.get("latin_name", "-")
        tissue = p.get("tissue", "-")
        matched_id = p.get("matched_spectrum_id")
        raw_latin = latin_name if latin_name not in ("", "-", None) else None
        latin_slug = slugify(raw_latin) if raw_latin else "unknown-plant"
        precursor_mz = p.get("precursor_mz", "-")
        ionmode = p.get("ionmode", "-")
        plant_sources.append({
            "pid": pid,
            "chinese_name": chinese_name,
            "latin_name": format_latin_name(latin_name),
            "tissue": tissue.capitalize() if tissue != "-" else "-",
            "matched_id": matched_id,
            "latin_slug": latin_slug,
            "precursor_mz": round(float(precursor_mz), 4) if precursor_mz not in ("-", None) else "-",
            "ionmode": ionmode,
        })

    # 去重
    uniq = {}
    for ps in plant_sources:
        key = (ps["chinese_name"], ps["latin_name"], ps["ionmode"])
        uniq[key] = ps
    plant_sources = list(uniq.values())
    for ps in plant_sources:
        if not ps.get("latin_slug"):
            ps["latin_slug"] = "unknown-plant"


    # ===== 表格 B：similar_samples =====
    similar_samples = []
    spectrum = compound.get_spectrum()
    if spectrum:
        intensities = spectrum.intensities
        if intensities.max() > 0:
            intensities = intensities / intensities.max()
        spectrum = Spectrum(mz=spectrum.mz, intensities=intensities, metadata=spectrum.metadata)

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

    # ===== 分页处理 =====
    plant_page = Paginator(plant_sources, 10).get_page(request.GET.get("plant_page"))
    sample_page = Paginator(similar_samples, 10).get_page(request.GET.get("sample_page"))

    return render(request, "web/compound_detail.html", {
        "compound": compound,
        "mol_img": mol_img,
        "rdkit": rdkit_props,
        "plant_sources": plant_page,      # 已分页
        "similar_samples": sample_page,   # 已分页
    })


def make_cache_key(prefix, latin_name, compound):
    raw_key = f"{latin_name}_{compound}"
    key_hash = hashlib.md5(raw_key.encode('utf-8')).hexdigest()
    return f"{prefix}_{key_hash}"


from django.shortcuts import render, get_object_or_404
from django.http import Http404
from matchms import Spectrum
from web.utils.plotting import plot_2_spectrum, format_latin_name
from django.utils.text import slugify
from urllib.parse import unquote
import numpy as np

def herb_compound_detail(request, latin_name, compound_id):
    """
    只展示【某一个植物(pid)】与【当前化合物】的谱图对比
    """

    # Step 0：获取 pid
    pid = request.GET.get("pid")
    if not pid:
        raise Http404("pid is required")

    # Step 1：获取化合物对象
    compound_obj = get_object_or_404(
        CompoundLibrary,
        pk=compound_id
    )

    # ✅ 从对象里取化合物名称
    compound_name = compound_obj.standard

    # ------------------------------------------------
    # Step 2：定位对应 pid 的植物
    # ------------------------------------------------
    plants = compound_obj.plants or []
    if isinstance(plants, dict):
        plants = list(plants.values())

    plant = next((p for p in plants if p.get("pid") == pid), None)
    if not plant:
        raise Http404(f"Plant with pid={pid} not found")

    # ------------------------------------------------
    # Step 3：构建「样品谱图」Spectrum（来自 plants）
    # ------------------------------------------------
    peaks = plant.get("peaks") or []
    if not peaks:
        raise Http404("Plant spectrum peaks empty")

    mzs = np.array([m for m, i in peaks], dtype=float)
    intensities = np.array([i for m, i in peaks], dtype=float)
    if intensities.max() > 0:
        intensities = intensities / intensities.max()

    sample_spec = Spectrum(
        mz=mzs,
        intensities=intensities,
        metadata={
            "ionmode": plant.get("ionmode"),
            "precursor_mz": plant.get("precursor_mz"),
            "pid": pid,
            "latin_name": plant.get("latin_name"),
            "chinese_name": plant.get("chinese_name"),
            "tissue": plant.get("tissue"),
        }
    )

    # ------------------------------------------------
    # Step 4 & 5：根据 database 判断是否画标准品谱图
    # ------------------------------------------------
    dbs = (compound_obj.database or "").lower().split()
    nist_like = {"nist", "nist20"}
    is_nist_only = all(db in nist_like for db in dbs)

    try:
        if is_nist_only:
            # 只画植物谱图，不画标准品
            img_base64 = plot_single_spectrum(sample_spec)
        else:
            # 植物 + 标准品谱图对比
            standard_spec = compound_obj.get_spectrum()
            if standard_spec is None:
                raise Http404("Standard spectrum not found")
            img_base64 = plot_2_spectrum(sample_spec, standard_spec)
    except Exception as e:
        raise RuntimeError(f"Spectrum plotting failed: {e}")


    # ------------------------------------------------
    # Step 6：组织模板需要的数据结构
    # ------------------------------------------------
    entry = {
        "id": pid,
        "chinese_name": plant.get("chinese_name", "-"),
        "latin_name": format_latin_name(plant.get("latin_name", "-")),
        "tissue": plant.get("tissue", "-"),
        "score": plant.get("score", 0.0),
    }

    comparison_list = [{"sample": entry, "image": img_base64}]
    matched_ids = [pid]

    # ------------------------------------------------
    # Step 7：渲染模板
    # ------------------------------------------------
    return render(request, "web/herb_compound_detail.html", {
        "compound": compound_name,  # ✅ 显示化合物名称
        "latin_name": format_latin_name(plant.get("latin_name", "")),
        "entries": [entry],
        "comparison_list": comparison_list,
        "matched_ids": matched_ids,
    })


# 植物谱图pickle路径
HERB_SPECTRA_POS = "/data2/jiangsiyu/ATNP_Database/model/herbs_spectra_pos_1.pickle"
HERB_SPECTRA_NEG = "/data2/jiangsiyu/ATNP_Database/model/herbs_spectra_neg_1.pickle"

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

from django.shortcuts import get_object_or_404, render
from django.http import HttpResponse
from web.models import CompoundLibrary

def similar_compare(request, compound_id, spectrum_idx):
    compound_obj = get_object_or_404(CompoundLibrary, pk=compound_id)

    # === 基准谱图 ===
    ref_spectrum = compound_obj.get_spectrum()
    if ref_spectrum is None:
        return HttpResponse("❌ No spectrum found for this compound", status=404)

    # === 离子模式 ===
    ionmode = (compound_obj.ionmode or "positive").lower()
    mode = "pos" if ionmode.startswith("pos") else "neg"

    from web.utils import identify

    # ✅ 正确触发 lazy load
    all_spectra = identify.get_refs(mode)

    if not all_spectra:
        return HttpResponse("❌ No reference spectra loaded", status=500)

    # === 取植物谱图 ===
    try:
        sample_entry = all_spectra[spectrum_idx]
    except IndexError:
        return HttpResponse("❌ Invalid spectrum index", status=404)

    # === 转 Spectrum（兼容 dict / Spectrum）===
    if isinstance(sample_entry, dict):
        sample_spectrum = sample_entry.get("spectrum")
    else:
        sample_spectrum = sample_entry

    if not hasattr(sample_spectrum, "peaks"):
        sample_spectrum = identify.dict_to_spectrum(sample_entry)

    # === 生成对比图 ===
    try:
        from web.utils.plot_tools import plot_2_spectrum
        comparison_plot = plot_2_spectrum(ref_spectrum, sample_spectrum)
    except Exception as e:
        logger.exception("Plotting error")
        return HttpResponse(f"⚠ Error while plotting: {e}", status=500)

    # === 相似度 ===
    try:
        similarity = float(request.GET.get("score", 0))
    except ValueError:
        similarity = 0.0

    # === 元信息 ===
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

        # ======================================================
        # 1️⃣ 先取候选标准品（不在 ORM 里算分子量）
        # ======================================================
        qs = CompoundLibrary.objects.filter(
            spectrum_type__iexact="standard"
        ).exclude(
            smiles__isnull=True
        ).exclude(
            smiles=""
        )

        matched_ids = []
        mw_cache = {}   # id -> molecular weight（可选，用于调试或展示）

        # ======================================================
        # 2️⃣ Python 层用 RDKit 计算分子量并筛选
        # ======================================================
        for obj in qs.iterator():
            try:
                mol = Chem.MolFromSmiles(obj.smiles)
                if mol is None:
                    continue

                mw = Descriptors.ExactMolWt(mol)
                mw_cache[obj.id] = mw

                if min_mw is not None and mw < min_mw:
                    continue
                if max_mw is not None and mw > max_mw:
                    continue

                matched_ids.append(obj.id)

            except Exception:
                continue

        if not matched_ids:
            return render(request, "web/molecular_weight_results.html", {
                "results": [],
                "min_mw": min_mw,
                "max_mw": max_mw,
                "error": "No compounds found in the given molecular weight range."
            })

        # ======================================================
        # 3️⃣ 再用 ORM 取回 + 按 InChIKey 聚合
        # ======================================================
        final_qs = CompoundLibrary.objects.filter(id__in=matched_ids)
        results = aggregate_by_inchikey(final_qs)

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

        spectrum = Spectrum(
            mz=mzs,
            intensities=intensities,
            metadata=metadata
        )

        spectrum = normalize_intensities(spectrum)
        spectrum = select_by_mz(spectrum, mz_from=50, mz_to=2000)
        spectrum = require_minimum_number_of_peaks(
            spectrum, n_required=3
        )

        if spectrum is None:
            raise ValueError("Spectrum discarded after preprocessing.")

    except Exception as e:
        return render(request, "web/msms_result.html", {
            "results": [],
            "error": f"MS/MS spectrum parsing error: {e}"
        })

    # ======================================================
    # 2️⃣ 谱图搜索（标准品库）
    # ======================================================
    raw_results = identify_spectrums([spectrum])
    filtered = [
        r for r in raw_results
        if r.get("score", 0) >= 0.6
    ]

    if not filtered:
        return render(request, "web/msms_result.html", {
            "results": [],
            "error": "No matched standard spectra found."
        })

    # ======================================================
    # 3️⃣ ORM 顺序必须与建模一致（standard）
    # ======================================================
    standard_qs = list(
        CompoundLibrary.objects.filter(
            spectrum_type__iexact="standard"
        ).order_by("id")
    )

    if not standard_qs:
        return render(request, "web/msms_result.html", {
            "results": [],
            "error": "Standard spectrum database is empty."
        })

    # ======================================================
    # 4️⃣ 按化合物聚合（类似 molecular_weight_search）
    # ======================================================
    compound_map = {}

    for r in filtered:
        idx = r.get("spectrum_index")
        score = r.get("score", 0)

        if idx is None or idx < 0 or idx >= len(standard_qs):
            continue

        obj = standard_qs[idx]

        # 优先用 InChIKey 聚合
        key = obj.inchikey or f"{obj.smiles}_{obj.precursor_mz}"

        if key not in compound_map:
            compound_map[key] = {
                "id": obj.id,
                "compound_name": obj.compound_name,
                "smiles": obj.smiles,
                "inchikey": obj.inchikey,
                "precursor_mz": obj.precursor_mz,
                "ionmode": obj.ionmode,
                "database": obj.database,
                "best_score": score,
            }
        else:
            compound_map[key]["best_score"] = max(
                compound_map[key]["best_score"],
                score
            )

    results = sorted(
        compound_map.values(),
        key=lambda x: x["best_score"],
        reverse=True
    )

    for r in results:
        r["best_score"] = f"{r['best_score']:.4f}"
        if r["precursor_mz"]:
            r["precursor_mz"] = f"{r['precursor_mz']:.4f}"

    return render(request, "web/msms_result.html", {
        "results": results,
        "error": None
    })

