from django.shortcuts import render, get_object_or_404, redirect
from django.core.cache import cache
from .models import CompoundLibrary
from django.db.models.functions import Lower
from django.db.models import Q, Min
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
from django.http import JsonResponse


def compound_list(request):
    return render(request, "web/compound_list.html")

def compound_list_api(request):
    draw = int(request.GET.get("draw", 1))
    start = int(request.GET.get("start", 0))
    length = int(request.GET.get("length", 20))

    # ===== 列搜索 =====
    search_standard = request.GET.get("columns[1][search][value]", "").strip()
    search_database = request.GET.get("columns[2][search][value]", "").strip()
    search_smiles = request.GET.get("columns[3][search][value]", "").strip()

    # ===== 基础 queryset（不带搜索）=====
    base_qs = (
        CompoundLibrary.objects
        .filter(spectrum_type__iexact="standard")
        .exclude(plants__isnull=True)
        .exclude(plants=[])
    )

    # ===== recordsTotal（⚠️ 不加搜索）=====
    records_total = (
        base_qs
        .values("standard")
        .distinct()
        .count()
    )

    # ===== 加搜索条件 =====
    qs = base_qs
    if search_standard:
        qs = qs.filter(standard__icontains=search_standard)
    if search_smiles:
        qs = qs.filter(smiles__icontains=search_smiles)
    if search_database:
        qs = qs.filter(database__icontains=search_database)

    # ===== recordsFiltered =====
    records_filtered = (
        qs
        .values("standard")
        .distinct()
        .count()
    )

    # ===== 分页 + 去重 =====
    qs = (
        qs.values("standard")
        .annotate(
            first_id=Min("id"),
            smiles=Min("smiles"),
        )
        .order_by(Lower("standard"))[start:start + length]
    )

    # ===== database 合并 =====
    standards = [r["standard"] for r in qs]
    db_map = defaultdict(set)

    db_qs = (
        CompoundLibrary.objects
        .filter(standard__in=standards)
        .values("standard", "database")
    )

    for r in db_qs:
        if r["database"]:
            db = r["database"].lower()
            if db in {"nist", "nist20"}:
                db_map[r["standard"]].add("NIST")
            else:
                db_map[r["standard"]].add(db.upper())

    # ===== 返回数据 =====
    data = []
    for i, row in enumerate(qs, start=start + 1):
        std = row["standard"]
        data.append({
            "index": i,
            "standard": std or "(unknown)",
            "database": ", ".join(sorted(db_map.get(std, []))) or "-",
            "smiles": row["smiles"] or "-",
            "action": (
                f'<a class="btn btn-sm btn-outline-primary" '
                f'href="/compound/{row["first_id"]}/">View</a>'
            )
        })

    return JsonResponse({
        "draw": draw,
        "recordsTotal": records_total,
        "recordsFiltered": records_filtered,
        "data": data
    })

def plant_list(request):
    return render(request, "web/plant_list.html")

def plant_list_api(request):
    draw = int(request.GET.get("draw", 1))
    start = int(request.GET.get("start", 0))
    length = int(request.GET.get("length", 20))

    # ===== 列搜索参数 =====
    search_latin = request.GET.get("columns[1][search][value]", "").strip().lower()
    search_chinese = request.GET.get("columns[2][search][value]", "").strip().lower()
    search_tissue = request.GET.get("columns[3][search][value]", "").strip().lower()

    # ===== 排序 =====
    order_col = int(request.GET.get("order[0][column]", 1))
    order_dir = request.GET.get("order[0][dir]", "asc")

    columns = ["index", "latin_name", "chinese_name", "tissue"]
    order_field = columns[order_col] if order_col < len(columns) else "latin_name"
    reverse = (order_dir == "desc")

    # ===== 1️⃣ 聚合 plants（来自 CompoundLibrary）=====
    qs = (
        CompoundLibrary.objects
        .filter(spectrum_type="standard")
        .exclude(plants__isnull=True)
        .exclude(plants=[])
        .values("plants")
    )

    plant_map = {}

    for row in qs:
        for p in row["plants"]:
            latin = p.get("latin_name")
            chinese = p.get("chinese_name")
            tissue = p.get("tissue")

            if not latin:
                continue

            key = latin.lower()

            if key not in plant_map:
                plant_map[key] = {
                    "latin_name": latin,
                    "chinese_name": chinese or "-",
                    "tissues": set(),
                }

            if tissue:
                plant_map[key]["tissues"].add(tissue)

    # ===== recordsTotal（不加搜索）=====
    records_total = len(plant_map)

    # ===== 2️⃣ 构建行数据 + 列搜索 =====
    rows = []
    for h in plant_map.values():
        tissue_str = ", ".join(
            t.strip().capitalize()
            for t in sorted(h["tissues"])
        ) or "-"

        row = {
            "latin_name": h["latin_name"],
            "chinese_name": h["chinese_name"],
            "tissue": tissue_str,
        }

        # --- 列搜索（AND 逻辑） ---
        if search_latin and search_latin not in row["latin_name"].lower():
            continue
        if search_chinese and search_chinese not in row["chinese_name"].lower():
            continue
        if search_tissue and search_tissue not in row["tissue"].lower():
            continue

        rows.append(row)

    # ===== recordsFiltered =====
    records_filtered = len(rows)

    # ===== 3️⃣ 排序 =====
    if order_field != "index":
        rows.sort(
            key=lambda x: x[order_field].lower(),
            reverse=reverse
        )

    # ===== 4️⃣ 分页 =====
    page_rows = rows[start:start + length]

    # ===== 5️⃣ DataTables 数据 =====
    data = []
    for i, row in enumerate(page_rows, start=start + 1):
        data.append({
            "index": i,
            "latin_name": row["latin_name"],
            "chinese_name": row["chinese_name"],
            "tissue": row["tissue"],
            "action": (
                f'<a class="btn btn-sm btn-outline-primary" '
                f'href="/plant/{row["latin_name"]}/">View</a>'
            )
        })

    return JsonResponse({
        "draw": draw,
        "recordsTotal": records_total,
        "recordsFiltered": records_filtered,
        "data": data
    })

def plant_detail(request, latin_name):
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

    return render(request, "web/plant_detail.html", {
        "latin_name": latin_name,
        "compounds": page_obj.object_list,
        "page_obj": page_obj,
    })


def home(request):
    return render(request, 'web/home.html')

def get_full_compound_for_detail(queryset):
    """
    从 queryset 里选出一个 compound 对象，
    并保证 plants 字段处理成列表、去重，和列表页点击一致。
    """
    for c in queryset:
        plants = c.plants or []
        if isinstance(plants, dict):
            plants = list(plants.values())

        # 去重
        uniq = {}
        for p in plants:
            key = (p.get("chinese_name"), p.get("latin_name"), p.get("ionmode"))
            uniq[key] = p
        c.plants = list(uniq.values())
        return c  # 返回处理后的对象
    return None

def search(request):
    query = request.GET.get("q", "").strip()
    if not query:
        return render(request, "search_not_found.html", {"query": query})

    # ======================================================
    # 1️⃣ 搜化合物（standard / title / smiles）
    # ======================================================
    qs = CompoundLibrary.objects.filter(
        spectrum_type__iexact="standard"
    ).filter(
        Q(standard__icontains=query) |
        Q(title__icontains=query) |
        Q(smiles__icontains=query)
    )

    if qs.exists():
        compound = get_full_compound_for_detail(qs)
        if compound:
            return redirect(reverse("compound_detail", args=[compound.pk]))

    # ======================================================
    # 2️⃣ 搜植物（从 plants JSON 中找）
    # ======================================================
    qs = CompoundLibrary.objects.filter(
        spectrum_type__iexact="standard"
    ).exclude(plants__isnull=True).exclude(plants=[])

    for obj in qs:
        plants = obj.plants or []
        if isinstance(plants, dict):
            plants = list(plants.values())
        for p in plants:
            latin = p.get("latin_name", "")
            chinese = p.get("chinese_name", "")
            if (latin and query.lower() in latin.lower()) or (chinese and query in chinese):
                return redirect(reverse("plant_detail", args=[latin]))

    # ======================================================
    # 3️⃣ 都没找到
    # ======================================================
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
from web.utils.similar_cache import get_similar_samples

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


    # ===== 表格 B：similar_samples（预计算版）=====
    raw_similar = get_similar_samples(
        compound.id,
        ionmode=compound.ionmode or "positive"
    )

    logger.warning(
        f"[DEBUG] compound_id={compound.id}, similar_count={len(raw_similar)}"
    )

    # 去重（同植物 + 组织 + 离子模式，只保留最高分）
    best = {}
    for r in raw_similar:
        key = (
            r.get("latin_name"),
            r.get("tissue"),
            r.get("ionmode"),
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

    # ✅ 如果 similar_samples 不为空才分页，否则直接 None
    sample_page = Paginator(similar_samples, 10).get_page(request.GET.get("sample_page")) if similar_samples else None

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

def plant_compound_detail(request, latin_name, compound_id):
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
    return render(request, "web/plant_compound_detail.html", {
        "compound": compound_name,  # ✅ 显示化合物名称
        "latin_name": format_latin_name(plant.get("latin_name", "")),
        "entries": [entry],
        "comparison_list": comparison_list,
        "matched_ids": matched_ids,
    })



from django.shortcuts import get_object_or_404, render
from django.http import HttpResponse
from web.models import CompoundLibrary

def similar_compare(request, compound_id, spectrum_idx):
    compound_obj = get_object_or_404(CompoundLibrary, pk=compound_id)

    # === 离子模式 ===
    ionmode = (compound_obj.ionmode or "positive").lower()
    mode = "pos" if ionmode.startswith("pos") else "neg"

    from web.utils import identify

    # ✅ lazy load 植物谱图
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

    # ------------------------------------------------
    # ✅ 判断是否 NIST-only
    # ------------------------------------------------
    dbs = (compound_obj.database or "").lower().split()
    nist_like = {"nist", "nist20"}
    is_nist_only = all(db in nist_like for db in dbs)

    # ------------------------------------------------
    # === 生成谱图 ===
    # ------------------------------------------------
    try:
        from web.utils.plot_tools import plot_2_spectrum, plot_single_spectrum

        if is_nist_only:
            # ✅ NIST：只画植物谱图
            comparison_plot = plot_single_spectrum(
                sample_spectrum,
                # title="Plant sample spectrum"
            )
        else:
            # ✅ 非 NIST：植物 vs 化合物
            ref_spectrum = compound_obj.get_spectrum()
            if ref_spectrum is None:
                return HttpResponse("❌ No spectrum found for this compound", status=404)

            comparison_plot = plot_2_spectrum(
                ref_spectrum,
                sample_spectrum
            )

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
        "is_nist_only": is_nist_only,  # 👈 可选：模板里用
    })

def structure_query(request):
    return render(request, "web/structure_query.html")

from web.utils.compound_aggregate import aggregate_by_inchikey, smiles_to_inchikey, normalize_mol


def structure_search(request):

    smiles = request.POST.get("smiles", "").strip()

    debug = {
        "input_smiles": smiles,
        "valid_query": False,
        "canonical_smiles": None,
        "total_standard": 0,
        "valid_smiles": 0,
        "exact_hits": 0,
        "max_similarity": 0.0,
        "threshold": 0.3,
    }

    if not smiles:
        return render(
            request,
            "web/structure_query.html",
            {"error": "No structure provided."}
        )

    # =========================================================
    # 1️⃣ 解析查询结构（⚠️ 不 normalize）
    # =========================================================
    mol_query = Chem.MolFromSmiles(smiles)
    if mol_query is None:
        return render(
            request,
            "web/structure_query.html",
            {"error": "Invalid SMILES."}
        )

    mol_query = normalize_mol(mol_query)  # ⭐ 关键
    canonical_smiles = Chem.MolToSmiles(mol_query, canonical=True)

    debug["valid_query"] = True
    debug["canonical_smiles"] = canonical_smiles

    # 查询指纹（只算一次）
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fp_query = AllChem.GetMorganFingerprintAsBitVect(
            mol_query, radius=2, nBits=2048
        )

    # 查询 InChIKey（只用于 exact match）
    try:
        query_inchikey = Chem.inchi.MolToInchiKey(mol_query)
    except Exception:
        query_inchikey = None

    # =========================================================
    # 2️⃣ 构建候选集（只取 standard）
    # =========================================================
    qs = CompoundLibrary.objects.filter(
        spectrum_type__iexact="standard"
    ).exclude(
        smiles__isnull=True
    ).exclude(
        smiles=""
    )

    debug["total_standard"] = qs.count()
    debug["valid_smiles"] = debug["total_standard"]

    # =========================================================
    # 3️⃣ InChIKey 精确匹配（最高优先级）
    # =========================================================
    exact_hits = []

    if query_inchikey:
        for obj in qs.iterator():
            try:
                if smiles_to_inchikey(obj.smiles) == query_inchikey:
                    exact_hits.append(obj)
            except Exception:
                continue

    debug["exact_hits"] = len(exact_hits)

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
    # 4️⃣ Morgan 指纹相似度搜索（⚠️ 只用数据库指纹）
    # =========================================================
    SIM_THRESHOLD = 0.3

    matched = []
    similarity_map = {}
    max_sim = 0.0

    for obj in qs.iterator():
        fp_db = obj.get_fingerprint()
        if fp_db is None:
            continue

        sim = DataStructs.TanimotoSimilarity(fp_query, fp_db)
        similarity_map[obj.id] = sim
        max_sim = max(max_sim, sim)

        if sim >= SIM_THRESHOLD:
            matched.append(obj)

    debug["max_similarity"] = round(max_sim, 3)

    if not matched:
        return render(
            request,
            "web/structure_results.html",
            {
                "results": [],
                "query_smiles": canonical_smiles,
                "debug": debug,
            }
        )

    # =========================================================
    # 5️⃣ InChIKey 聚合 + 取每组最高相似度
    # =========================================================
    results = aggregate_by_inchikey(matched)

    for r in results:
        sims = [
            similarity_map[obj.id]
            for obj in matched
            if smiles_to_inchikey(obj.smiles) == r["inchikey"]
        ]
        r["similarity"] = f"{max(sims):.3f}" if sims else "0.000"

    results.sort(
        key=lambda x: float(x["similarity"]),
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
        print("msms_input:", msms_input)
        parent_mz  = request.GET.get("parent_mz")
        print("parent_mz:", parent_mz)
        ion_mode   = request.GET.get("ion_mode", "").lower().strip()
        print("ion_mode:", ion_mode)

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
        print("Normalized intensities:", spectrum.intensities)
        spectrum = select_by_mz(spectrum, mz_from=50, mz_to=2000)
        print("Selected by mz:", spectrum.mz)
        spectrum = require_minimum_number_of_peaks(
            spectrum, n_required=3
        )
        print("Required minimum number of peaks:", spectrum)

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
    print(f"Total raw results: {len(raw_results)}")
    filtered = [
        r for r in raw_results
        if r.get("score", 0) >= 0.6
    ]
    print(f"Found {len(filtered)} results")
    print(f"Filtered results: {filtered}")

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

