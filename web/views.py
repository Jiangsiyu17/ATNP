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
from django.db.models import Min, Max
from django.http import JsonResponse

def compound_list(request):
    return render(request, "web/compound_list.html")

def compound_list_api(request):
    draw = int(request.GET.get("draw", 1))
    start = int(request.GET.get("start", 0))
    length = int(request.GET.get("length", 20))

    # ===== MW筛选参数 =====
    mw_min = request.GET.get("mw_min")
    mw_max = request.GET.get("mw_max")

    # ===== 列搜索 =====
    search_standard = request.GET.get("columns[1][search][value]", "").strip()
    search_database = request.GET.get("columns[2][search][value]", "").strip().lower()
    search_smiles = request.GET.get("columns[3][search][value]", "").strip()
    search_antitumor = request.GET.get("columns[4][search][value]", "").strip().lower()
    search_ionmode = request.GET.get("columns[5][search][value]", "").strip().lower()

    # ===== 基础 queryset =====
    base_qs = CompoundLibrary.objects.filter(spectrum_type__iexact="standard")

    qs = base_qs

    # =========================
    # 🧠 RDKit MW 计算缓存
    # =========================
    mw_cache = {}

    def calc_mw(smiles):
        if not smiles:
            return None
        if smiles in mw_cache:
            return mw_cache[smiles]

        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            mw_cache[smiles] = None
            return None

        mw = Descriptors.MolWt(mol)
        mw_cache[smiles] = mw
        return mw

    # =========================
    # 🔥 MW筛选（核心）
    # =========================
    if mw_min or mw_max:
        try:
            mw_min_f = float(mw_min) if mw_min else None
            mw_max_f = float(mw_max) if mw_max else None

            filtered_ids = []

            for obj in qs:
                mw = calc_mw(obj.smiles)
                if mw is None:
                    continue

                if mw_min_f is not None and mw < mw_min_f:
                    continue
                if mw_max_f is not None and mw > mw_max_f:
                    continue

                filtered_ids.append(obj.id)

            qs = qs.filter(id__in=filtered_ids)

        except ValueError:
            pass

    # =========================
    # 其他字段筛选
    # =========================
    if search_standard:
        qs = qs.filter(standard__icontains=search_standard)

    if search_smiles:
        qs = qs.filter(smiles__icontains=search_smiles)

    if search_database:
        qs = qs.filter(database__iexact=search_database)

    if search_antitumor in {"true", "false"}:
        qs = qs.filter(antitumor=(search_antitumor == "true"))

    if search_ionmode in {"positive", "negative"}:
        qs = qs.filter(ionmode__iexact=search_ionmode)

    # =========================
    # 总数
    # =========================
    records_total = base_qs.values("standard").distinct().count()
    records_filtered = qs.values("standard").distinct().count()

    # =========================
    # 聚合
    # =========================
    qs_agg = (
        qs.values("standard")
        .annotate(
            first_id=Min("id"),
            smiles=Min("smiles"),
            antitumor_flag=Max("antitumor")   # 👈 关键：只要有一个True就算True
        )
        .order_by("-antitumor_flag", Lower("standard"))  # 👈 True在前
    )[start:start + length]

    standards = [r["standard"] for r in qs_agg]

    db_map = defaultdict(set)
    extra_map = {}
    candidates_by_standard = defaultdict(list)

    full_qs = qs.filter(standard__in=standards)

    for obj in full_qs:
        key = obj.standard
        candidates_by_standard[key].append(obj)

        if obj.database:
            db_lower = obj.database.lower()
            if db_lower in {"nist", "nist20"}:
                db_map[key].add("NIST")
            else:
                db_map[key].add(db_lower.upper())

        if key not in extra_map:
            extra_map[key] = {
                "antitumor": False,
                "ionmode": obj.ionmode
            }

        if obj.antitumor:
            extra_map[key]["antitumor"] = True

    candidate_standard_ids = [
        obj.standard_id
        for objects in candidates_by_standard.values()
        for obj in objects
        if obj.standard_id
    ]
    matched_pairs = set(
        CompoundLibrary.objects.filter(
            spectrum_type="sample",
            matched_spectrum_id__in=candidate_standard_ids,
        ).values_list("matched_spectrum_id", "ionmode")
    )
    detail_id_map = {}
    for standard_name, objects in candidates_by_standard.items():
        matched_objects = [
            obj
            for obj in objects
            if (str(obj.standard_id), obj.ionmode) in matched_pairs
        ]
        if matched_objects:
            detail_id_map[standard_name] = min(obj.id for obj in matched_objects)

    # =========================
    # 返回数据
    # =========================
    data = []
    for i, row in enumerate(qs_agg, start=start + 1):
        std = row["standard"]
        extra = extra_map.get(std, {})

        antitumor_val = extra.get("antitumor")
        antitumor_str = "True" if antitumor_val else "False"

        data.append({
            "index": i,
            "standard": std or "(unknown)",
            "database": ", ".join(sorted(db_map.get(std, []))) or "-",
            "smiles": row["smiles"] or "-",
            "antitumor": antitumor_str,
            "ionmode": (extra.get("ionmode") or "-").lower(),
            "action": (
                f'<a class="btn btn-sm btn-outline-primary" '
                f'href="/compound/{detail_id_map.get(std, row["first_id"])}/">View</a>'
            )
        })

    return JsonResponse({
        "draw": draw,
        "recordsTotal": records_total,
        "recordsFiltered": records_filtered,
        "data": data
    })


def home(request):
    return render(request, 'web/home.html')


def search(request):
    query = request.GET.get("q", "").strip()

    if not query:
        return render(
            request,
            "web/search_not_found.html",
            {"query": query}
        )

    # ======================================================
    # 1️⃣ 搜化合物名称，并先展示相似命中列表
    # ======================================================
    compound_qs = CompoundLibrary.objects.filter(
        spectrum_type__iexact="standard"
    ).filter(
        Q(standard__icontains=query) |
        Q(title__icontains=query)
    )

    if compound_qs.exists():
        # 同名化合物可能因数据库来源或离子模式不同而有多条谱图记录。
        # 搜索结果按名称合并，并保留一条记录作为详情页入口。
        grouped_results = (
            compound_qs
            .values("standard")
            .annotate(
                first_id=Min("id"),
                smiles=Min("smiles"),
                database=Min("database"),
                ionmode=Min("ionmode"),
            )
            .order_by(Lower("standard"))
        )
        result_page = Paginator(grouped_results, 20).get_page(
            request.GET.get("page")
        )
        for result in result_page.object_list:
            database_name = (result.get("database") or "").strip()
            if database_name.lower() in {"nist20", "nist 20"}:
                result["database"] = "NIST"

        return render(
            request,
            "web/search_results.html",
            {
                "query": query,
                "results": result_page,
            },
        )

    # ======================================================
    # 2️⃣ 搜植物
    # ======================================================
    sample = CompoundLibrary.objects.filter(
        spectrum_type="sample"
    ).filter(
        Q(latin_name__icontains=query) |
        Q(chinese_name__icontains=query)
    ).first()

    if sample:

        std_obj = CompoundLibrary.objects.filter(
            spectrum_type="standard",
            standard_id=sample.matched_spectrum_id
        ).first()

        if std_obj:

            return redirect(
                reverse(
                    "compound_detail",
                    args=[std_obj.pk]
                )
            )

    # ======================================================
    # 3️⃣ Not Found
    # ======================================================

    return render(
        request,
        "web/search_not_found.html",
        {"query": query}
    )


from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, QED
import rdkit.Chem.AllChem as AllChem
from web.utils import sascorer

from django.core.paginator import Paginator
from web.utils.similar_cache import get_similar_samples

from web.utils.pubmed_utils import get_pubmed_papers


def parse_precursor_mz(precursor_mz, pepmass):
    """Return a display-safe precursor m/z without failing the whole page."""
    if precursor_mz is not None:
        try:
            return round(float(precursor_mz), 4)
        except (TypeError, ValueError):
            pass

    if pepmass:
        try:
            return round(float(str(pepmass).split()[0]), 4)
        except (TypeError, ValueError, IndexError):
            pass

    return "-"


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
    plant_sources=[]

    if compound.standard_id:
        matched_samples=CompoundLibrary.objects.filter(
            spectrum_type="sample",
            matched_spectrum_id=str(compound.standard_id),
            ionmode=compound.ionmode
        ).only(
            "id",
            "chinese_name",
            "latin_name",
            "tissue",
            "matched_spectrum_id",
            "precursor_mz",
            "pepmass",
            "ionmode",
        )

        for s in matched_samples:
            raw_latin=s.latin_name if s.latin_name not in ("","-",None) else None
            plant_sources.append({
                "pid":s.id,
                "chinese_name":s.chinese_name or "-",
                "latin_name": format_latin_name (s.latin_name or "-"),
                "tissue": s.tissue.capitalize() if s.tissue else "-",
                "matched_id": s.matched_spectrum_id,
                "latin_slug": slugify(raw_latin) if raw_latin else "unknown-plant",
                "precursor_mz": parse_precursor_mz(s.precursor_mz, s.pepmass),
                "ionmode": s.ionmode or "-"
            })
    uniq={}

    for ps in plant_sources:
        key=(
            ps["chinese_name"],
            ps["latin_name"],
            ps["ionmode"]
        )
        uniq[key]=ps

    plant_sources=list(
        uniq.values()
    )

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
            # "ionmode": r.get("ionmode", "-"),
            "spectrum_idx": r.get("spectrum_index"),
        }
        for r in sorted(best.values(), key=lambda x: x["score"], reverse=True)
    ]
     
    # ===== 分页处理 =====
    plant_page = Paginator(plant_sources, 10).get_page(request.GET.get("plant_page"))

    # ✅ 如果 similar_samples 不为空才分页，否则直接 None
    sample_page = Paginator(similar_samples, 10).get_page(request.GET.get("sample_page")) if similar_samples else None

    # ===== 表格 C：PubMed 文献 =====
    pubmed_raw = get_pubmed_papers(compound.standard)

    # ✅ 可选：按年份排序（推荐）
    pubmed_raw = sorted(
        pubmed_raw,
        key=lambda x: x.get("year", 0),
        reverse=True
    )

    pubmed_page = Paginator(pubmed_raw, 10).get_page(
        request.GET.get("pubmed_page")
    )
    print("PubMed数量:", len(pubmed_raw))

    return render(request, "web/compound_detail.html", {
        "compound": compound,
        "mol_img": mol_img,
        "rdkit": rdkit_props,
        "plant_sources": plant_page,      # 已分页
        "similar_samples": sample_page,   # 已分页
        "pubmed_page": pubmed_page,       # 已分页
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

        # 获取 matched 中第一个对象来读取 antitumor
        obj_for_antitumor = next(
            (obj for obj in matched if smiles_to_inchikey(obj.smiles) == r["inchikey"]),
            None
        )

        # ✅ 转成字符串 "True"/"False" 直接展示
        if obj_for_antitumor:
            r["antitumor"] = "True" if obj_for_antitumor.antitumor else "False"
        else:
            r["antitumor"] = "False"

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
from django.http import JsonResponse
from web.models import CompoundLibrary

def mw_api(request):
    draw = int(request.GET.get("draw", 1))
    start = int(request.GET.get("start", 0))
    length = int(request.GET.get("length", 20))

    # ===== MW筛选 =====
    mw_min = request.GET.get("mw_min")
    mw_max = request.GET.get("mw_max")

    try:
        mw_min = float(mw_min) if mw_min else None
        mw_max = float(mw_max) if mw_max else None
    except ValueError:
        mw_min = mw_max = None

    # ===== 列筛选 =====
    search_database = request.GET.get("columns[2][search][value]", "").strip().lower()
    search_antitumor = request.GET.get("columns[5][search][value]", "").strip().lower()
    search_ionmode = request.GET.get("columns[6][search][value]", "").strip().lower()

    # ===== 基础 queryset =====
    base_qs = CompoundLibrary.objects.filter(
        spectrum_type__iexact="standard",
        smiles__isnull=False
    ).exclude(smiles="")

    qs = base_qs

    # ===== MW筛选 =====
    if mw_min is not None:
        qs = qs.filter(mw__gte=mw_min)
    if mw_max is not None:
        qs = qs.filter(mw__lte=mw_max)

    # ===== database筛选（关键统一）=====
    if search_database:
        if search_database == "nist":
            qs = qs.filter(database__in=["nist", "nist20"])
        else:
            qs = qs.filter(database__iexact=search_database)

    # ===== antitumor =====
    if search_antitumor in {"true", "false"}:
        qs = qs.filter(antitumor=(search_antitumor == "true"))

    # ===== ionmode =====
    if search_ionmode in {"positive", "negative"}:
        qs = qs.filter(ionmode__iexact=search_ionmode)

    # =========================
    # 📊 统计
    # =========================
    records_total = base_qs.values(
        "standard",
        "ionmode"
    ).distinct().count()

    records_filtered = qs.values(
        "standard",
        "ionmode"
    ).distinct().count()

    # =========================
    # 🧠 按 standard 聚合（关键）
    # =========================
    qs_agg = (
        qs.values(
            "standard",
            "ionmode"
        )
        .annotate(
            first_id=Min("id"),
            smiles=Min("smiles"),
            mw=Min("mw"), 
            antitumor_flag=Max("antitumor")
        )
        .order_by(
            "-antitumor_flag",
            Lower("standard"),
            Lower("ionmode")
        )
    )[start:start + length]

    standards = [(r["standard"], r["ionmode"]) for r in qs_agg ]

    # ===== 构建 database 合并 =====
    db_map = defaultdict(set)
    extra_map = {}

    full_qs = []

    for std, ion in standards:
        full_qs.extend(
            qs.filter(
                standard=std,
                ionmode=ion
            )
        )

    for obj in full_qs:
        key = (obj.standard, obj.ionmode)

        # database 合并（完全复制 compoundlist 逻辑）
        if obj.database:
            db_lower = obj.database.lower()
            if db_lower in {"nist", "nist20"}:
                db_map[key].add("NIST")
            else:
                db_map[key].add(db_lower.upper())

        # antitumor + ionmode
        if key not in extra_map:
            extra_map[key] = {
                "antitumor": obj.antitumor,
                "ionmode": obj.ionmode
            }

    # =========================
    # 📦 返回数据
    # =========================
    data = []
    for i, row in enumerate(qs_agg, start=start + 1):
        std = row["standard"]
        ion = row["ionmode"]

        key = (std, ion)

        extra = extra_map.get(key,{})

        data.append({
            "index": i,
            "standard": std or "(unknown)",
            "database": ", ".join(sorted(db_map.get(key, []))) or "-",
            "smiles": row["smiles"] or "-",
            "mw": round(row["mw"], 2) if row["mw"] else "-",
            "antitumor": "True" if extra.get("antitumor") else "False",
            "ionmode": ion.lower(),
            "action": f'<a class="btn btn-sm btn-outline-primary" href="/compound/{row["first_id"]}/">View</a>'
        })

    return JsonResponse({
        "draw": draw,
        "recordsTotal": records_total,
        "recordsFiltered": records_filtered,
        "data": data
    })

def molecular_weight_search(request):
    # ✅ 改成 GET
    min_mw = request.GET.get("mw_min", "").strip()
    max_mw = request.GET.get("mw_max", "").strip()

    try:
        min_mw = float(min_mw) if min_mw else None
        max_mw = float(max_mw) if max_mw else None
    except ValueError:
        return render(request, "web/molecular_weight_query.html", {
            "error": "Invalid number input."
        })

    # 如果两个都没填 → 不查，回表单页
    if min_mw is None and max_mw is None:
        return render(request, "web/molecular_weight_query.html")

    # ✅ 直接进入结果页面（DataTables 会自动调 mw_api）
    return render(request, "web/molecular_weight_results.html", {
        "min_mw": min_mw,
        "max_mw": max_mw,
    })


from django.shortcuts import render
from django.db.models import Q
import numpy as np
from matchms import Spectrum
from matchms.filtering import (
    normalize_intensities,
    select_by_mz,
    require_minimum_number_of_peaks
)
import gensim
import hnswlib
from spec2vec import SpectrumDocument
from spec2vec.vector_operations import calc_vector
import os
from django.conf import settings



# ================== 预加载模型和索引 ==================
MODEL_DIR = settings.BASE_DIR / "model"

IONMODES = {
    "positive": {
        "spectra": os.path.join(MODEL_DIR, "standards_spectra_pos.pickle"),
        "index": os.path.join(MODEL_DIR, "standards_index_pos.bin"),
        "model": os.path.join(MODEL_DIR, "Ms2Vec_allGNPSpositive.hdf5")
    },
    "negative": {
        "spectra": os.path.join(MODEL_DIR, "standards_spectra_neg.pickle"),
        "index": os.path.join(MODEL_DIR, "standards_index_neg.bin"),
        "model": os.path.join(MODEL_DIR, "Ms2Vec_allGNPSnegative.hdf5")
    }
}

TOPK = 10  # 返回前10相似谱图

# ================== 预加载函数 ==================
def load_model_and_index(ion_mode):
    cfg = IONMODES.get(ion_mode)
    if not cfg:
        return None, None, None

    # 1️⃣ spectra
    with open(cfg["spectra"], "rb") as f:
        spectra = pickle.load(f)

    # 2️⃣ Word2Vec 模型
    w2v_model = gensim.models.Word2Vec.load(cfg["model"])

    # 3️⃣ HNSW index
    hnsw_index = hnswlib.Index(space="cosine", dim=w2v_model.vector_size)
    hnsw_index.load_index(cfg["index"])
    hnsw_index.set_ef(100)
    return spectra, w2v_model, hnsw_index


# ================== 搜索页面 ==================
def msms_search(request):
    """显示搜索页面"""
    return render(request, 'web/msms_search.html')


# ================== 搜索结果 ==================
def msms_result(request):
    results = []
    error = None

    if not request.GET:
        return render(request, "web/msms_result.html", {
            "results": [],
            "error": None
        })

    # 1️⃣ 解析用户输入谱图
    try:
        msms_input = request.GET.get("msms_spectrum", "").strip()
        parent_mz  = request.GET.get("parent_mz")
        ion_mode   = request.GET.get("ion_mode", "positive").lower().strip()

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
        spectrum = require_minimum_number_of_peaks(spectrum, n_required=3)
        if spectrum is None:
            raise ValueError("Spectrum discarded after preprocessing.")

    except Exception as e:
        return render(request, "web/msms_result.html", {
            "results": [],
            "error": f"MS/MS spectrum parsing error: {e}"
        })

    # 2️⃣ 选择对应 ionmode 的模型和索引
    spectra_list, w2v_model, hnsw_index = load_model_and_index(ion_mode)
    if not spectra_list or not w2v_model or not hnsw_index:
        return render(request, "web/msms_result.html", {
            "results": [],
            "error": f"No preloaded model/index for ion mode '{ion_mode}'"
        })

    # 3️⃣ 转为 Spec2Vec 向量
    try:
        doc = SpectrumDocument(spectrum, n_decimals=2)
        query_vec = calc_vector(w2v_model, doc, allowed_missing_percentage=5)
        if query_vec is None or np.all(query_vec == 0):
            raise ValueError("Cannot compute vector for this spectrum.")
        query_vec /= np.linalg.norm(query_vec)
    except Exception as e:
        return render(request, "web/msms_result.html", {
            "results": [],
            "error": f"Vectorization error: {e}"
        })

    # 4️⃣ HNSW TopK 检索
    labels, distances = hnsw_index.knn_query(query_vec, k=TOPK)

    # 5️⃣ 构建返回结果
    for idx, dist in zip(labels[0], distances[0]):
        spec = spectra_list[idx]
        obj_id = spec.metadata.get("compound_id")
        try:
            obj = CompoundLibrary.objects.get(id=obj_id)
        except CompoundLibrary.DoesNotExist:
            continue

        # 统一 database 名称
        db_name = (obj.database or "").strip().lower()
        if db_name in ["nist", "nist20"]:
            db_name = "NIST"
        elif db_name in ["in-house", "inhouse"]:
            db_name = "ATNP"
        else:
            db_name = obj.database or "-"

        results.append({
            "id": obj.id,
            "compound_name": getattr(obj, "standard", None) or getattr(obj, "name", None),
            "smiles": obj.smiles,
            "inchikey": obj.inchikey,
            "precursor_mz": f"{obj.precursor_mz:.4f}" if obj.precursor_mz else None,
            "ionmode": obj.ionmode,
            "database": db_name,
            "best_score": f"{max(0, 1 - dist):.4f}",   # cosine distance -> similarity
            "antitumor": bool(obj.antitumor)  # 假设模型字段为 BooleanField
        })

    if not results:
        error = "No matched standard spectra found."

    return render(request, "web/msms_result.html", {
        "results": results,
        "error": error
    })
