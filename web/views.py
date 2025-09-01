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
        "database": ", ".join(sorted(r["databases"])).upper() or "-",
        "smiles": r["smiles"],
        "pepmass": r["pepmass"],  # 添加 pepmass 字段
    } for r in rows_dict.values()]

    rows.sort(key=lambda x: _canon(x["standard"]))

    # 确保传递给模板的数据没有问题
    return render(request, 'web/herb_detail.html', {
        'latin_name': format_latin_name(latin_name),
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


def compound_detail(request, pk):
    logger = logging.getLogger(__name__)
    logger.info(f"→ Enter compound_detail, pk={pk}")

    # 获取化合物对象
    compound = get_object_or_404(CompoundLibrary, pk=pk)
    mol_img = plot_ref_mol(compound.smiles) if compound.smiles else None

    # ===== 表格 A：数据库来源 =====
    qs = CompoundLibrary.objects.filter(
        standard=compound.standard
    ).exclude(latin_name__isnull=True).exclude(latin_name__exact='').distinct()

    plant_sources = []
    for item in qs:
        # 生成安全 URL
        compound_url = quote(compound.standard, safe='')  # 对特殊字符（包括 / + - 等）做编码
        plant_sources.append({
            "chinese_name": item.chinese_name,
            "latin_name": format_latin_name(item.latin_name),
            "latin_slug": slugify(item.latin_name),
            "compound_raw": compound.standard,
            "compound_url": compound_url,
        })

    
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
                "tissue": r.get("tissue"),
                "score": r.get("score"),
                "latin_slug": slugify(r.get("latin_name") or "")
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