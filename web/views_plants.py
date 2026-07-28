import hashlib

from django.core.cache import cache
from django.http import Http404, JsonResponse
from django.shortcuts import get_object_or_404, render
from django.utils.text import slugify

from .models import CompoundLibrary
from .utils.plot_tools import plot_single_spectrum
from .utils.plotting import format_latin_name, plot_2_spectrum


def plant_list(request):
    return render(request, "web/plant_list.html")


def plant_list_api(request):
    draw = int(request.GET.get("draw", 1))
    start = int(request.GET.get("start", 0))
    length = int(request.GET.get("length", 20))

    search_latin = request.GET.get("columns[1][search][value]", "").strip().lower()
    search_chinese = request.GET.get("columns[2][search][value]", "").strip().lower()
    search_tissue = request.GET.get("columns[3][search][value]", "").strip().lower()

    order_col = int(request.GET.get("order[0][column]", 1))
    order_dir = request.GET.get("order[0][dir]", "asc")
    columns = ["index", "latin_name", "chinese_name", "tissue"]
    order_field = columns[order_col] if order_col < len(columns) else "latin_name"
    reverse = order_dir == "desc"

    cache_key = "plant_list:aggregated:v1"
    all_rows = cache.get(cache_key)

    if all_rows is None:
        qs = CompoundLibrary.objects.filter(
            spectrum_type="sample"
        ).values_list(
            "latin_name",
            "chinese_name",
            "tissue",
        )

        plant_map = {}
        for latin, chinese, tissue in qs.iterator(chunk_size=2000):
            latin = (latin or "").strip()
            chinese = (chinese or "").strip()
            tissue = (tissue or "").strip()
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

        all_rows = [
            {
                "latin_name": plant["latin_name"],
                "chinese_name": plant["chinese_name"],
                "tissue": ", ".join(sorted(plant["tissues"])) or "-",
            }
            for plant in plant_map.values()
        ]
        cache.set(cache_key, all_rows, timeout=300)

    records_total = len(all_rows)
    rows = []
    for row in all_rows:
        if search_latin and search_latin not in row["latin_name"].lower():
            continue
        if search_chinese and search_chinese not in row["chinese_name"].lower():
            continue
        if search_tissue and search_tissue not in row["tissue"].lower():
            continue
        rows.append(row)

    rows.sort(key=lambda row: row[order_field].lower(), reverse=reverse)
    page_rows = rows[start:start + length]
    data = [
        {
            "index": index,
            "latin_name": row["latin_name"],
            "chinese_name": row["chinese_name"],
            "tissue": row["tissue"],
            "action": (
                f'<a class="btn btn-sm btn-outline-primary" '
                f'href="/plant/{row["latin_name"]}/">View</a>'
            ),
        }
        for index, row in enumerate(page_rows, start=start + 1)
    ]

    return JsonResponse({
        "draw": draw,
        "recordsTotal": records_total,
        "recordsFiltered": len(rows),
        "data": data,
    })


def plant_detail_api(request, latin_name):
    draw = int(request.GET.get("draw", 1))
    start = int(request.GET.get("start", 0))
    length = int(request.GET.get("length", 20))

    search_standard = request.GET.get("columns[1][search][value]", "").strip()
    search_database = request.GET.get("columns[3][search][value]", "").strip().lower()
    search_smiles = request.GET.get("columns[4][search][value]", "").strip()
    search_antitumor = request.GET.get("columns[5][search][value]", "").strip().lower()
    search_ionmode = request.GET.get("columns[6][search][value]", "").strip().lower()

    cache_suffix = hashlib.sha256(
        latin_name.strip().lower().encode("utf-8")
    ).hexdigest()
    cache_key = f"plant_detail:compounds:v1:{cache_suffix}"
    all_compounds = cache.get(cache_key)

    if all_compounds is None:
        matched_ids = list(
            CompoundLibrary.objects.filter(
                spectrum_type="sample",
                latin_name__iexact=latin_name,
                matched_spectrum_id__isnull=False,
            ).exclude(
                matched_spectrum_id=""
            ).values_list(
                "matched_spectrum_id",
                flat=True,
            ).distinct()
        )

        standards = CompoundLibrary.objects.filter(
            spectrum_type="standard",
            standard_id__in=matched_ids,
        ).order_by("id").values(
            "id",
            "standard_id",
            "standard",
            "precursor_mz",
            "pepmass",
            "database",
            "smiles",
            "antitumor",
            "ionmode",
        )

        standard_map = {}
        for compound in standards.iterator(chunk_size=1000):
            standard_map.setdefault(compound["standard_id"], compound)

        all_compounds = list(standard_map.values())
        cache.set(cache_key, all_compounds, timeout=300)

    records_total = len(all_compounds)
    compounds = all_compounds
    if search_standard:
        compounds = [
            item for item in compounds
            if search_standard.lower() in (item["standard"] or "").lower()
        ]
    if search_smiles:
        compounds = [
            item for item in compounds
            if search_smiles.lower() in (item["smiles"] or "").lower()
        ]
    if search_database:
        compounds = [
            item for item in compounds
            if search_database in (item["database"] or "").lower()
        ]
    if search_antitumor in {"true", "false"}:
        flag = search_antitumor == "true"
        compounds = [item for item in compounds if item["antitumor"] == flag]
    if search_ionmode in {"positive", "negative"}:
        compounds = [
            item for item in compounds
            if (item["ionmode"] or "").lower() == search_ionmode
        ]

    compounds.sort(key=lambda item: (item["standard"] or "").lower())
    records_filtered = len(compounds)
    page_compounds = compounds[start:start + length]

    data = []
    for index, compound in enumerate(page_compounds, start=start + 1):
        if compound["precursor_mz"]:
            precursor = f'{compound["precursor_mz"]:.4f}'
        elif compound["pepmass"]:
            try:
                precursor = f'{float(compound["pepmass"].split()[0]):.4f}'
            except (TypeError, ValueError, IndexError):
                precursor = compound["pepmass"]
        else:
            precursor = "-"

        database = (compound["database"] or "-").upper().replace("NIST20", "NIST")
        data.append({
            "index": index,
            "standard": compound["standard"] or "(unknown)",
            "precursor_mz": precursor,
            "database": database,
            "smiles": compound["smiles"] or "-",
            "antitumor": "True" if compound["antitumor"] else "False",
            "ionmode": (compound["ionmode"] or "-").lower(),
            "action": (
                f'<a class="btn btn-sm btn-outline-primary" '
                f'href="/compound/{compound["id"]}/">View</a>'
            ),
        })

    return JsonResponse({
        "draw": draw,
        "recordsTotal": records_total,
        "recordsFiltered": records_filtered,
        "data": data,
    })


def plant_detail(request, latin_name):
    return render(request, "web/plant_detail.html", {"latin_name": latin_name})


def plant_compound_detail(request, latin_name, compound_id):
    pid = request.GET.get("pid")
    if not pid:
        raise Http404("pid is required")
    try:
        pid = int(pid)
    except ValueError as exc:
        raise Http404("invalid pid") from exc

    compound_obj = get_object_or_404(
        CompoundLibrary,
        pk=compound_id,
        spectrum_type="standard",
    )
    sample_filters = {
        "pk": pid,
        "spectrum_type": "sample",
        "matched_spectrum_id": str(compound_obj.standard_id),
    }
    if compound_obj.ionmode:
        sample_filters["ionmode__iexact"] = compound_obj.ionmode

    sample_obj = get_object_or_404(CompoundLibrary, **sample_filters)
    if latin_name != slugify(sample_obj.latin_name or ""):
        raise Http404("Plant does not match this sample")

    sample_spec = sample_obj.get_spectrum()
    if not sample_spec:
        raise Http404("Sample spectrum not found")

    standard_spec = compound_obj.get_spectrum()
    databases = (compound_obj.database or "").lower().split()
    nist_like = {"nist", "nist20"}
    is_nist_only = bool(databases) and all(
        database in nist_like for database in databases
    )

    try:
        if is_nist_only or not standard_spec:
            image = plot_single_spectrum(sample_spec)
        else:
            image = plot_2_spectrum(sample_spec, standard_spec)
        if not image:
            raise RuntimeError("Empty image")
    except Exception as exc:
        raise RuntimeError(f"Spectrum plotting failed: {exc}") from exc

    entry = {
        "id": sample_obj.id,
        "chinese_name": sample_obj.chinese_name or "-",
        "latin_name": format_latin_name(sample_obj.latin_name or "-"),
        "tissue": sample_obj.tissue or "-",
        "score": sample_obj.score or 0,
        "image": image,
    }
    return render(request, "web/plant_compound_detail.html", {
        "compound": compound_obj.standard,
        "latin_name": entry["latin_name"],
        "entry": entry,
    })
