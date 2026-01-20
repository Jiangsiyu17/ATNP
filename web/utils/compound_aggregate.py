from rdkit import Chem
from rdkit.Chem import inchi


def smiles_to_inchikey(smiles: str):
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return inchi.MolToInchiKey(mol)
    except Exception:
        return None


def _format_mz(val):
    try:
        return f"{float(val):.4f}"
    except (TypeError, ValueError):
        return "-"


def aggregate_by_inchikey(
    objects,
    *,
    keep_best_key=None,   # "similarity" / "score" 等
):
    """
    输入：
        objects: iterable[CompoundLibrary]
    输出：
        list[dict] —— InChIKey 去重后的统一结果
    """

    rows_dict = {}        # inchikey → record
    inchikey_cache = {}  # smiles → inchikey

    for obj in objects:
        smiles = (obj.smiles or "").strip()
        if not smiles:
            continue

        if smiles not in inchikey_cache:
            inchikey_cache[smiles] = smiles_to_inchikey(smiles)

        inchikey = inchikey_cache[smiles]
        if not inchikey:
            continue

        # ========= 构造当前对象的 record =========
        record = {
            "inchikey": inchikey,

            # ⭐ 化合物名称优先级
            "compound_name": (
                obj.standard
                or obj.chinese_name
                or obj.latin_name
                or "(unknown)"
            ),

            # ⭐ detail 跳转用
            "first_id": obj.id,

            # ⭐ 数值 / 展示字段
            "precursor_mz": obj.precursor_mz or obj.pepmass,
            "smiles": smiles,
            "plants": getattr(obj, "plants", "") or "",

            # ⭐ 聚合字段（一定要初始化）
            "ionmodes": set(),
            "databases": set(),
        }

        if obj.ionmode:
            record["ionmodes"].add(obj.ionmode)

        if obj.database:
            db = obj.database.lower()
            record["databases"].add(
                "NIST" if db in {"nist20", "nist"} else db.upper()
            )

        # ========= InChIKey 第一次出现 =========
        if inchikey not in rows_dict:
            rows_dict[inchikey] = record
            continue

        # ========= 已存在：做合并 =========
        r = rows_dict[inchikey]

        # 1️⃣ 名称补全
        if r["compound_name"] == "(unknown)" and record["compound_name"] != "(unknown)":
            r["compound_name"] = record["compound_name"]
            r["first_id"] = record["first_id"]

        # 2️⃣ ionmode / database 合并
        r["ionmodes"].update(record["ionmodes"])
        r["databases"].update(record["databases"])

        # 3️⃣ 数值字段兜底（只在原来为空时）
        if not r["precursor_mz"] and record["precursor_mz"]:
            r["precursor_mz"] = record["precursor_mz"]

        if not r["plants"] and record["plants"]:
            r["plants"] = record["plants"]

        # 4️⃣ 保留最高分数对象（structure / msms 用）
        if keep_best_key and hasattr(obj, keep_best_key):
            new = getattr(obj, keep_best_key, None)
            old = r.get(keep_best_key, -1)
            if new is not None and new > old:
                r[keep_best_key] = new
                r["first_id"] = obj.id

    # ========= 转为模板可直接用的格式 =========
    results = []
    for r in rows_dict.values():
        results.append({
            "inchikey": r["inchikey"],
            "compound_name": r["compound_name"],
            "first_id": r["first_id"],
            "precursor_mz": _format_mz(r.get("precursor_mz")),
            "smiles": r["smiles"],
            "plants": r["plants"],
            "ionmode": ", ".join(sorted(r["ionmodes"])) or "-",
            "database": ", ".join(sorted(r["databases"])) or "-",
        })

    results.sort(key=lambda x: x["compound_name"])
    return results

from rdkit.Chem.MolStandardize import rdMolStandardize

def normalize_mol(mol):
    """
    RDKit 分子标准化：
    - 去盐
    - 中和电荷
    """
    if mol is None:
        return None

    try:
        remover = rdMolStandardize.SaltRemover()
        mol = remover.StripMol(mol, dontRemoveEverything=True)

        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)

        return mol
    except Exception:
        return mol
