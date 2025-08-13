# web/utils/spectrum_io.py
from matchms.importing import load_from_mgf
from web.models import CompoundLibrary

def import_mgf_to_db(mgf_path, spectrum_type):
    """
    mgf_path:  样品/标品库 mgf 文件
    spectrum_type: "sample" | "standard"
    """
    for spec in load_from_mgf(mgf_path):
        meta = spec.metadata

        obj = CompoundLibrary(
            standard        = meta.get("STANDARD") or meta.get("NAME"),
            smiles          = meta.get("SMILES"),
            database        = meta.get("DATABASE") or "LOCAL",
            spectrum_type   = spectrum_type,
            # 其余已有字段自行补全……
            precursor_mz    = meta.get("PRECURSOR_MZ") or meta.get("PEPMASS")
        )
        obj.set_spectrum(spec)
        obj.save()
