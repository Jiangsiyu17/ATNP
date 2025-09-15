import pickle

with open("/data2/jiangsiyu/ATNP_Database/model_copy/herbs_spectra_neg.pickle", "rb") as f:
    spectra = pickle.load(f)

print(f"Total spectra: {len(spectra)}")
spec = spectra[0]  # 看第一个谱图

print("\n--- Metadata ---")
for k, v in spec.metadata.items():
    print(f"{k}: {v}")

print("\n--- Peaks (first 10) ---")
print("m/z:", spec.peaks.mz[:10])
print("intensities:", spec.peaks.intensities[:10])
