# web/management/commands/precalc_similarities.py
import time
from django.core.management.base import BaseCommand
from django.db import transaction
from web.models import CompoundLibrary, CompoundSimilarity
from web.utils.similarity import generate_spectrum_comparison
from tqdm import tqdm


class Command(BaseCommand):
    help = "预计算每个标准品与样本的相似度，保存到 CompoundSimilarity 表"

    def handle(self, *args, **options):
        start_time = time.time()

        # 取标准品和样本
        standards = CompoundLibrary.objects.filter(spectrum_type="standard")
        samples = CompoundLibrary.objects.filter(spectrum_type="sample")

        self.stdout.write(self.style.NOTICE(
            f"共找到 {standards.count()} 条标准品记录，{samples.count()} 条样本记录，开始计算相似度..."
        ))

        total_saved = 0

        with transaction.atomic():
            for compound in tqdm(standards, desc="Processing compounds", unit="compound"):
                print(f"Processing compound {compound.id}", flush=True)
                # 删除旧记录
                CompoundSimilarity.objects.filter(compound=compound).delete()

                # 获取相似样本，注意不再传 samples 参数
                similar_samples = generate_spectrum_comparison(
                    compound, top_k=20, score_threshold=None
                )

                for item in similar_samples:
                    sample_obj = item["sample"]
                    score = item["score"]

                    CompoundSimilarity.objects.create(
                        compound=compound,
                        similar_compound=sample_obj,
                        similarity_score=score,
                        latin_name=sample_obj.latin_name,
                        chinese_name=sample_obj.chinese_name,
                        tissue=sample_obj.tissue
                    )

                total_saved += len(similar_samples)

        elapsed_time = time.time() - start_time
        self.stdout.write(self.style.SUCCESS(
            f"✅ 预计算完成，总共存储了 {total_saved} 条相似记录，耗时 {elapsed_time:.2f} 秒"
        ))
