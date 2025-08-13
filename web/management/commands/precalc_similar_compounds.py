from django.core.management.base import BaseCommand
from django.db import transaction
from web.models import CompoundLibrary, CompoundSimilarity
from web.utils.plotting import generate_spectrum_comparison  # 你前面给的 plotting 文件里的方法


class Command(BaseCommand):
    help = "预计算每个化合物的相似植物及光谱图，并存到 CompoundSimilarity 表中"

    def handle(self, *args, **options):
        compounds = CompoundLibrary.objects.filter(spectrum_type="standard")
        self.stdout.write(self.style.NOTICE(f"共找到 {compounds.count()} 个化合物标准库记录，开始预计算..."))

        total_saved = 0

        with transaction.atomic():
            for idx, compound in enumerate(compounds, start=1):
                self.stdout.write(f"[{idx}/{compounds.count()}] 处理 {compound.standard} ...")

                # 先删掉旧的
                CompoundSimilarity.objects.filter(compound=compound).delete()

                # 调用已有的光谱比较逻辑，获取相似样本
                similar_samples = generate_spectrum_comparison(
                    CompoundLibrary.objects.filter(spectrum_type="sample"),  # 比对样本库
                    only_nist=False
                )

                # 存表
                for sample_data in similar_samples:
                    sample = sample_data["sample"]
                    img_base64 = sample_data["image"]

                    CompoundSimilarity.objects.create(
                        compound=compound,
                        plant_name_cn=sample.chinese_name,
                        plant_name_lat=sample.latin_name,
                        spectrum_image_base64=img_base64
                    )

                total_saved += len(similar_samples)

        self.stdout.write(self.style.SUCCESS(f"✅ 预计算完成，总共存储了 {total_saved} 条相似记录"))
