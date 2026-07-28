import os

def process_mgf(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:

        block = []
        in_target = False  # 标记当前谱图是否需要修改

        for line in f_in:
            line_strip = line.strip()

            # 开始一个新的谱图块
            if line_strip == "BEGIN IONS":
                block = []
                in_target = False

            # 检测 DATABASE 字段
            if line_strip.startswith("DATABASE="):
                if "IN-HOUSE" in line_strip:
                    line = "DATABASE=ATNP\n"
                    in_target = True

            # 如果是目标谱图，修改 ANTITUMOR
            elif in_target and line_strip.startswith("ANTITUMOR="):
                if "FALSE" in line_strip:
                    line = "ANTITUMOR=TRUE\n"

            block.append(line)

            # 谱图块结束，写入文件
            if line_strip == "END IONS":
                f_out.writelines(block)

        print(f"处理完成: {input_path} -> {output_path}")


if __name__ == "__main__":
    # 👉 在这里填你的两个 mgf 文件路径
    mgf_files = [
        "/data2/jiangsiyu/ATNP_Database/5_negative_standards_with_plants_final_clean.mgf",
        "/data2/jiangsiyu/ATNP_Database/5_positive_standards_with_plants_final_clean.mgf"
    ]

    for input_file in mgf_files:
        if not os.path.exists(input_file):
            print(f"文件不存在: {input_file}")
            continue

        output_file = input_file.replace(".mgf", "_modified.mgf")
        process_mgf(input_file, output_file)

    print("全部处理完成！")