import csv
import os
from hit_omniverse import HIT_SIM_DATASET_DIR

# 假设 txt 文件是以逗号分隔的
txt_file_path = os.path.join(HIT_SIM_DATASET_DIR, "walk.txt")
excel_file_path = os.path.join(HIT_SIM_DATASET_DIR, "walk_bak1.csv")

# 打开并读取TXT文件
with open(txt_file_path, 'r') as txt_file:
    lines = txt_file.readlines()

# 创建并写入CSV文件
with open(excel_file_path, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    for line in lines:
        # 将每行的数字按空格分隔并处理每个数字
        numbers = []
        for num in line.split():
            # 将数字转换为浮点数并检查是否小于0.001
            if abs(float(num)) < 0.001:
                numbers.append(0)
            else:
                numbers.append(float(num))
        # 写入处理后的数字
        writer.writerow(numbers)

print("TXT文件的内容已成功写入CSV文件。")