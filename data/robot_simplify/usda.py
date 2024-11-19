from pxr import Usd

# 加载USD文件
usd_file_path = 'robot_simplify.usd'  # 替换为你的USD文件路径
usda_file_path = 'robot_simplify.usda'  # 替换为输出USDA文件路径

# 打开USD文件并保存为USDA格式
stage = Usd.Stage.Open(usd_file_path)
stage.GetRootLayer().Export(usda_file_path)
