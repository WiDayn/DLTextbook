# -*- coding: utf-8 -*-
# @时间: 2024-11-27
# @作者: 曾强
# @邮箱: skv@live.com
#
# 整体描述: 遍历所有目录，输出项目依赖
# 输入:
# 输出:
# 限制性条件: pip install pipreqs
# 算法/数据来源(论文/代码):
# 修改历史:
# 使用示例:
import os
import subprocess

def generate_requirements_txt(folder_path):
    if os.path.isdir(folder_path):
        try:
            subprocess.run(["pipreqs", folder_path, "--force", "--ignore", "tests"], check=True)
            print(f"requirements.txt 已成功生成在 {folder_path} 目录下！遍历并非完全准确，请手动核对是否有不需要的依赖！")
        except subprocess.CalledProcessError as e:
            print(f"生成 requirements.txt 时发生错误: {e}")
    else:
        print("输入的路径无效，请重新输入。")

if __name__ == "__main__":
    folder_path = "./"
    generate_requirements_txt(folder_path)