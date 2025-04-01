#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
from PIL import Image, ImageEnhance


def convert_icons_to_white(icons_dir):
    """
    将指定目录下所有icons8开头的PNG图片转换为白色版本，并保存为w-icons8-开头的文件

    Args:
        icons_dir: 图标目录路径
    """
    # 获取所有PNG图片
    all_icon_files = glob.glob(os.path.join(icons_dir, "*.png"))
    # 特别关注icons8开头的图片
    icons8_files = [f for f in all_icon_files if os.path.basename(
        f).startswith("icons8-")]

    if not icons8_files:
        print(f"在 {icons_dir} 目录下未找到icons8开头的PNG图片文件")
        return

    print(f"找到 {len(icons8_files)} 个icons8图标文件")
    process = input("是否生成所有icons8图标的白色版本？(y/n): ")

    if process.lower() != 'y':
        print("操作已取消")
        return

    # 处理每个图片
    for icon_file in icons8_files:
        filename = os.path.basename(icon_file)
        # 生成白色版本的文件名
        white_filename = "w-" + filename
        white_filepath = os.path.join(icons_dir, white_filename)

        try:
            # 打开图片
            with Image.open(icon_file) as img:
                # 转换为RGBA模式（如果还不是）
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')

                # 获取图片数据
                data = img.getdata()

                # 创建新的像素数据
                new_data = []
                for item in data:
                    # 如果像素不是完全透明的
                    if item[3] > 0:
                        # 将非透明像素转换为白色，保持原始透明度
                        new_data.append((255, 255, 255, item[3]))
                    else:
                        # 保持完全透明的像素不变
                        new_data.append(item)

                # 更新图片数据
                img.putdata(new_data)

                # 保存为新的白色版本文件
                img.save(white_filepath)

                print(f"已生成 {white_filename} 的白色版本")

        except Exception as e:
            print(f"处理 {filename} 时出错: {e}")

    print("所有icons8图标的白色版本生成完成！")


if __name__ == "__main__":
    # 设置图标目录路径
    icons_directory = "labelme/icons"

    # 生成icons8图标的白色版本
    convert_icons_to_white(icons_directory)
