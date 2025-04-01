#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
from PIL import Image


def resize_icons(icons_dir, target_size=(32, 32)):
    """
    调整指定目录下所有PNG图片为指定大小

    Args:
        icons_dir: 图标目录路径
        target_size: 目标尺寸，默认为32x32
    """
    # 获取所有PNG图片
    all_icon_files = glob.glob(os.path.join(icons_dir, "*.png"))
    # 特别关注icons8开头的图片
    icons8_files = [f for f in all_icon_files if os.path.basename(
        f).startswith("icons8-")]

    if not all_icon_files:
        print(f"在 {icons_dir} 目录下未找到PNG图片文件")
        return

    print(f"找到 {len(all_icon_files)} 个PNG图标文件，其中 {len(icons8_files)} 个是icons8图标")
    process = input("是否处理所有图标文件？(y/n，默认仅处理icons8图标): ")

    # 默认只处理icons8图标
    files_to_process = icons8_files if process.lower() != 'y' else all_icon_files

    if not files_to_process:
        print("没有找到需要处理的图标文件。")
        return

    print(f"将处理 {len(files_to_process)} 个图标文件...")

    # 创建备份目录
    backup_dir = os.path.join(icons_dir, "original_size")
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
        print(f"创建备份目录: {backup_dir}")

    # 处理每个图片
    for icon_file in files_to_process:
        filename = os.path.basename(icon_file)

        try:
            # 备份原图
            backup_path = os.path.join(backup_dir, filename)
            if not os.path.exists(backup_path):
                import shutil
                shutil.copy2(icon_file, backup_path)

            # 打开图片
            with Image.open(icon_file) as img:
                # 获取原始尺寸
                original_size = img.size

                # 调整大小，使用高质量的重采样
                resized_img = img.resize(target_size, Image.Resampling.LANCZOS)

                # 保存调整后的图片，覆盖原文件
                resized_img.save(icon_file)

                print(
                    f"已将 {filename} 从 {original_size[0]}x{original_size[1]} 调整为 {target_size[0]}x{target_size[1]} 像素")
        except Exception as e:
            print(f"处理 {filename} 时出错: {e}")

    print("所有图标调整完成！")
    print(f"原始图标已备份至: {backup_dir}")


if __name__ == "__main__":
    # 设置图标目录路径
    icons_directory = "labelme/icons"

    # 调整图标大小为32x32像素
    resize_icons(icons_directory, (32, 32))
