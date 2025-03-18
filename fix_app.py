#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re

# 读取app.py文件
with open('labelme/app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 检查是否有重复的showLabelNames定义
pattern = r'# 添加显示标签名称的动作\s+showLabelNames = action\('
matches = re.findall(pattern, content)

if len(matches) > 1:
    print(f"发现 {len(matches)} 处 showLabelNames 定义")

    # 找到第二个定义的位置（在__init__方法外）
    first_pos = content.find('# 添加显示标签名称的动作')
    second_pos = content.find('# 添加显示标签名称的动作', first_pos + 1)

    if second_pos != -1:
        # 找到方法定义结束的位置
        end_of_action_block = content.find('\n\n', second_pos)
        if end_of_action_block != -1:
            # 删除第二个定义
            new_content = content[:second_pos] + content[end_of_action_block:]

            # 写回文件
            with open('labelme/app.py', 'w', encoding='utf-8') as f:
                f.write(new_content)

            print("成功移除重复的代码块")
else:
    print("没有发现重复定义")
