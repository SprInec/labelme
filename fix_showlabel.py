#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 读取app.py文件
with open('labelme/app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")

# 找到包含showLabelNames的行
modified = False
for i, line in enumerate(lines):
    if 'showLabelNames=showLabelNames,' in line:
        print(f"Found line {i+1}: {line.strip()}")
        # 删除这一行
        lines[i] = '            # showLabelNames line removed to fix error\n'
        print(f"Line {i+1} modified")
        modified = True

if not modified:
    print("未找到 showLabelNames=showLabelNames 行")
    # 检查是否存在第二个showLabelNames定义
    found_showLabelNames = False
    for i, line in enumerate(lines):
        if '# 添加显示标签名称的动作' in line:
            found_showLabelNames = True
            print(f"Found showLabelNames definition at line {i+1}")
            if found_showLabelNames and i > 1000:  # 如果在1000行以后找到定义
                # 删除整个代码块
                end_line = i + 10  # 假设代码块有10行
                for j in range(i, min(end_line, len(lines))):
                    if lines[j].strip() == "":
                        end_line = j
                        break

                print(f"Removing lines {i+1} to {end_line}")
                for j in range(i, end_line):
                    lines[j] = "# " + lines[j]  # 注释掉这些行
                modified = True
                break

# 写回文件
if modified:
    with open('labelme/app.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print("修复完成")
else:
    print("未修改文件")
