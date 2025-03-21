import osam

# 打印已注册的模型类型
print("已注册的模型类型:")
for model_type in osam.apis.registered_model_types:
    print(f"  - {model_type.__name__}")

# 检查get_model_type_by_name函数的文档
print("\nget_model_type_by_name函数文档:")
print(osam.apis.get_model_type_by_name.__doc__)

# 测试几种常见模型名称
test_model_names = [
    "sam:latest",
    "sam:300m",
    "sam:100m",
    "efficientsam:latest",
    "efficientsam:10m",
    "sam2:large",
    "sam2:latest",
    "sam2:base",
    "sam2:base_plus",
    "sam2:small",
    "sam2:tiny"
]

print("\n尝试获取各个模型类型:")
for model_name in test_model_names:
    try:
        model_type = osam.apis.get_model_type_by_name(model_name)
        print(f"  - {model_name}: 对应 {model_type.__name__}")
    except Exception as e:
        print(f"  - {model_name}: 错误 - {e}")
