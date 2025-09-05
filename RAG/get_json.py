import json
import sys

def generate_json(num_entries, filename="test3.json"):
    """
    生成指定条目数的JSON文件，键名始终为2位数字（不足补零）
    :param num_entries: 条目数量（整数）
    :param filename: 输出文件名
    """
    # 强制键名为2位数字（00, 01, ..., 99, 100, 101, ..., 199, ...）
    data = {
        str(i).zfill(2): ""  # 始终补零到2位
        for i in range(num_entries)
    }
    
    # 写入JSON文件
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"已生成 {num_entries} 条数据到 {filename}，键名格式：00, 01, ..., 99, 100, 101, ...")

if __name__ == "__main__":
    try:
        # 从命令行参数获取条目数
        num = int(sys.argv[1]) if len(sys.argv) > 1 else 200  # 默认61条
        generate_json(num)
    except ValueError:
        print("错误：请输入有效的整数参数")