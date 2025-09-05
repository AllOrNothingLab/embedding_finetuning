import torch
import torch_mlu 
from torch_mlu.utils.model_transfer import transfer
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# ================= 0. 设置MLU设备 =================
# 设置可见的MLU设备（使用前4个MLU卡）
device_ids = [0, 1, 2, 3]
print(f"使用的MLU设备: {device_ids}")

# ================= 1. 定义简单模型 =================
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer = nn.Linear(10, 5)
    
    def forward(self, x):
        print(f"输入设备: {x.device}")
        return self.layer(x)

# ================= 2. 创建模型并应用DataParallel =================
model = SimpleModel()
model.to('mlu')  # 将模型移动到MLU设备
dp_model = nn.DataParallel(model, device_ids=device_ids)

print(f"\n模型已封装为DataParallel: {type(dp_model)}")

# ================= 3. 创建测试数据 =================
# 创建在主机内存中的数据
input_data = torch.randn(4, 10)  # batch_size=4

# 将数据移动到MLU设备
input_var = input_data.to('mlu')
print(f"输入数据设备: {input_var.device}")

# ================= 4. 执行前向传播测试 =================
print("\n===== 执行前向传播 =====")
output = dp_model(input_var)
print(f"输出设备: {output.device}")
print(f"输出形状: {output.shape}")

# ================= 5. 验证输出一致性 =================
# 对比单卡和多卡输出
single_model = SimpleModel().to('mlu')
single_output = single_model(input_data.to('mlu'))

print("\n===== 验证输出一致性 =====")
print(f"单卡输出: {single_output[0].detach().cpu().numpy()}")
print(f"多卡输出: {output[0].detach().cpu().numpy()}")
print(f"最大差异: {(single_output - output).abs().max().item():.6f}")

# ================= 6. 简单反向传播测试 =================
print("\n===== 执行反向传播 =====")
target = torch.randint(0, 5, (4,)).to('mlu')
criterion = nn.CrossEntropyLoss()
loss = criterion(output, target)
loss.backward()

print(f"损失值: {loss.item():.4f}")
print("测试完成！")