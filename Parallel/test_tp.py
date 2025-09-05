# 2D张量并行
# torchrun --nproc_per_node=4 test_tp.py

import torch
import torch_mlu 
from torch_mlu.utils.model_transfer import transfer
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.distributed import ProcessGroup

# 初始化分布式环境（适配MLU，使用cncl后端）
def init_distributed():
    # MLU使用cncl后端（替代GPU的nccl）
    dist.init_process_group(backend='cncl')
    # 设置默认设备为MLU
    torch.set_default_device('mlu')

# 正确划分2D进程组（2x2网格）
def create_2d_process_groups():
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    # 仅支持2x2网格（4个进程）
    assert world_size == 4, "当前仅支持4个进程（2x2网格）"
    grid_size = 2  # 2x2网格
    
    # 确定当前进程的行和列（关键修正）
    # rank0: 行0，列0；rank1: 行0，列1；rank2: 行1，列0；rank3: 行1，列1
    row = rank // grid_size
    col = rank % grid_size
    
    # 创建行进程组（同一行的进程）
    if row == 0:
        row_group = [0, 1]  # 行0包含rank0和1
    else:
        row_group = [2, 3]  # 行1包含rank2和3
    row_pg = dist.new_group(row_group)
    
    # 创建列进程组（同一列的进程）
    if col == 0:
        col_group = [0, 2]  # 列0包含rank0和2
    else:
        col_group = [1, 3]  # 列1包含rank1和3
    col_pg = dist.new_group(col_group)
    
    return grid_size, row, col, row_pg, col_pg

# 2D并行线性层（核心逻辑不变，适配MLU）
class TensorParallel2DLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, grid_size: int, row: int, col: int, row_pg: ProcessGroup, col_pg: ProcessGroup):
        super().__init__()
        self.grid_size = grid_size
        self.row = row
        self.col = col
        self.row_pg = row_pg
        self.col_pg = col_pg
        
        self.in_features_per_rank = in_features // grid_size
        self.out_features_per_rank = out_features // grid_size
        
        # 权重初始化在MLU上
        self.weight = nn.Parameter(torch.randn(
            self.out_features_per_rank,
            self.in_features_per_rank,
            device='mlu'
        ))
        self.bias = nn.Parameter(torch.randn(
            self.out_features_per_rank,
            device='mlu'
        ))

    def forward(self, x: torch.Tensor):
        x = torch.matmul(x, self.weight.t()) + self.bias  # 本地计算
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=self.row_pg)  # 行内聚合
        return x

# 测试主函数（修正输入拆分逻辑）
def main():
    init_distributed()
    rank = dist.get_rank()
    grid_size, row, col, row_pg, col_pg = create_2d_process_groups()
    
    # 全局输入形状：[16, 512]，输出：[16, 256]
    in_features = 512
    out_features = 256
    
    # 创建模型
    model = TensorParallel2DLinear(
        in_features=in_features,
        out_features=out_features,
        grid_size=grid_size,
        row=row,
        col=col,
        row_pg=row_pg,
        col_pg=col_pg
    )
    
    # 生成全局输入（仅rank0创建）
    if rank == 0:
        x_global = torch.randn(16, in_features, device='mlu')  # 全局输入
    else:
        x_global = None
    
    # 输入拆分（关键修正：使用全局进程组而非列进程组，确保src=0在组内）
    # 按列维度拆分为2份（每份256维），发送到对应进程
    x = torch.empty(16, in_features // grid_size, device='mlu')
    # 拆分后的片段：列0进程（0,2）接收第0份，列1进程（1,3）接收第1份
    scatter_list = list(torch.chunk(x_global, grid_size, dim=1)) if rank == 0 else None
    dist.scatter(x, scatter_list, src=0, group=dist.group.WORLD)  # 使用全局进程组
    
    # 前向传播
    output = model(x)
    
    # 打印局部形状
    print(f"Rank {rank} (row {row}, col {col}): 输入形状 {x.shape} → 输出形状 {output.shape}")
    
    # 收集全局输出（仅rank0）
    if rank == 0:
        output_list = [torch.empty(16, out_features // grid_size, device='mlu') for _ in range(4)]
        dist.gather(output, output_list, dst=0)
        output_global = torch.cat(output_list, dim=1)
        print(f"\nRank 0: 全局输出形状 {output_global.shape}（预期[16, 256]）")

if __name__ == "__main__":
    main()