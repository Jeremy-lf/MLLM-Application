## MoE

MoE称为混合专家模型 (Mixed Expert Models)， 是基于 Transformer 架构的模型，主要由两个核心部分组成：

* 稀疏 MoE 层：取代传统 Transformer 的前馈网络（FFN）层。MoE 层由多个“专家”（如 8 个）组成，每个专家是一个独立的神经网络，通常是 FFN；
* 门控网络或路由器：决定哪些 Token 由哪个专家处理。例如，“More”可能被分配给第二个专家，而“Parameters”可能被分配给第一个。有时，一个 Token 甚至可以被多个专家处理。路由方式由可学习的参数控制，并与整个模型一同训练，是 MoE 关键机制之一。
<img width="1080" height="540" alt="image" src="https://github.com/user-attachments/assets/26dcfdc2-a50d-4c4b-96b1-f40d71e43651" />


### Deepseek MOE 架构
和基础 MOE 结构的区别是：

* 更精细地划分专家网络，提升每个专家的专业性，提高知识表达的准确度。
* 引入部分共享专家，减少不同专家间的知识冗余，提升计算效率；所有 tokens 都会经过的共享专家，每个 token 会用计算的 Router 权重，来选择 topK 个专家，然后和共享的专家的输出一起加权求和。

DeepseekMOE 其实是有两类专家的：

* 共享专家（Shared Expert）：1 个共享专家，用于捕捉通用、全局的特征信息。
* 路由专家（Routed Experts）：每个 MoE 层都包含 256 个路由专家，负责精细化处理输入 tokens 的专业特征。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)
        
    def forward(self, x):
        return F.softmax(self.fc(x), dim=1)

class MoE(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Expert(input_dim, output_dim) for _ in range(num_experts)])
        self.gating_network = GatingNetwork(input_dim, num_experts)
        
    def forward(self, x):
        # 计算每个专家的权重
        expert_weights = self.gating_network(x)
        
        # 获取每个专家的输出
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        
        # 加权求和得到最终输出
        output = torch.sum(expert_outputs * expert_weights.unsqueeze(-1), dim=1)
        
        return output

# 示例使用
if __name__ == "__main__":
    # 定义模型参数
    input_dim = 10
    output_dim = 5
    num_experts = 4
    batch_size = 32
    
    # 创建模型
    model = MoE(input_dim, output_dim, num_experts)
    
    # 生成随机输入数据
    x = torch.randn(batch_size, input_dim)
    
    # 前向传播
    output = model(x)
    
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

```
