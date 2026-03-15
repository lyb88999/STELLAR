import torch
import torch.nn as nn

class RealTrafficModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=2):
        super().__init__()
        # 存储初始化参数
        self.__init__args__ = (input_dim,)
        self.__init__kwargs__ = {
            'hidden_dim': hidden_dim,
            'num_classes': num_classes
        }
        
        # 恢复STIN-IDS的经典架构 (Simple MLP)
        # 结构: Input(25) -> 64 -> 32 -> Output(2)
        # 适合处理经过特征选择后的低维向量
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 25 -> 64
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2) # 64 -> 32
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes) # 32 -> 2
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        
        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Output
        x = self.fc3(x)
        return x