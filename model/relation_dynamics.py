"""
ERD-Net全局关系动态建模模块
基于ERD-Net论文的Global Relation Dynamics创新
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

class GlobalRelationDynamics(nn.Module):
    """全局关系动态建模模块，基于ERD-Net创新"""
    
    def __init__(self, num_ents, num_rels, h_dim, device):
        super(GlobalRelationDynamics, self).__init__()
        self.num_ents = num_ents
        self.num_rels = num_rels
        self.h_dim = h_dim
        # 修夏设备兼容性问题
        if isinstance(device, int):
            self.device = f'cuda:{device}'
        else:
            self.device = device if device in ['cpu', 'cuda'] or device.startswith('cuda:') else 'cpu'
        
        # GRU用于关系动态演化
        self.rel_gru = nn.GRUCell(2 * h_dim, h_dim)
        
        # 关系-实体交互计算网络
        self.rel_ent_interaction = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(h_dim, h_dim)
        )
        
        # 注意力机制用于实体聚合
        self.attention = nn.Linear(h_dim, 1)
        
        # 初始化隐状态
        self.register_buffer('rel_h0', torch.zeros(num_rels, h_dim))
        self._reset_parameters()
        
    def _reset_parameters(self):
        """初始化参数"""
        for module in self.rel_ent_interaction:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.attention.weight)
        nn.init.zeros_(self.attention.bias)
        
    def _extract_relation_entities(self, graph, rel_type):
        """从图中提取与特定关系相关的实体"""
        related_entities = set()
        
        if hasattr(graph, 'edges') and len(graph.edges()[0]) > 0:
            src, dst = graph.edges()
            edge_types = graph.edata.get('type', torch.zeros(len(src), dtype=torch.long, device=self.device))
            
            # 获取该关系类型的边
            mask = (edge_types == rel_type) | (edge_types == rel_type + self.num_rels)  # 包括反向关系
            if mask.sum() > 0:
                related_entities.update(src[mask].cpu().tolist())
                related_entities.update(dst[mask].cpu().tolist())
                
        return list(related_entities)
        
    def forward(self, graphs, rel_embs, ent_embs, time_idx=0):
        """
        Args:
            graphs: 时序图列表
            rel_embs: 关系嵌入 [num_rels, h_dim]
            ent_embs: 实体嵌入 [num_ents, h_dim]
            time_idx: 当前时间步
        """
        try:
            if not graphs:
                return rel_embs
                
            # 保存原始关系嵌入作为备份
            original_rel_embs = rel_embs.clone().detach()
            
            # 计算关系-实体交互信息（简化版）
            rel_ent_input = torch.zeros_like(rel_embs, device=self.device)
            
            # 简化处理：直接使用平均实体特征
            if len(ent_embs) > 0:
                avg_ent_feature = torch.mean(ent_embs, dim=0, keepdim=True)
                # 确保rel_ent_input的维度与rel_embs匹配
                if rel_embs.size(0) == self.num_rels:
                    rel_ent_input = avg_ent_feature.expand(self.num_rels, -1)
                else:
                    # 处理num_rels*2的情况（包括反向关系）
                    rel_ent_input = avg_ent_feature.expand(rel_embs.size(0), -1)
                rel_ent_input = self.rel_ent_interaction(rel_ent_input)
            
            # 组合原始关系嵌入和交互信息
            rel_input = torch.cat([rel_embs, rel_ent_input], dim=1)  # [num_rels, 2*h_dim]
            
            # GRU更新关系表示
            if time_idx == 0:
                self.rel_h0 = self.rel_gru(rel_input, rel_embs)
                self.rel_h0 = F.normalize(self.rel_h0, dim=1)
            else:
                self.rel_h0 = self.rel_gru(rel_input, self.rel_h0)
                self.rel_h0 = F.normalize(self.rel_h0, dim=1)
            
            # 检查结果是否有效
            if torch.isnan(self.rel_h0).any() or torch.isinf(self.rel_h0).any():
                print("Warning: Invalid values in relation dynamics, using original embeddings")
                return original_rel_embs
            
            return self.rel_h0
            
        except Exception as e:
            print(f"Error in relation dynamics forward: {e}")
            return rel_embs  # 返回原始嵌入
        
    def reset_hidden_state(self):
        """重置隐状态"""
        self.rel_h0.zero_()