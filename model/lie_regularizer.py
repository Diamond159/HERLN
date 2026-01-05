"""
Lie Group Regularization for HERLN
简化版：专注于有效的正则化技术

核心思想：
1. L2 范数正则化（防止过拟合）
2. 关系嵌入正交约束（增强区分性）
3. 实体对一致性（同关系下的实体对应相似）
"""

import torch
import torch.nn as nn
from torch import Tensor


class RelationAwareLieRegularizer(nn.Module):
    """
    简化版正则化器
    
    专注于三个有效技术：
    1. L2 范数正则化（防止过拟合）
    2. 关系嵌入正交约束（增强区分性）
    3. 实体对一致性（同关系下的实体对应相似）
    """
    
    def __init__(self, dim: int, p: float = 2.0, 
                 ent_weight: float = 0.0001, 
                 rel_weight: float = 0.001,
                 pair_weight: float = 0.001):
        super().__init__()
        self.dim = dim
        self.p = p
        self.ent_weight = ent_weight
        self.rel_weight = rel_weight
        self.pair_weight = pair_weight
    
    def forward(self, ent_emb: Tensor, rel_emb: Tensor, 
                triplets: Tensor = None) -> Tensor:
        """
        Args:
            ent_emb: 实体嵌入 [num_entities, dim] 或批次实体 [batch_entities, dim]
            rel_emb: 关系嵌入 [num_relations, dim]
            triplets: 三元组 [batch, 3]，可选，用于计算实体对正则化
        
        Returns:
            regularization_loss: 标量损失
        """
        loss = torch.tensor(0.0, device=ent_emb.device)
        
        # 1. L2 正则化（只对批次涉及的实体）
        if self.ent_weight > 0 and triplets is not None:
            involved_entities = torch.unique(torch.cat([triplets[:, 0], triplets[:, 2]]))
            batch_ent = ent_emb[involved_entities]
            ent_l2 = torch.norm(batch_ent, p=2) / (batch_ent.numel() + 1e-8)
            loss = loss + self.ent_weight * ent_l2
        
        # 2. 关系嵌入正交约束
        # 鼓励不同关系有不同的表示方向
        if self.rel_weight > 0 and rel_emb.shape[0] > 1:
            # 归一化关系嵌入
            rel_norm = rel_emb / (torch.norm(rel_emb, dim=-1, keepdim=True) + 1e-8)
            # 计算关系间的相似度矩阵
            sim_matrix = torch.mm(rel_norm, rel_norm.t())
            # 减去对角线（自相似度）
            eye = torch.eye(sim_matrix.shape[0], device=sim_matrix.device)
            off_diag = sim_matrix - eye
            # 最小化非对角线元素（鼓励正交）
            ortho_loss = torch.abs(off_diag).mean()
            loss = loss + self.rel_weight * ortho_loss
        
        # 3. 批次内对比学习（简化版）
        if triplets is not None and self.pair_weight > 0 and len(triplets) > 1:
            s_emb = ent_emb[triplets[:, 0]]
            o_emb = ent_emb[triplets[:, 2]]
            rel_idx = triplets[:, 1]
            
            # 实体对表示
            pair_repr = torch.cat([s_emb, o_emb], dim=-1)  # [batch, 2*dim]
            pair_norm = pair_repr / (torch.norm(pair_repr, dim=-1, keepdim=True) + 1e-8)
            
            # 同关系的实体对应该相似
            batch_size = len(triplets)
            same_rel_mask = (rel_idx.unsqueeze(0) == rel_idx.unsqueeze(1)).float()
            # 去掉对角线
            same_rel_mask = same_rel_mask - torch.eye(batch_size, device=same_rel_mask.device)
            
            if same_rel_mask.sum() > 0:
                # 计算实体对相似度
                pair_sim = torch.mm(pair_norm, pair_norm.t())
                # 同关系的实体对应该更相似
                same_rel_sim = (pair_sim * same_rel_mask).sum() / (same_rel_mask.sum() + 1e-8)
                # 最大化同关系相似度（取负作为损失）
                contrast_loss = -same_rel_sim
                loss = loss + self.pair_weight * contrast_loss
        
        return loss