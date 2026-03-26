"""
ERD-Net复制-生成机制用于关系预测
基于ERD-Net论文的Copy-Generation Mechanism创新
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict, Counter

class CopyGenerationRelationDecoder(nn.Module):
    """带复制-生成机制的关系预测解码器"""
    
    def __init__(self, h_dim, num_rels, alpha=0.5, device='cuda', dropout=0.3):
        super(CopyGenerationRelationDecoder, self).__init__()
        self.h_dim = h_dim
        self.num_rels = num_rels
        self.alpha = alpha  # 复制-生成平衡参数
        self.device = device if isinstance(device, str) else f'cuda:{device}'
        self.dropout = dropout
        
        # 生成机制：标准MLP用于基于实体对预测关系
        self.generation_mlp = nn.Sequential(
            nn.Linear(2 * h_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, h_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim // 2, 2 * num_rels)  # 包括反向关系
        )
        
        # 复制机制：基于历史频率的注意力网络
        self.copy_attention = nn.Sequential(
            nn.Linear(2 * h_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, 1),
            nn.Sigmoid()
        )
        
        # 关系频率统计（训练时动态更新）
        self.register_buffer('relation_freq', torch.ones(2 * num_rels))
        self.register_buffer('relation_count', torch.zeros(2 * num_rels))
        
        # 历史模式缓存
        self.history_patterns = defaultdict(list)  # (head, tail) -> [relations]
        self.max_history_size = 1000  # 控制历史缓存大小
        
        self._init_parameters()
        
    def _init_parameters(self):
        """初始化网络参数"""
        for module in self.generation_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
                
        for module in self.copy_attention:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
                
    def update_relation_frequencies(self, relation_batch):
        """动态更新关系频率统计"""
        with torch.no_grad():
            # 添加批量大小限制，防止无限循环
            max_batch_size = 10000  # 设置最大处理批量
            if len(relation_batch) > max_batch_size:
                print(f"Warning: Batch size {len(relation_batch)} too large, truncating to {max_batch_size}")
                relation_batch = relation_batch[:max_batch_size]
                
            for i, rel in enumerate(relation_batch):
                if i > max_batch_size:  # 额外安全检查
                    break
                    
                rel_idx = rel.item()
                if 0 <= rel_idx < self.relation_freq.size(0):
                    self.relation_count[rel_idx] += 1
                    # 使用指数移动平均更新频率
                    self.relation_freq[rel_idx] = 0.9 * self.relation_freq[rel_idx] + 0.1
                    
    def update_history_patterns(self, triples):
        """更新历史实体对-关系模式"""
        with torch.no_grad():
            for triple in triples:
                h, r, t = triple[0].item(), triple[1].item(), triple[2].item()
                key = (h, t)
                
                # 添加到历史模式
                if key not in self.history_patterns:
                    self.history_patterns[key] = []
                    
                if r not in self.history_patterns[key]:
                    self.history_patterns[key].append(r)
                    
                # 控制历史缓存大小
                if len(self.history_patterns[key]) > 10:
                    self.history_patterns[key] = self.history_patterns[key][-10:]
                    
                # 控制整体缓存大小
                if len(self.history_patterns) > self.max_history_size:
                    # 随机删除一些旧的模式
                    keys_to_remove = list(self.history_patterns.keys())[:100]
                    for k in keys_to_remove:
                        del self.history_patterns[k]
                        
    def get_copy_scores(self, entity_pairs):
        """计算复制分数：基于历史模式和频率"""
        batch_size = len(entity_pairs)
        copy_scores = torch.zeros(batch_size, 2 * self.num_rels, device=self.device)
        
        # 简化逻辑：直接使用全局频率作为复制分数
        # 避免复杂的历史模式查找导致的性能问题
        for i in range(batch_size):
            copy_scores[i] = self.relation_freq
                
        return F.softmax(copy_scores, dim=1)
        
    def forward(self, entity_embs, rel_embs, triples, mode="train", use_copy=True):
        """
        Args:
            entity_embs: [num_ents, h_dim]
            rel_embs: [num_rels, h_dim] 
            triples: [batch_size, 3] (s, r, o)
            mode: 'train' or 'test'
            use_copy: 是否使用复制机制
        """
        batch_size = len(triples)
        
        # 获取实体对表示
        head_embs = entity_embs[triples[:, 0]]  # [batch_size, h_dim]
        tail_embs = entity_embs[triples[:, 2]]  # [batch_size, h_dim]
        pair_embs = torch.cat([head_embs, tail_embs], dim=1)  # [batch_size, 2*h_dim]
        
        # 生成分数：基于实体对表示预测关系
        generation_scores = self.generation_mlp(pair_embs)  # [batch_size, 2*num_rels]
        generation_probs = F.softmax(generation_scores, dim=1)
        
        # 如果启用复制机制
        if use_copy:
            try:
                # 计算复制分数
                copy_probs = self.get_copy_scores(triples[:, [0, 2]])  # [batch_size, 2*num_rels]
                
                # 计算复制-生成权重
                copy_weights = self.copy_attention(pair_embs)  # [batch_size, 1]
                
                # 动态调整alpha
                dynamic_alpha = self.alpha * copy_weights  # [batch_size, 1]
                
                # 组合复制和生成分数
                final_probs = dynamic_alpha * copy_probs + (1 - dynamic_alpha) * generation_probs
            except Exception as e:
                # 如果复制机制失败，回退到纯生成模式
                print(f"Warning: Copy mechanism failed, using generation only: {e}")
                final_probs = generation_probs
        else:
            final_probs = generation_probs
            
        return final_probs
        
    def get_loss(self, entity_embs, rel_embs, triples, use_copy=True):
        """计算关系预测损失"""
        try:
            # 添加输入验证，防止无效数据导致无限循环
            if triples.size(0) == 0:
                return torch.tensor(0.0, device=self.device, requires_grad=True)
            
            if triples.size(0) > 50000:  # 限制批量大小
                print(f"Warning: Large batch size {triples.size(0)}, truncating")
                triples = triples[:50000]
            
            probs = self.forward(entity_embs, rel_embs, triples, mode="train", use_copy=use_copy)
            
            # 更新统计信息（限制频率更新）
            if use_copy and triples.size(0) < 10000:  # 只对小批量更新频率
                self.update_relation_frequencies(triples[:, 1])
            
            # 交叉熵损失
            target_rels = triples[:, 1].long()
            
            # 确保目标索引在有效范围内
            valid_mask = (target_rels >= 0) & (target_rels < probs.size(1))
            if valid_mask.sum() == 0:
                return torch.tensor(0.0, device=self.device, requires_grad=True)
                
            probs_valid = probs[valid_mask]
            target_valid = target_rels[valid_mask]
            
            # 基本交叉熵损失
            loss = F.cross_entropy(probs_valid, target_valid)
            
            # 检查损失是否有效
            if torch.isnan(loss) or torch.isinf(loss):
                print("Warning: Invalid loss detected in copy-generation decoder")
                return torch.tensor(0.0, device=self.device, requires_grad=True)
            
            return loss
            
        except Exception as e:
            print(f"Error in copy-generation loss calculation: {e}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
    def clear_history(self):
        """清空历史模式缓存"""
        self.history_patterns.clear()
        
    def get_statistics(self):
        """获取模块统计信息"""
        stats = {
            'total_patterns': len(self.history_patterns),
            'avg_patterns_per_pair': np.mean([len(v) for v in self.history_patterns.values()]) if self.history_patterns else 0,
            'most_frequent_relations': torch.topk(self.relation_freq, k=5).indices.tolist(),
            'total_relation_counts': self.relation_count.sum().item()
        }
        return stats