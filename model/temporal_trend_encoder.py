"""
Periodic Trend Temporal Encoding Module for Temporal Knowledge Graph Completion

This module incorporates periodic trends (周期趋势) into temporal knowledge graph modeling,
capturing both linear evolution trends and cyclical patterns.

Key innovations:
1. Mixed periodic-trend time embedding combining linear and cosine components
2. Temporal gating mechanism for entity and relation embeddings
3. Angle-constrained loss to maintain geometric consistency
4. Temporal contrastive learning across time steps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional


class PeriodicTrendTimeEmbedding(nn.Module):
    """
    Generates periodic+trend time embeddings for entities and relations
    
    Formula:
        timevec = alpha * alpha_t * t + (1-alpha) * cos(2*pi*beta_t*t)
        where:
        - First term captures linear long-term evolution
        - Second term captures cyclical patterns
        - alpha balances both components
    """
    
    def __init__(self, num_entities: int, h_dim: int, alpha: float = 0.5):
        super(PeriodicTrendTimeEmbedding, self).__init__()
        
        self.num_entities = num_entities
        self.h_dim = h_dim
        self.alpha = alpha  # Balance between linear and periodic components
        self.pi = torch.tensor(math.pi)
        
        # Per-entity linear trend parameters (learned)
        self.alpha_t = nn.Parameter(torch.Tensor(num_entities, h_dim))
        nn.init.normal_(self.alpha_t, mean=0, std=0.1)
        
        # Per-entity periodic frequency parameters (learned)
        self.beta_t = nn.Parameter(torch.Tensor(num_entities, h_dim))
        nn.init.normal_(self.beta_t, mean=1.0, std=0.5)
        
        # Temporal attention weight matrix
        self.temporal_w = nn.Parameter(torch.Tensor(h_dim * 2, h_dim))
        nn.init.xavier_uniform_(self.temporal_w)
        
        # Static embedding basis
        self.static_emb = nn.Parameter(torch.Tensor(num_entities, h_dim))
        nn.init.normal_(self.static_emb)
    
    def forward(self, t: torch.Tensor, device: str = 'cuda') -> torch.Tensor:
        """
        Generate time-dependent entity embeddings
        
        Args:
            t: Time step(s), shape (batch_size,) or scalar
            device: Device type
            
        Returns:
            Time-modulated embeddings, shape (num_entities, h_dim)
        """
        t = t.to(device) if isinstance(t, torch.Tensor) else torch.tensor(t, device=device, dtype=torch.float32)
        
        # Linear trend component
        linear_component = self.alpha * self.alpha_t * t.unsqueeze(0).unsqueeze(-1)
        
        # Periodic component (cosine-based)
        periodic_component = (1 - self.alpha) * torch.cos(
            2 * self.pi.to(device) * self.beta_t * t.unsqueeze(0).unsqueeze(-1)
        )
        
        # Combine components
        timevec = linear_component + periodic_component  # (num_entities, h_dim, 1) -> squeeze
        timevec = timevec.squeeze(-1)  # (num_entities, h_dim)
        
        # Concatenate with static embeddings and project
        attn_input = torch.cat([self.static_emb, timevec], dim=1)  # (num_entities, 2*h_dim)
        time_modulated_emb = torch.mm(attn_input, self.temporal_w)  # (num_entities, h_dim)
        
        return time_modulated_emb


class TemporalGatingModule(nn.Module):
    """
    Temporal gating mechanism for adaptive fusion of static and dynamic embeddings
    
    Formula:
        time_weight = sigmoid(W * x + b)
        output = time_weight * dynamic_emb + (1 - time_weight) * static_emb
    """
    
    def __init__(self, h_dim: int, dropout: float = 0.2):
        super(TemporalGatingModule, self).__init__()
        
        self.h_dim = h_dim
        
        # Gating weight and bias
        self.gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))
        nn.init.xavier_uniform_(self.gate_weight, gain=nn.init.calculate_gain('sigmoid'))
        
        self.gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.gate_bias)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, dynamic_emb: torch.Tensor, static_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dynamic_emb: Dynamic embeddings, shape (..., h_dim)
            static_emb: Static embeddings, shape (..., h_dim)
            
        Returns:
            Gated embeddings, shape (..., h_dim)
        """
        # Compute gating coefficients
        gate = torch.sigmoid(torch.mm(dynamic_emb, self.gate_weight) + self.gate_bias)
        
        if self.dropout is not None:
            gate = self.dropout(gate)
        
        # Adaptive fusion
        output = gate * dynamic_emb + (1 - gate) * static_emb
        
        return output


class TemporalSlideWindow(nn.Module):
    """
    Manages sliding window history for temporal sequences
    
    Handles dynamic history length based on position in sequence
    """
    
    def __init__(self, max_history_len: int = 10):
        super(TemporalSlideWindow, self).__init__()
        self.max_history_len = max_history_len
    
    def get_history_window(self, current_idx: int, sequence_len: int) -> Tuple[int, int]:
        """
        Get the start and end indices for history window
        
        Args:
            current_idx: Current position in sequence
            sequence_len: Total sequence length
            
        Returns:
            (start_idx, end_idx) of history window
        """
        start_idx = max(0, current_idx - self.max_history_len)
        end_idx = current_idx
        
        return start_idx, end_idx
    
    def get_time_indices(self, current_idx: int, device: str = 'cuda') -> torch.Tensor:
        """
        Get relative time indices for history window
        
        Args:
            current_idx: Current position
            device: Device type
            
        Returns:
            Tensor of relative time indices
        """
        start_idx, end_idx = self.get_history_window(current_idx, current_idx + 1)
        time_indices = torch.arange(start_idx, end_idx, dtype=torch.float32, device=device)
        
        # Convert to relative distances (more recent = larger values)
        time_indices = time_indices - time_indices[0]
        
        return time_indices


class AngleConstrainedLoss(nn.Module):
    """
    Constrains temporal evolution by limiting the angle between static and dynamic embeddings
    
    Formula:
        loss = hinge_loss(cos(expected_angle) - cosine_similarity)
    """
    
    def __init__(self, angle_degree: float = 10.0, weight: float = 0.1):
        super(AngleConstrainedLoss, self).__init__()
        
        self.angle_degree = angle_degree
        self.angle_rad = torch.tensor(angle_degree * math.pi / 180.0)
        self.weight = weight
    
    def forward(self, static_emb: torch.Tensor, dynamic_embs: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            static_emb: Static embeddings, shape (num_entities, h_dim)
            dynamic_embs: List of dynamic embeddings across time steps
            
        Returns:
            Total angle constraint loss
        """
        loss = 0.0
        
        for time_step, dynamic_emb in enumerate(dynamic_embs):
            # Expected angle increases linearly with time
            expected_angle = self.angle_rad * (time_step + 1)
            expected_cos = torch.cos(expected_angle)
            
            # Compute actual cosine similarity
            dot_product = torch.sum(static_emb * dynamic_emb, dim=1)
            static_norm = torch.norm(static_emb, p=2, dim=1)
            dynamic_norm = torch.norm(dynamic_emb, p=2, dim=1)
            
            cosine_sim = dot_product / (static_norm * dynamic_norm + 1e-8)
            
            # Hinge loss: penalize when actual angle > expected angle
            margin = expected_cos.to(static_emb.device) - cosine_sim
            margin = torch.clamp(margin, min=0.0)
            
            loss += torch.sum(margin)
        
        return self.weight * loss / len(dynamic_embs) if len(dynamic_embs) > 0 else torch.tensor(0.0)


class TemporalContrastiveLoss(nn.Module):
    """
    Contrastive learning loss across time steps
    
    Encourages alignment between global history embeddings and local dynamic embeddings
    Uses InfoNCE (Noise Contrastive Estimation) loss
    """
    
    def __init__(self, h_dim: int, temperature: float = 0.07, dropout: float = 0.2):
        super(TemporalContrastiveLoss, self).__init__()
        
        self.h_dim = h_dim
        self.temperature = temperature
        
        # Projection head for embeddings
        self.proj_head = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, 128)
        )
        
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, global_hist_embs: torch.Tensor, local_dynamic_embs: torch.Tensor,
                triplets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            global_hist_embs: Global history embeddings, shape (num_entities, h_dim)
            local_dynamic_embs: Local dynamic embeddings, shape (num_entities, h_dim)
            triplets: Triple indices, shape (batch_size, 3)
            
        Returns:
            Contrastive loss
        """
        # Project embeddings
        global_proj = F.normalize(self.proj_head(global_hist_embs), dim=1)  # (num_entities, 128)
        local_proj = F.normalize(self.proj_head(local_dynamic_embs), dim=1)   # (num_entities, 128)
        
        # Extract subject and object embeddings
        subject_ids = triplets[:, 0]
        object_ids = triplets[:, 2]
        
        global_subject = global_proj[subject_ids]  # (batch_size, 128)
        local_subject = local_proj[subject_ids]    # (batch_size, 128)
        
        global_object = global_proj[object_ids]    # (batch_size, 128)
        local_object = local_proj[object_ids]      # (batch_size, 128)
        
        # Concatenate subject and object
        global_query = torch.cat([global_subject, global_object], dim=1)  # (batch_size, 256)
        local_query = torch.cat([local_subject, local_object], dim=1)     # (batch_size, 256)
        
        # Compute similarity matrix
        similarity = torch.mm(global_query, local_query.t()) / self.temperature  # (batch_size, batch_size)
        
        # Create labels (diagonal elements should have highest similarity)
        labels = torch.arange(len(triplets), device=similarity.device)
        
        # Contrastive loss
        loss = self.loss_fn(similarity, labels)
        
        return loss


class TemporalTrendEncoder(nn.Module):
    """
    Complete temporal trend encoding module combining all components
    
    Integrates:
    1. Periodic trend time embeddings
    2. Temporal gating
    3. Sliding window management
    4. Angle constraint loss
    5. Contrastive learning
    """
    
    def __init__(self, num_entities: int, h_dim: int, 
                 max_history_len: int = 10,
                 alpha_balance: float = 0.5,
                 angle_degree: float = 10.0,
                 temperature: float = 0.07,
                 use_gating: bool = True,
                 use_angle_loss: bool = True,
                 use_contrastive_loss: bool = True,
                 dropout: float = 0.2):
        super(TemporalTrendEncoder, self).__init__()
        
        self.num_entities = num_entities
        self.h_dim = h_dim
        self.use_gating = use_gating
        self.use_angle_loss = use_angle_loss
        self.use_contrastive_loss = use_contrastive_loss
        
        # Core modules
        self.time_embedding = PeriodicTrendTimeEmbedding(num_entities, h_dim, alpha=alpha_balance)
        
        if use_gating:
            self.temporal_gating = TemporalGatingModule(h_dim, dropout)
        
        self.slide_window = TemporalSlideWindow(max_history_len)
        
        if use_angle_loss:
            self.angle_loss = AngleConstrainedLoss(angle_degree, weight=0.1)
        
        if use_contrastive_loss:
            self.contrastive_loss = TemporalContrastiveLoss(h_dim, temperature, dropout)
    
    def get_temporal_emb(self, t: torch.Tensor, device: str = 'cuda') -> torch.Tensor:
        """Get time-modulated embeddings for entities"""
        return self.time_embedding(t, device)
    
    def apply_temporal_gating(self, dynamic_emb: torch.Tensor, static_emb: torch.Tensor) -> torch.Tensor:
        """Apply temporal gating between dynamic and static embeddings"""
        if not self.use_gating:
            return dynamic_emb
        return self.temporal_gating(dynamic_emb, static_emb)
    
    def get_history_window(self, current_idx: int, sequence_len: int) -> Tuple[int, int]:
        """Get history window indices"""
        return self.slide_window.get_history_window(current_idx, sequence_len)
    
    def compute_angle_loss(self, static_emb: torch.Tensor, dynamic_embs: List[torch.Tensor]) -> torch.Tensor:
        """Compute angle constraint loss"""
        if not self.use_angle_loss:
            return torch.tensor(0.0, device=static_emb.device)
        return self.angle_loss(static_emb, dynamic_embs)
    
    def compute_contrastive_loss(self, global_hist_embs: torch.Tensor, 
                                local_dynamic_embs: torch.Tensor,
                                triplets: torch.Tensor) -> torch.Tensor:
        """Compute temporal contrastive loss"""
        if not self.use_contrastive_loss:
            return torch.tensor(0.0, device=global_hist_embs.device)
        return self.contrastive_loss(global_hist_embs, local_dynamic_embs, triplets)
