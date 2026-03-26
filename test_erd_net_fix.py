#!/usr/bin/env python3
"""
ERD-Net集成修复验证脚本
用于测试修复后的ERD-Net功能是否正常工作
"""

import torch
import os
import sys

def test_erd_net_components():
    """测试ERD-Net组件的基本功能"""
    print("=" * 60)
    print("ERD-Net组件修复验证")
    print("=" * 60)
    
    # 设置随机种子
    torch.manual_seed(42)
    
    # 基本参数
    num_ents = 1000
    num_rels = 50
    h_dim = 200
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    print(f"设备: {device}")
    print(f"实体数: {num_ents}, 关系数: {num_rels}, 隐藏维度: {h_dim}")
    print("-" * 60)
    
    # 1. 测试关系动态模块
    print("1. 测试全局关系动态模块...")
    try:
        from model.relation_dynamics import GlobalRelationDynamics
        rel_dynamics = GlobalRelationDynamics(num_ents, num_rels, h_dim, device)
        
        # 模拟输入
        rel_embs = torch.randn(num_rels, h_dim)
        ent_embs = torch.randn(num_ents, h_dim)
        
        if device.startswith('cuda'):
            rel_embs = rel_embs.cuda()
            ent_embs = ent_embs.cuda()
            rel_dynamics = rel_dynamics.cuda()
        
        # 前向传播测试
        updated_rel_embs = rel_dynamics.forward([], rel_embs, ent_embs, time_idx=0)
        
        print(f"   ✅ 关系动态模块初始化成功")
        print(f"   ✅ 输入形状: {rel_embs.shape}")
        print(f"   ✅ 输出形状: {updated_rel_embs.shape}")
        print(f"   ✅ 输出是否有效: {not torch.isnan(updated_rel_embs).any()}")
        
    except Exception as e:
        print(f"   ❌ 关系动态模块测试失败: {e}")
    
    print()
    
    # 2. 测试复制生成解码器
    print("2. 测试复制-生成解码器...")
    try:
        from model.copy_generation_decoder import CopyGenerationRelationDecoder
        copy_decoder = CopyGenerationRelationDecoder(h_dim, num_rels, device=device)
        
        # 模拟输入
        rel_embs = torch.randn(num_rels, h_dim)
        ent_embs = torch.randn(num_ents, h_dim)
        triplets = torch.randint(0, min(num_ents, num_rels), (100, 3))
        
        if device.startswith('cuda'):
            rel_embs = rel_embs.cuda()
            ent_embs = ent_embs.cuda()
            triplets = triplets.cuda()
            copy_decoder = copy_decoder.cuda()
        
        # 前向传播测试
        copy_scores = copy_decoder.forward([], rel_embs, ent_embs, training=False)
        
        print(f"   ✅ 复制-生成解码器初始化成功")
        print(f"   ✅ 关系嵌入形状: {rel_embs.shape}")
        print(f"   ✅ 输出形状: {copy_scores.shape}")
        print(f"   ✅ 输出是否有效: {not torch.isnan(copy_scores).any()}")
        
        # 损失计算测试
        loss = copy_decoder.get_loss([], triplets)
        print(f"   ✅ 损失计算成功: {loss.item():.6f}")
        
    except Exception as e:
        print(f"   ❌ 复制-生成解码器测试失败: {e}")
    
    print()
    
    # 3. 测试RecurrentRGCN模型初始化
    print("3. 测试RecurrentRGCN模型初始化...")
    try:
        from model.rrgcn import RecurrentRGCN
        
        model = RecurrentRGCN(
            decoder_name="convtranse",
            encoder_name="uvrgcn",
            num_ents=num_ents,
            num_rels=num_rels,
            h_dim=h_dim,
            opn="sub",
            sequence_len=10,
            dropout=0.2,
            use_cuda=torch.cuda.is_available(),
            gpu=0,
            entity_prediction=True,
            relation_prediction=True,
            use_relation_dynamics=True,
            use_copy_generation=True
        )
        
        print(f"   ✅ RecurrentRGCN模型初始化成功")
        print(f"   ✅ 使用关系动态: {model.use_relation_dynamics}")
        print(f"   ✅ 使用复制生成: {model.use_copy_generation}")
        print(f"   ✅ 设备: {model.device}")
        
    except Exception as e:
        print(f"   ❌ RecurrentRGCN模型初始化失败: {e}")
    
    print()
    print("=" * 60)
    print("ERD-Net组件测试完成")
    print("=" * 60)

def test_basic_training():
    """测试基本训练流程（不启用ERD-Net）"""
    print("\n测试基本训练流程...")
    
    cmd = "python src/main.py --dataset ICEWS14s --train-history-len 10 --test-history-len 10 --dilate-len 1 --lr 0.001 --n-epochs 2 --entity-prediction --relation-prediction --gpu 0 --ft_epochs 2"
    
    print(f"执行命令: {cmd}")
    print("这将运行2个epoch的基本训练以验证修复...")
    
    return cmd

if __name__ == "__main__":
    # 测试ERD-Net组件
    test_erd_net_components()
    
    # 输出建议的测试命令
    print("\n建议的测试步骤:")
    print("1. 首先运行基本训练（不启用ERD-Net）:")
    basic_cmd = test_basic_training()
    print(f"   {basic_cmd}")
    
    print("\n2. 然后运行启用ERD-Net的训练:")
    erd_cmd = basic_cmd + " --use-relation-dynamics --use-copy-generation"
    print(f"   {erd_cmd}")
    
    print("\n3. 如果ERD-Net训练正常，可以运行完整训练:")
    full_cmd = "python src/main.py --dataset ICEWS14s --train-history-len 10 --test-history-len 10 --dilate-len 1 --lr 0.001 --n-epochs 30 --entity-prediction --relation-prediction --gpu 0 --ft_epochs 5 --use-relation-dynamics --use-copy-generation"
    print(f"   {full_cmd}")