#!/usr/bin/env python3
"""
测试ERD-Net创新功能的验证脚本
验证全局关系动态建模和复制-生成机制
"""

import sys
import os
import numpy as np
import torch

sys.path.append('src')
sys.path.append('model')

def test_erd_net_features():
    """测试ERD-Net创新功能"""
    print("🧪 ERD-Net创新功能验证开始")
    print("="*60)
    
    success = True
    
    # 测试1: 全局关系动态建模模块
    try:
        from model.relation_dynamics import GlobalRelationDynamics
        
        # 创建测试参数
        num_ents = 100
        num_rels = 10
        h_dim = 64
        device = 'cpu'
        
        # 初始化模块
        rel_dynamics = GlobalRelationDynamics(num_ents, num_rels, h_dim, device)
        print("✅ GlobalRelationDynamics模块导入和初始化成功")
        
        # 创建测试数据
        rel_embs = torch.randn(num_rels, h_dim)
        ent_embs = torch.randn(num_ents, h_dim)
        
        # 测试forward方法
        updated_rel_embs = rel_dynamics([], rel_embs, ent_embs, time_idx=0)
        if updated_rel_embs.shape == (num_rels, h_dim):
            print("✅ 全局关系动态更新功能正常")
        else:
            print("❌ 全局关系动态输出形状错误")
            success = False
            
    except ImportError as e:
        print(f"❌ GlobalRelationDynamics导入失败: {e}")
        success = False
    except Exception as e:
        print(f"❌ GlobalRelationDynamics测试失败: {e}")
        success = False
    
    # 测试2: 复制-生成机制
    try:
        from model.copy_generation_decoder import CopyGenerationRelationDecoder
        
        # 创建测试参数
        h_dim = 64
        num_rels = 10
        alpha = 0.5
        device = 'cpu'
        
        # 初始化模块
        copy_gen_decoder = CopyGenerationRelationDecoder(h_dim, num_rels, alpha, device)
        print("✅ CopyGenerationRelationDecoder模块导入和初始化成功")
        
        # 创建测试数据
        entity_embs = torch.randn(100, h_dim)
        rel_embs = torch.randn(num_rels, h_dim)
        triples = torch.randint(0, 100, (32, 3))  # batch_size=32
        
        # 测试forward方法
        probs = copy_gen_decoder.forward(entity_embs, rel_embs, triples, mode="train")
        if probs.shape == (32, 2 * num_rels):
            print("✅ 复制-生成机制forward功能正常")
        else:
            print("❌ 复制-生成机制输出形状错误")
            success = False
            
        # 测试损失计算
        loss = copy_gen_decoder.get_loss(entity_embs, rel_embs, triples)
        if isinstance(loss, torch.Tensor) and loss.dim() == 0:
            print("✅ 复制-生成机制损失计算功能正常")
        else:
            print("❌ 复制-生成机制损失计算错误")
            success = False
            
    except ImportError as e:
        print(f"❌ CopyGenerationRelationDecoder导入失败: {e}")
        success = False
    except Exception as e:
        print(f"❌ CopyGenerationRelationDecoder测试失败: {e}")
        success = False
    
    # 测试3: 配置参数检查
    try:
        from src.config import args
        
        # 检查新增的配置参数
        erd_params = [
            'use_relation_dynamics', 'use_copy_generation', 'copy_gen_alpha',
            'relation_dynamics_lr', 'two_stage_training', 'pretrain_epochs',
            'freeze_entity_embs', 'freeze_relation_embs'
        ]
        
        missing_params = []
        for param in erd_params:
            if not hasattr(args, param):
                missing_params.append(param)
        
        if not missing_params:
            print("✅ 所有ERD-Net配置参数已正确添加")
        else:
            print(f"❌ 缺失配置参数: {missing_params}")
            success = False
            
    except ImportError as e:
        print(f"❌ 配置参数检查失败: {e}")
        success = False
    
    return success

def print_usage_examples():
    """打印使用示例"""
    print("\n" + "="*60)
    print("📋 ERD-Net创新功能使用示例:")
    print("="*60)
    
    print("\n🚀 启用全局关系动态建模:")
    print("python src/main.py -d ICEWS14s --relation-prediction --use-relation-dynamics --encoder uvrgcn")
    
    print("\n🔄 启用复制-生成机制:")
    print("python src/main.py -d ICEWS14s --relation-prediction --use-copy-generation --copy-gen-alpha 0.3")
    
    print("\n📚 启用两阶段训练:")
    print("python src/main.py -d ICEWS14s --relation-prediction --two-stage-training --pretrain-epochs 15 --freeze-relation-embs")
    
    print("\n⚡ 完整ERD-Net功能组合:")
    print("python src/main.py -d ICEWS14s \\")
    print("  --relation-prediction \\")
    print("  --use-relation-dynamics \\") 
    print("  --use-copy-generation --copy-gen-alpha 0.3 \\")
    print("  --two-stage-training --pretrain-epochs 20 \\")
    print("  --freeze-entity-embs \\")
    print("  --encoder uvrgcn --decoder convtranse")
    
    print("\n🔧 与现有功能组合:")
    print("python src/main.py -d ICEWS14s \\")
    print("  --relation-prediction \\")
    print("  --use-relation-dynamics \\")
    print("  --use-copy-generation \\")
    print("  --alternating-training --inverse-training-ratio 0.3 \\")
    print("  --use-lie-reg --lie-rel-weight 0.01 \\")
    print("  --enable-line-graph \\")
    print("  --encoder uvrgcn")
    
    print("\n💡 参数说明:")
    print("  --use-relation-dynamics     : 启用ERD-Net全局关系动态建模")
    print("  --use-copy-generation       : 启用复制-生成机制用于关系预测")
    print("  --copy-gen-alpha           : 复制-生成平衡参数 (0.0-1.0，默认0.5)")
    print("  --relation-dynamics-lr     : 关系动态组件学习率 (默认0.001)")
    print("  --two-stage-training       : 启用两阶段训练策略")
    print("  --pretrain-epochs          : 预训练轮数 (默认20)")
    print("  --freeze-entity-embs       : 第二阶段冻结实体嵌入")
    print("  --freeze-relation-embs     : 第二阶段冻结关系嵌入")
    
    print("\n📊 推荐配置:")
    print("  - 关系预测重点: --copy-gen-alpha 0.3 (更多生成，少复制)")
    print("  - 频繁关系处理: --copy-gen-alpha 0.7 (更多复制，少生成)")
    print("  - 稳定训练: --two-stage-training --pretrain-epochs 20")
    print("  - 动态学习: --use-relation-dynamics --relation-dynamics-lr 0.001")

if __name__ == "__main__":
    print("🔬 ERD-Net创新功能集成验证")
    print("="*60)
    
    success = test_erd_net_features()
    
    if success:
        print("\n🎉 所有ERD-Net功能验证通过！")
        print_usage_examples()
    else:
        print("\n❌ 部分功能验证失败，请检查实现")
        print("\n🔧 可能的解决方案:")
        print("1. 检查模块导入路径")
        print("2. 确认所有依赖文件已创建")
        print("3. 验证配置参数正确添加")