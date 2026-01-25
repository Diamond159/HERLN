#!/usr/bin/env python3
"""
验证交替正向-反向训练功能的测试脚本
"""

import sys
import os
sys.path.append('src')

def test_alternating_training():
    """测试交替训练功能"""
    try:
        # 测试导入
        from utils import create_inverse_triples
        from config import args
        import numpy as np
        
        print("✅ 交替训练模块导入成功")
        
        # 测试配置参数
        has_alternating = hasattr(args, 'alternating_training')
        has_inverse_ratio = hasattr(args, 'inverse_training_ratio')
        has_inverse_weight = hasattr(args, 'inverse_loss_weight')
        
        print(f"🔧 配置参数检查:")
        print(f"  - alternating_training: {'✅' if has_alternating else '❌'}")
        print(f"  - inverse_training_ratio: {'✅' if has_inverse_ratio else '❌'}")
        print(f"  - inverse_loss_weight: {'✅' if has_inverse_weight else '❌'}")
        
        # 测试反向三元组创建
        test_triples = np.array([
            [0, 0, 1],  # (head=0, relation=0, tail=1)
            [1, 1, 2],  # (head=1, relation=1, tail=2)
            [2, 0, 0],  # (head=2, relation=0, tail=0)
        ])
        num_rels = 2
        
        print(f"📝 原始三元组: {test_triples.tolist()}")
        
        inverse_triples = create_inverse_triples(test_triples, num_rels)
        print(f"🔄 反向三元组: {inverse_triples.tolist()}")
        
        # 验证反向三元组的正确性
        expected_inverse = np.array([
            [1, 2, 0],  # (tail=1, relation=0+2, head=0)
            [2, 3, 1],  # (tail=2, relation=1+2, head=1)
            [0, 2, 2],  # (tail=0, relation=0+2, head=2)
        ])
        
        if np.array_equal(inverse_triples, expected_inverse):
            print("✅ 反向三元组创建正确")
        else:
            print("❌ 反向三元组创建错误")
            print(f"期望: {expected_inverse.tolist()}")
            return False
        
        # 测试空数组情况
        empty_triples = np.array([]).reshape(0, 3)
        inverse_empty = create_inverse_triples(empty_triples, num_rels)
        if inverse_empty.shape == (0, 3):
            print("✅ 空数组处理正确")
        else:
            print("❌ 空数组处理错误")
            return False
        
        print("🎉 所有交替训练功能测试通过！")
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def print_usage_examples():
    """打印使用示例"""
    print("\n" + "="*60)
    print("📋 使用示例:")
    print("="*60)
    
    print("\n🚀 启用交替训练:")
    print("python src/main.py -d ICEWS14s --alternating-training --inverse-training-ratio 0.3 --encoder uvrgcn")
    
    print("\n🔧 参数说明:")
    print("  --alternating-training      : 启用交替正向-反向训练")
    print("  --inverse-training-ratio    : 反向训练的比例 (0.0-1.0，默认0.5)")
    print("  --inverse-loss-weight       : 反向训练的损失权重 (默认1.0)")
    
    print("\n💡 建议配置:")
    print("  - 初始训练: --inverse-training-ratio 0.2")
    print("  - 防过拟合: --inverse-training-ratio 0.4")
    print("  - 强正则化: --inverse-training-ratio 0.6")
    
    print("\n📊 与其他功能组合:")
    print("python src/main.py -d ICEWS14s \\")
    print("  --alternating-training \\")
    print("  --inverse-training-ratio 0.3 \\")
    print("  --enable-line-graph \\")
    print("  --self-loop --layer-norm \\")
    print("  --encoder uvrgcn --decoder convtranse")

if __name__ == "__main__":
    print("🔄 交替正向-反向训练功能验证")
    print("="*50)
    
    success = test_alternating_training()
    
    if success:
        print_usage_examples()
    else:
        print("\n❌ 验证失败，请检查代码实现")