#!/usr/bin/env python3
"""
简化的ERD-Net修复验证脚本
仅测试模块导入和基本语法，不需要完整依赖
"""

import sys
import os

def test_import_and_syntax():
    """测试ERD-Net模块导入和基本语法"""
    print("=" * 60)
    print("ERD-Net模块导入和语法验证")
    print("=" * 60)
    
    # 添加模型路径
    sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    
    # 1. 测试关系动态模块导入
    print("1. 测试关系动态模块...")
    try:
        # 检查文件是否存在
        relation_dynamics_file = os.path.join('model', 'relation_dynamics.py')
        if os.path.exists(relation_dynamics_file):
            print(f"   ✅ 关系动态文件存在: {relation_dynamics_file}")
            
            # 检查语法
            with open(relation_dynamics_file, 'r', encoding='utf-8') as f:
                content = f.read()
                compile(content, relation_dynamics_file, 'exec')
            print("   ✅ 关系动态模块语法正确")
        else:
            print(f"   ❌ 关系动态文件不存在: {relation_dynamics_file}")
    except SyntaxError as e:
        print(f"   ❌ 关系动态模块语法错误: {e}")
    except Exception as e:
        print(f"   ⚠️  关系动态模块检查异常: {e}")
    
    # 2. 测试复制生成解码器导入
    print("\n2. 测试复制生成解码器...")
    try:
        copy_decoder_file = os.path.join('model', 'copy_generation_decoder.py')
        if os.path.exists(copy_decoder_file):
            print(f"   ✅ 复制生成解码器文件存在: {copy_decoder_file}")
            
            # 检查语法
            with open(copy_decoder_file, 'r', encoding='utf-8') as f:
                content = f.read()
                compile(content, copy_decoder_file, 'exec')
            print("   ✅ 复制生成解码器模块语法正确")
        else:
            print(f"   ❌ 复制生成解码器文件不存在: {copy_decoder_file}")
    except SyntaxError as e:
        print(f"   ❌ 复制生成解码器语法错误: {e}")
    except Exception as e:
        print(f"   ⚠️  复制生成解码器检查异常: {e}")
    
    # 3. 测试主模型文件
    print("\n3. 测试主模型文件...")
    try:
        rrgcn_file = os.path.join('model', 'rrgcn.py')
        if os.path.exists(rrgcn_file):
            print(f"   ✅ RRGCN模型文件存在: {rrgcn_file}")
            
            # 检查语法
            with open(rrgcn_file, 'r', encoding='utf-8') as f:
                content = f.read()
                compile(content, rrgcn_file, 'exec')
            print("   ✅ RRGCN模型语法正确")
        else:
            print(f"   ❌ RRGCN模型文件不存在: {rrgcn_file}")
    except SyntaxError as e:
        print(f"   ❌ RRGCN模型语法错误: {e}")
    except Exception as e:
        print(f"   ⚠️  RRGCN模型检查异常: {e}")
    
    # 4. 测试配置文件
    print("\n4. 测试配置文件...")
    try:
        config_file = os.path.join('src', 'config.py')
        if os.path.exists(config_file):
            print(f"   ✅ 配置文件存在: {config_file}")
            
            # 检查ERD-Net相关配置
            with open(config_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if '--use-relation-dynamics' in content:
                print("   ✅ 关系动态配置参数存在")
            else:
                print("   ❌ 关系动态配置参数缺失")
                
            if '--use-copy-generation' in content:
                print("   ✅ 复制生成配置参数存在")
            else:
                print("   ❌ 复制生成配置参数缺失")
                
            compile(content, config_file, 'exec')
            print("   ✅ 配置文件语法正确")
        else:
            print(f"   ❌ 配置文件不存在: {config_file}")
    except SyntaxError as e:
        print(f"   ❌ 配置文件语法错误: {e}")
    except Exception as e:
        print(f"   ⚠️  配置文件检查异常: {e}")
    
    print("\n" + "=" * 60)
    print("模块验证完成")
    print("=" * 60)

def check_file_structure():
    """检查文件结构"""
    print("\n文件结构检查:")
    required_files = [
        'model/relation_dynamics.py',
        'model/copy_generation_decoder.py', 
        'model/rrgcn.py',
        'src/config.py',
        'src/main.py'
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"✅ {file_path} (大小: {file_size} bytes)")
        else:
            print(f"❌ {file_path} (文件不存在)")

def provide_next_steps():
    """提供下一步操作建议"""
    print("\n" + "=" * 60)
    print("下一步操作建议:")
    print("=" * 60)
    print("1. 如果所有文件检查通过，可以尝试安装依赖:")
    print("   pip install torch==1.6.0 torchvision==0.7.0 dgl-cu102==0.5.2 tqdm pandas rdflib")
    print()
    print("2. 然后运行基本训练测试（不启用ERD-Net）:")
    print("   python src/main.py --dataset ICEWS14s --train-history-len 10 --test-history-len 10 --dilate-len 1 --lr 0.001 --n-epochs 1 --entity-prediction --relation-prediction --gpu 0 --ft_epochs 1")
    print()
    print("3. 如果基本训练正常，再启用ERD-Net功能:")
    print("   python src/main.py --dataset ICEWS14s --train-history-len 10 --test-history-len 10 --dilate-len 1 --lr 0.001 --n-epochs 1 --entity-prediction --relation-prediction --gpu 0 --ft_epochs 1 --use-relation-dynamics --use-copy-generation")
    print("=" * 60)

if __name__ == "__main__":
    test_import_and_syntax()
    check_file_structure()
    provide_next_steps()