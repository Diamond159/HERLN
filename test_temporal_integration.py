#!/usr/bin/env python
"""
Integration Test for Periodic Trend Temporal Encoding

This script validates that all components of the temporal trend encoding
are properly integrated and can be initialized without errors.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
from model.temporal_trend_encoder import (
    PeriodicTrendTimeEmbedding,
    TemporalGatingModule,
    TemporalSlideWindow,
    AngleConstrainedLoss,
    TemporalContrastiveLoss,
    TemporalTrendEncoder
)


def test_periodic_trend_embedding():
    """Test PeriodicTrendTimeEmbedding"""
    print("\n" + "="*80)
    print("TEST 1: PeriodicTrendTimeEmbedding")
    print("="*80)
    
    try:
        num_ents = 1000
        h_dim = 200
        
        emb = PeriodicTrendTimeEmbedding(num_ents, h_dim, alpha=0.5)
        
        # Test forward pass
        t = torch.tensor(5.0)
        output = emb(t, device='cpu')
        
        assert output.shape == (num_ents, h_dim), f"Expected shape ({num_ents}, {h_dim}), got {output.shape}"
        
        print(f"✓ Output shape: {output.shape}")
        print(f"✓ Output dtype: {output.dtype}")
        print(f"✓ Output range: [{output.min():.4f}, {output.max():.4f}]")
        print("✓ PASSED")
        
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_temporal_gating():
    """Test TemporalGatingModule"""
    print("\n" + "="*80)
    print("TEST 2: TemporalGatingModule")
    print("="*80)
    
    try:
        h_dim = 200
        batch_size = 50
        
        gating = TemporalGatingModule(h_dim, dropout=0.2)
        
        # Test forward pass
        dynamic_emb = torch.randn(batch_size, h_dim)
        static_emb = torch.randn(batch_size, h_dim)
        
        output = gating(dynamic_emb, static_emb)
        
        assert output.shape == (batch_size, h_dim), f"Expected shape ({batch_size}, {h_dim}), got {output.shape}"
        
        print(f"✓ Output shape: {output.shape}")
        print(f"✓ Output range: [{output.min():.4f}, {output.max():.4f}]")
        print("✓ PASSED")
        
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_temporal_slide_window():
    """Test TemporalSlideWindow"""
    print("\n" + "="*80)
    print("TEST 3: TemporalSlideWindow")
    print("="*80)
    
    try:
        max_len = 10
        window = TemporalSlideWindow(max_len)
        
        # Test at various positions
        positions = [0, 5, 15, 50]
        
        for pos in positions:
            start, end = window.get_history_window(pos, pos + 1)
            print(f"✓ Position {pos}: history window [{start}, {end})")
            assert start >= 0 and start <= end, f"Invalid window: [{start}, {end})"
        
        # Test time indices
        time_indices = window.get_time_indices(10, device='cpu')
        print(f"✓ Time indices shape: {time_indices.shape}")
        
        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_angle_constrained_loss():
    """Test AngleConstrainedLoss"""
    print("\n" + "="*80)
    print("TEST 4: AngleConstrainedLoss")
    print("="*80)
    
    try:
        num_ents = 100
        h_dim = 200
        
        loss_fn = AngleConstrainedLoss(angle_degree=10.0, weight=0.1)
        
        static_emb = torch.randn(num_ents, h_dim)
        dynamic_embs = [torch.randn(num_ents, h_dim) for _ in range(5)]
        
        loss = loss_fn(static_emb, dynamic_embs)
        
        assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"
        
        print(f"✓ Loss value: {loss.item():.6f}")
        print(f"✓ Loss dtype: {loss.dtype}")
        print("✓ PASSED")
        
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_temporal_contrastive_loss():
    """Test TemporalContrastiveLoss"""
    print("\n" + "="*80)
    print("TEST 5: TemporalContrastiveLoss")
    print("="*80)
    
    try:
        num_ents = 100
        h_dim = 200
        batch_size = 50
        
        loss_fn = TemporalContrastiveLoss(h_dim, temperature=0.07, dropout=0.2)
        
        global_embs = torch.randn(num_ents, h_dim)
        local_embs = torch.randn(num_ents, h_dim)
        triplets = torch.randint(0, num_ents, (batch_size, 3))
        
        loss = loss_fn(global_embs, local_embs, triplets)
        
        assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"
        
        print(f"✓ Loss value: {loss.item():.6f}")
        print(f"✓ Loss dtype: {loss.dtype}")
        print("✓ PASSED")
        
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_temporal_trend_encoder():
    """Test complete TemporalTrendEncoder"""
    print("\n" + "="*80)
    print("TEST 6: TemporalTrendEncoder (Complete)")
    print("="*80)
    
    try:
        num_ents = 100
        h_dim = 200
        
        encoder = TemporalTrendEncoder(
            num_entities=num_ents,
            h_dim=h_dim,
            max_history_len=10,
            alpha_balance=0.5,
            angle_degree=10.0,
            temperature=0.07,
            use_gating=True,
            use_angle_loss=True,
            use_contrastive_loss=True,
            dropout=0.2
        )
        
        # Test components
        t = torch.tensor(5.0)
        temporal_emb = encoder.get_temporal_emb(t, device='cpu')
        print(f"✓ Temporal embedding shape: {temporal_emb.shape}")
        
        dynamic_emb = torch.randn(num_ents, h_dim)
        static_emb = torch.randn(num_ents, h_dim)
        gated_emb = encoder.apply_temporal_gating(dynamic_emb, static_emb)
        print(f"✓ Gated embedding shape: {gated_emb.shape}")
        
        history_window = encoder.get_history_window(5, 20)
        print(f"✓ History window: {history_window}")
        
        angle_loss = encoder.compute_angle_loss(static_emb, [dynamic_emb, dynamic_emb])
        print(f"✓ Angle loss: {angle_loss.item():.6f}")
        
        triplets = torch.randint(0, num_ents, (50, 3))
        contrastive_loss = encoder.compute_contrastive_loss(dynamic_emb, static_emb, triplets)
        print(f"✓ Contrastive loss: {contrastive_loss.item():.6f}")
        
        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hrgcn_integration():
    """Test HRGCN integration with temporal gating"""
    print("\n" + "="*80)
    print("TEST 7: HRGCN Integration")
    print("="*80)
    
    try:
        from model.hrgcn import HawkesRGCNLayer
        
        in_feat = 200
        out_feat = 200
        num_rels = 500
        
        # Test without temporal gating
        layer_no_gating = HawkesRGCNLayer(
            in_feat, out_feat, num_rels, 
            use_temporal_gating=False
        )
        print("✓ HRGCN layer without temporal gating created")
        
        # Test with temporal gating
        layer_with_gating = HawkesRGCNLayer(
            in_feat, out_feat, num_rels,
            use_temporal_gating=True
        )
        print("✓ HRGCN layer with temporal gating created")
        
        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rrgcn_initialization():
    """Test RecurrentRGCN initialization with new parameters"""
    print("\n" + "="*80)
    print("TEST 8: RecurrentRGCN Initialization")
    print("="*80)
    
    try:
        from model.rrgcn import RecurrentRGCN
        
        model = RecurrentRGCN(
            decoder_name="convtranse",
            encoder_name="hrgcn",
            num_ents=100,
            num_rels=50,
            h_dim=200,
            opn="sub",
            sequence_len=10,
            num_bases=-1,
            num_basis=100,
            num_hidden_layers=2,
            dropout=0.2,
            self_loop=True,
            skip_connect=False,
            layer_norm=True,
            use_temporal_trend=True,
            temporal_gating=True,
            time_embedding_alpha=0.5,
            angle_degree=10.0,
            temporal_temperature=0.07,
            use_angle_constraint=True,
            use_temporal_contrastive=True,
            angle_constraint_weight=0.1,
            temporal_contrastive_weight=0.1
        )
        
        print(f"✓ RecurrentRGCN created successfully")
        print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"✓ Temporal trend encoder enabled: {model.use_temporal_trend}")
        print(f"✓ Temporal gating enabled: {model.temporal_gating}")
        print("✓ PASSED")
        
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "TEMPORAL TREND ENCODER INTEGRATION TEST" + " "*20 + "║")
    print("╚" + "="*78 + "╝")
    
    tests = [
        test_periodic_trend_embedding,
        test_temporal_gating,
        test_temporal_slide_window,
        test_angle_constrained_loss,
        test_temporal_contrastive_loss,
        test_temporal_trend_encoder,
        test_hrgcn_integration,
        test_rrgcn_initialization
    ]
    
    results = []
    for test_func in tests:
        result = test_func()
        results.append(result)
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for i, (test_func, result) in enumerate(zip(tests, results)):
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{i+1}. {test_func.__name__:.<40} {status}")
    
    passed = sum(results)
    total = len(results)
    
    print("\n" + "="*80)
    print(f"Total: {passed}/{total} tests passed")
    print("="*80 + "\n")
    
    if passed == total:
        print("✓ All tests passed! Integration is successful.")
        return 0
    else:
        print(f"✗ {total - passed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
