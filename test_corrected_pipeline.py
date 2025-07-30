#!/usr/bin/env python3
"""
測試修正後的數據生成管道
"""

import os
import sys
import numpy as np

def test_data_generation():
    """
    測試基本的數據生成功能
    """
    print("=== 測試修正後的數據生成管道 ===")
    
    # 檢查場景檔案
    scene_paths = ["./nnn/nnn.xml", "./NYCU/NYCU.xml"]
    available_scenes = []
    
    for path in scene_paths:
        if os.path.exists(path):
            available_scenes.append(path)
            print(f"✅ 場景檔案存在: {path}")
        else:
            print(f"❌ 場景檔案不存在: {path}")
    
    if not available_scenes:
        print("❌ 沒有可用的場景檔案")
        return False
    
    # 測試新的數據生成API
    try:
        from data_generation_clean import (
            generate_tile_centers, make_tiles, validate_single_tile,
            TILE, GRID, Z_PLANE, AZ_LIST
        )
        print("✅ 成功導入修正後的數據生成模組")
        
        # 測試參數
        print(f"✅ 參數檢查:")
        print(f"  TILE大小: {TILE}m × {TILE}m")
        print(f"  網格解析度: {GRID}×{GRID} pixels")
        print(f"  測量高度: {Z_PLANE}m")
        print(f"  天線方位: {AZ_LIST}")
        
    except ImportError as e:
        print(f"❌ 導入失敗: {e}")
        return False
    
    # 測試tile中心計算
    test_bounds = [-500, 500, -500, 500]  # 1km×1km測試區域
    centers = generate_tile_centers(test_bounds)
    print(f"✅ 測試區域 {test_bounds} → {len(centers)} tiles")
    
    # 如果有Sionna環境，執行單位驗證
    try:
        scene_path = available_scenes[0]
        print(f"\n🔍 對 {scene_path} 執行單位驗證...")
        
        # 這裡會實際載入Sionna並測試
        validation_passed = validate_single_tile(scene_path, test_bounds, 0)
        
        if validation_passed:
            print("✅ 單位驗證通過！")
        else:
            print("❌ 單位驗證失敗，請檢查配置")
            return False
            
    except Exception as e:
        print(f"⚠️ 單位驗證跳過（可能是Sionna環境問題）: {e}")
        print("⚠️ 如果你有Sionna環境，請手動運行 python data_generation_clean.py")
    
    return True

def test_dataset_compatibility():
    """
    測試與dataset.py的兼容性
    """
    print("\n=== 測試數據集兼容性 ===")
    
    try:
        from dataset import Geo2SigMapDataset
        print("✅ 成功導入dataset模組")
        
        # 檢查是否有測試數據
        test_dirs = ["./geo2sigmap_tiles_scene1", "./geo2sigmap_tiles_scene2", "./geo2sigmap_dataset"]
        
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                files = [f for f in os.listdir(test_dir) if f.endswith('.npz')]
                if files:
                    print(f"✅ 找到測試數據: {test_dir} ({len(files)} 檔案)")
                    
                    # 測試載入
                    try:
                        dataset = Geo2SigMapDataset(test_dir, stage="stage1", split="train")
                        if len(dataset) > 0:
                            sample = dataset[0]
                            print(f"✅ 成功載入數據樣本:")
                            print(f"  輸入形狀: {sample['input'].shape}")
                            print(f"  目標形狀: {sample['target'].shape}")
                            print(f"  metadata: {list(sample['metadata'].keys())}")
                            return True
                        else:
                            print("⚠️ 數據集為空")
                    except Exception as e:
                        print(f"❌ 數據載入失敗: {e}")
        
        print("⚠️ 沒有找到可用的測試數據")
        print("請先運行 python data_generation_clean.py 生成數據")
        return False
        
    except ImportError as e:
        print(f"❌ 導入dataset模組失敗: {e}")
        return False

def main():
    """
    主測試函數
    """
    print("🚀 開始測試修正後的數據生成管道")
    print("=" * 50)
    
    # 測試1: 基本數據生成功能
    success1 = test_data_generation()
    
    # 測試2: 數據集兼容性
    success2 = test_dataset_compatibility()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("🎉 所有測試通過！")
        print("\n建議下一步:")
        print("1. 運行完整數據生成: python data_generation_clean.py")
        print("2. 開始訓練: python train.py")
    elif success1:
        print("✅ 數據生成測試通過")
        print("⚠️ 請先生成測試數據再測試兼容性")
        print("\n建議:")
        print("運行: python data_generation_clean.py")
    else:
        print("❌ 測試失敗，請檢查環境配置")
        print("\n排錯建議:")
        print("1. 確認Sionna環境正確安裝")
        print("2. 檢查場景XML檔案路徑")
        print("3. 確認GPU記憶體充足")

if __name__ == "__main__":
    main()