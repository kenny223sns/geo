#!/usr/bin/env python3
"""
大場景切片數據生成使用範例
"""

import os
import numpy as np
from data_generation import generate_dataset_from_large_scenes, generate_tile_grid
from dataset import Geo2SigMapDataset, create_dataloaders, visualize_sample

def example_tile_generation():
    """
    範例：從大場景生成tile數據集
    """
    print("=== 大場景切片數據生成範例 ===")
    
    # 1. 使用你的實際場景路徑
    scene_paths = [
        "./nnn/nnn.xml",      # nnn場景
        "./NYCU/NYCU.xml",    # NYCU場景
    ]
    
    # 2. 設定場景邊界（較小範圍用於測試）
    scene_bounds = [
        [-500, 500, -500, 500],    # nnn場景測試範圍 1km x 1km
        [-400, 400, -400, 400],    # NYCU場景測試範圍 0.8km x 0.8km
    ]
    
    # 3. 預覽tile劃分
    print("\n計算tile劃分...")
    for i, bounds in enumerate(scene_bounds):
        tiles = generate_tile_grid(bounds, tile_size=512, overlap=0)
        print(f"場景 {i+1}: {bounds} → {len(tiles)} tiles")
        print(f"預期樣本數: {len(tiles) * 4} (4個方位)")
    
    # 4. 生成數據集（小規模測試）
    output_dir = "./test_tile_dataset"
    
    if not os.path.exists(output_dir):
        print(f"\n開始生成數據集到: {output_dir}")
        try:
            generate_dataset_from_large_scenes(
                scene_paths=scene_paths,
                scene_bounds_list=scene_bounds,
                output_dir=output_dir,
                tile_overlap=0  # 測試時不重疊
            )
            print("✅ 數據集生成完成！")
        except Exception as e:
            print(f"❌ 數據集生成失敗: {e}")
            return False
    else:
        print(f"數據集已存在: {output_dir}")
    
    return True

def example_dataset_loading():
    """
    範例：載入和測試數據集
    """
    print("\n=== 數據集載入測試 ===")
    
    dataset_dir = "./test_tile_dataset"
    
    if not os.path.exists(dataset_dir):
        print(f"數據集目錄不存在: {dataset_dir}")
        return False
    
    # 檢查生成的檔案
    files = [f for f in os.listdir(dataset_dir) if f.endswith('.npz')]
    print(f"生成的數據檔案數: {len(files)}")
    
    if len(files) == 0:
        print("沒有找到數據檔案")
        return False
    
    # 顯示檔案命名格式
    print("檔案命名範例:")
    for i, filename in enumerate(files[:5]):  # 顯示前5個
        print(f"  {filename}")
    if len(files) > 5:
        print(f"  ... 還有 {len(files)-5} 個檔案")
    
    # 測試數據載入
    try:
        # Stage 1 數據集
        print("\n測試 Stage 1 數據集...")
        dataset_s1 = Geo2SigMapDataset(dataset_dir, stage="stage1", split="train")
        print(f"Stage 1 訓練集大小: {len(dataset_s1)}")
        
        if len(dataset_s1) > 0:
            sample = dataset_s1[0]
            print(f"輸入形狀: {sample['input'].shape}")  # 應該是 [2, 128, 128] 
            print(f"目標形狀: {sample['target'].shape}")  # 應該是 [1, 128, 128]
            print(f"metadata: {sample['metadata']}")
        
        # Stage 2 數據集
        print("\n測試 Stage 2 數據集...")
        dataset_s2 = Geo2SigMapDataset(dataset_dir, stage="stage2", split="train")
        print(f"Stage 2 訓練集大小: {len(dataset_s2)}")
        
        if len(dataset_s2) > 0:
            sample = dataset_s2[0]
            print(f"輸入形狀: {sample['input'].shape}")  # 應該是 [4, 128, 128]
            print(f"目標形狀: {sample['target'].shape}")  # 應該是 [1, 128, 128]
        
        print("✅ 數據集載入成功！")
        return True
        
    except Exception as e:
        print(f"❌ 數據集載入失敗: {e}")
        return False

def example_dataloader_creation():
    """
    範例：創建DataLoader進行訓練
    """
    print("\n=== DataLoader 創建測試 ===")
    
    dataset_dir = "./test_tile_dataset"
    
    try:
        # 創建 DataLoader
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir=dataset_dir,
            batch_size=4,  # 小batch測試
            num_workers=0,  # 避免多進程問題
            stage="stage1",
            enable_augmentation=True
        )
        
        print(f"訓練集 batches: {len(train_loader)}")
        print(f"驗證集 batches: {len(val_loader)}")
        print(f"測試集 batches: {len(test_loader)}")
        
        # 測試一個batch
        if len(train_loader) > 0:
            batch = next(iter(train_loader))
            print(f"Batch 輸入形狀: {batch['input'].shape}")
            print(f"Batch 目標形狀: {batch['target'].shape}")
            print("✅ DataLoader 創建成功！")
            return True
        else:
            print("⚠️ DataLoader 為空")
            return False
            
    except Exception as e:
        print(f"❌ DataLoader 創建失敗: {e}")
        return False

def main():
    """
    主函數：運行所有測試
    """
    print("🚀 大場景切片數據管道測試")
    print("=" * 50)
    
    # 測試1: 生成數據集
    success1 = example_tile_generation()
    
    if success1:
        # 測試2: 載入數據集
        success2 = example_dataset_loading()
        
        if success2:
            # 測試3: 創建DataLoader
            success3 = example_dataloader_creation()
            
            if success3:
                print("\n🎉 所有測試通過！")
                print("\n下一步:")
                print("1. 準備你的大場景XML檔案")
                print("2. 調整 scene_bounds 參數")
                print("3. 運行完整數據生成: python data_generation.py")
                print("4. 開始訓練: python train.py")
            else:
                print("\n❌ DataLoader 測試失敗")
        else:
            print("\n❌ 數據載入測試失敗")
    else:
        print("\n❌ 數據生成測試失敗")
        print("請檢查:")
        print("- 場景XML檔案是否存在")
        print("- Sionna環境是否正確安裝")
        print("- GPU記憶體是否充足")

if __name__ == "__main__":
    main()