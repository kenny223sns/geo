#!/usr/bin/env python3
"""
快速測試：檢查場景能否正常載入
"""

def test_scene_loading():
    """
    測試場景載入
    """
    scene_paths = ["./nnn/nnn.xml", "./NYCU/NYCU.xml"]
    
    print("=== 測試場景載入 ===")
    
    for i, path in enumerate(scene_paths):
        print(f"\n測試場景 {i+1}: {path}")
        
        try:
            # 檢查檔案是否存在
            import os
            if not os.path.exists(path):
                print(f"❌ 檔案不存在: {path}")
                continue
            
            print(f"✅ 檔案存在: {path}")
            
            # 嘗試用Sionna載入
            try:
                from sionna.rt import load_scene
                scene = load_scene(path)
                print(f"✅ Sionna載入成功")
                
                # 檢查場景內容
                print(f"場景物件數量: {len(scene.objects)}")
                
            except ImportError:
                print("⚠️ Sionna未安裝，無法測試場景載入")
            except Exception as e:
                print(f"❌ Sionna載入失敗: {e}")
                
        except Exception as e:
            print(f"❌ 測試失敗: {e}")

if __name__ == "__main__":
    test_scene_loading()