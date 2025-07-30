#!/usr/bin/env python3
"""
✅ 大場景切片數據生成 - 正確的單位轉換版本
基於用戶提供的落地細節，修正關鍵的功率單位問題
"""

import os
import numpy as np
import tensorflow as tf
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, RadioMapSolver

# GPU設置
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# ✅ 固定參數設置
TILE = 512.0           # tile大小 512m x 512m
RES = 4.0             # 解析度 4m/pixel
GRID = int(TILE/RES)  # 網格大小 128x128
FREQUENCY = 3.66e9    # 3.66 GHz
Z_PLANE = 1.5         # ✅ 固定地面測量高度 1.5m
AZ_LIST = [0, 90, 180, 270]  # 四個方位角(度)

# RadioMapSolver通用參數
MAX_DEPTH = 8
SAMPLES_PER_TX = 5_000_000  # 5M條射線（可調到7M）
RM_COMMON = {
    "cell_size": (RES, RES),
    "size": [TILE, TILE],
    "orientation": [0, 0, 0],
    "specular_reflection": True,
    "diffraction": True,
    "refraction": False,  # 簡化
    "max_depth": MAX_DEPTH,
    "samples_per_tx": SAMPLES_PER_TX
}

# ✅ 0 dBi 等效各向性天線（用於生成Piso）
TX_ARRAY_ISO = {
    "num_rows": 1,
    "num_cols": 1,
    "vertical_spacing": 0.5,  # 單位：波長倍數
    "horizontal_spacing": 0.5,
    "pattern": "iso",
    "polarization": "V"
}

# ✅ 3GPP TR38901 定向天線（用於生成Pdir）
TX_ARRAY_3GPP = {
    "num_rows": 1,
    "num_cols": 1,
    "vertical_spacing": 0.5,  # 單位：波長倍數
    "horizontal_spacing": 0.5,
    "pattern": "tr38901",     # ✅ 3GPP TR 38.901標準天線方向圖
    "polarization": "V"       # 單極化
}

RX_ARRAY_CONFIG = TX_ARRAY_ISO  # RX用各向性

def w_to_dbm(p_w):
    """
    ✅ 正確的功率單位轉換：W → dBm
    避免數值下溢，clip到極小值
    """
    p_mw = np.clip(p_w, 1e-18, None) * 1e3  # W → mW
    return 10.0 * np.log10(p_mw)

def generate_tile_centers(bounds, stride=TILE):
    """
    為大場景生成tile中心座標列表
    """
    x_min, x_max, y_min, y_max = bounds
    xs = np.arange(x_min + TILE/2, x_max - TILE/2 + 1e-6, stride)
    ys = np.arange(y_min + TILE/2, y_max - TILE/2 + 1e-6, stride)
    return [(float(cx), float(cy)) for cx in xs for cy in ys]

def pseudo_building_height_map(tile_center):
    """
    ✅ 暫時版建築高度圖B（基於tile位置的確定性隨機）
    實際使用時替換為真實OSM/建築光柵化結果
    """
    rng = np.random.default_rng(int(tile_center[0]*13 + tile_center[1]*17))
    
    # 模擬不同密度區域：市中心/商業/郊區
    r = np.hypot(tile_center[0], tile_center[1])
    if r < 1000:      # 市中心
        base, spread = 20, 60
    elif r < 3000:    # 一般都市
        base, spread = 10, 30
    else:             # 郊區
        base, spread = 5, 15
    
    B = rng.uniform(base, base + spread, size=(GRID, GRID)).astype(np.float32)
    return B

def compute_puma_friis(tx_pos, centers_xy):
    """
    ✅ 極輕量P_UMa近似：Friis/log-distance路徑模型
    """
    dx = centers_xy[..., 0] - tx_pos[0]
    dy = centers_xy[..., 1] - tx_pos[1]
    dz = Z_PLANE - tx_pos[2]
    d = np.sqrt(dx*dx + dy*dy + dz*dz) + 1e-6  # 避免除零
    
    # 路徑損(dB)：PL = PL0 + 10n*log10(d/d0)
    f_GHz = FREQUENCY / 1e9
    PL0 = 32.4 + 20 * np.log10(f_GHz)  # free-space @1m
    n = 2.3
    PL = PL0 + 10 * n * np.log10(d / 1.0)
    
    # 回傳路徑增益(dB) = -PL（當作粗先驗）
    return (-PL).astype(np.float32)

def make_grid_centers(cx, cy):
    """
    生成tile內每個cell中心的(x,y)座標
    """
    xs = np.linspace(cx - TILE/2 + RES/2, cx + TILE/2 - RES/2, GRID)
    ys = np.linspace(cy - TILE/2 + RES/2, cy + TILE/2 - RES/2, GRID)
    X, Y = np.meshgrid(xs, ys)  # [GRID, GRID]
    return np.stack([X, Y], axis=-1).astype(np.float32)

def run_piso_pdir_for_tile(scene, cx, cy, tx_pos, rx_height=Z_PLANE):
    """
    ✅ 為單個tile生成Piso和多方位Pdir
    關鍵：使用0dBm發射功率，確保rm.rss轉dBm後就是路徑增益PG
    """
    # 清場
    for name in list(scene.transmitters): 
        scene.remove(name)
    for name in list(scene.receivers):    
        scene.remove(name)

    # ✅ 設置天線陣列：TX用各向性，RX用各向性
    scene.tx_array = PlanarArray(**TX_ARRAY_ISO)
    scene.rx_array = PlanarArray(**RX_ARRAY_CONFIG)

    # Receiver放在切片平面中心（API需要，但不使用RSS值）
    rx = Receiver(name="rx", position=[cx, cy, rx_height])
    scene.add(rx)

    rm_solver = RadioMapSolver()
    rm_args = dict(RM_COMMON, center=[cx, cy, Z_PLANE])

    # === 1. 生成Piso（0 dBm, 各向性） ===
    tx_iso = Transmitter(name="tx_iso", position=tx_pos, power_dbm=0.0)  # ✅ 0 dBm!!
    scene.add(tx_iso)
    
    rm_iso = rm_solver(scene, **rm_args)
    # ✅ rm_iso.rss[0]是W → 轉dBm；因為Tx=0dBm，這個dBm即為路徑增益PG
    Piso_dB = w_to_dbm(rm_iso.rss[0].numpy()).astype(np.float32)
    scene.remove("tx_iso")

    # === 2. 生成多方位Pdir（0 dBm, 3GPP天線） ===
    # ✅ 切換到3GPP TR38901定向天線
    scene.tx_array = PlanarArray(**TX_ARRAY_3GPP)
    
    Pdirs = []
    for i, az in enumerate(AZ_LIST):
        # ✅ yaw方位角 + 可選的電下傾
        elec_downtilt_deg = 8.0  # 電下傾角度
        orientation = [
            np.deg2rad(az),                    # yaw (方位角)
            np.deg2rad(-elec_downtilt_deg),    # pitch (下傾)
            0.0                                # roll
        ]
        
        tx = Transmitter(
            name=f"tx_dir_{i}", 
            position=tx_pos,
            orientation=orientation, 
            power_dbm=0.0  # ✅ 0 dBm!!
        )
        scene.add(tx)
        
        rm_dir = rm_solver(scene, **rm_args)
        Pdir_dB = w_to_dbm(rm_dir.rss[0].numpy()).astype(np.float32)
        Pdirs.append(Pdir_dB)
        
        scene.remove(f"tx_dir_{i}")

    return Piso_dB, Pdirs

def link_budget_from_pdir(Pdir_dB):
    """
    ✅ 正確的鏈路預算計算：在numpy中進行，避免雙重計數
    S = Ptx + Gtx + Pdir + Grx - IL （全部dB/dBm）
    """
    # 隨機參數（可外部傳入）：讓同一張Pdir生成多張S增廣
    Ptx = np.random.uniform(10, 35)   # dBm
    Gtx = np.random.uniform(10, 20)   # dBi  
    Grx = np.random.uniform(10, 20)   # dBi
    IL = np.random.uniform(-10, 10)   # dB（損失→會被減）
    
    # ✅ S = Ptx + Gtx + Pdir + Grx - IL （全部dB/dBm）
    return (Ptx + Gtx + Pdir_dB + Grx - IL).astype(np.float32)

def process_tile(scene, cx, cy, out_dir, tile_id):
    """
    ✅ 處理單個tile：生成Piso/Pdir/S/B/P_UMa並儲存
    """
    os.makedirs(out_dir, exist_ok=True)

    # 設定BS位置（tile中心附近小偏移）
    tx_x = cx + np.random.uniform(-50, 50)
    tx_y = cy + np.random.uniform(-50, 50)
    tx_h = np.random.uniform(30, 120)  # 先隨機；未來可用「tile最高建築+5m」
    tx_pos = [tx_x, tx_y, float(tx_h)]

    # ✅ 生成Piso/Pdir（注意0dBm → dBm即PG）
    Piso_dB, Pdirs = run_piso_pdir_for_tile(scene, cx, cy, tx_pos)

    # 生成B與P_UMa粗先驗
    B = pseudo_building_height_map((cx, cy))
    G_xy = make_grid_centers(cx, cy)
    P_uma = compute_puma_friis(tx_pos, G_xy)

    # ✅ 每個方位產一筆S
    for k, Pdir_dB in enumerate(Pdirs):
        S_dBm = link_budget_from_pdir(Pdir_dB)

        np.savez(
            os.path.join(out_dir, f"tile_{tile_id:05d}_az{k:02d}.npz"),
            B=B, 
            Piso=Piso_dB, 
            Pdir=Pdir_dB, 
            S=S_dBm, 
            P_uma=P_uma,
            tx_position=np.array(tx_pos, dtype=np.float32),
            tile_center=np.array([cx, cy], dtype=np.float32),
            z_plane=np.array([Z_PLANE], dtype=np.float32),
            azimuth=np.array([AZ_LIST[k]], dtype=np.float32)
        )
    
    print(f"✅ Tile {tile_id} 完成 at ({cx:.0f},{cy:.0f}) - {len(Pdirs)}個方位")

def make_tiles(scene_path, bounds, out_dir, stride=TILE):
    """
    ✅ 主函數：從大場景生成tile資料集
    """
    print(f"載入場景: {scene_path}")
    scene = load_scene(scene_path)
    
    centers = generate_tile_centers(bounds, stride=stride)
    print(f"場景邊界 {bounds} → {len(centers)} tiles")

    for i, (cx, cy) in enumerate(centers):
        try:
            process_tile(scene, cx, cy, out_dir, tile_id=i)
        except Exception as e:
            print(f"❌ Tile {i} 失敗 @({cx:.1f},{cy:.1f}): {e}")
            continue
    
    print(f"✅ 完成！共 {len(centers)} tiles × {len(AZ_LIST)} 方位 = {len(centers)*len(AZ_LIST)} 樣本")

def validate_single_tile(scene_path, bounds, test_tile_id=0):
    """
    ✅ 單位驗證：檢查鏈路預算是否正確
    設Ptx=30, Gtx=0, Grx=0, IL=0，用鏈路預算vs直接30dBm應該一致
    """
    print("🔍 單位驗證測試...")
    
    scene = load_scene(scene_path)
    centers = generate_tile_centers(bounds, stride=TILE)
    
    if len(centers) == 0:
        print("❌ 沒有可用的tiles")
        return False
    
    cx, cy = centers[test_tile_id % len(centers)]
    tx_pos = [cx, cy + 100, 50.0]  # 固定位置測試
    
    print(f"測試tile: ({cx}, {cy}), TX: {tx_pos}")
    
    # 方法1：用鏈路預算
    Piso_dB, Pdirs = run_piso_pdir_for_tile(scene, cx, cy, tx_pos)
    Pdir_dB = Pdirs[0]  # 第一個方位
    
    # 固定參數的鏈路預算
    Ptx, Gtx, Grx, IL = 30.0, 0.0, 0.0, 0.0  # 簡化測試
    S_budget = Ptx + Gtx + Pdir_dB + Grx - IL
    
    # 方法2：直接用30dBm重跑
    scene.tx_array = PlanarArray(**TX_ARRAY_3GPP)
    scene.rx_array = PlanarArray(**RX_ARRAY_CONFIG)
    
    for name in list(scene.transmitters): scene.remove(name)
    for name in list(scene.receivers): scene.remove(name)
    
    rx = Receiver(name="rx", position=[cx, cy, Z_PLANE])
    scene.add(rx)
    
    tx_direct = Transmitter(
        name="tx_direct", position=tx_pos, 
        orientation=[0.0, np.deg2rad(-8.0), 0.0],  # 同樣的方位
        power_dbm=30.0  # 直接30dBm
    )
    scene.add(tx_direct)
    
    rm_solver = RadioMapSolver()
    rm_direct = rm_solver(scene, **dict(RM_COMMON, center=[cx, cy, Z_PLANE]))
    S_direct = w_to_dbm(rm_direct.rss[0].numpy())
    
    # 比較
    diff = np.abs(S_budget - S_direct)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"鏈路預算結果範圍: [{np.min(S_budget):.1f}, {np.max(S_budget):.1f}] dBm")
    print(f"直接30dBm結果範圍: [{np.min(S_direct):.1f}, {np.max(S_direct):.1f}] dBm")
    print(f"最大差異: {max_diff:.3f} dB")
    print(f"平均差異: {mean_diff:.3f} dB")
    
    if max_diff < 0.5:  # 容許數值誤差
        print("✅ 單位驗證通過！")
        return True
    else:
        print("❌ 單位驗證失敗！")
        return False

if __name__ == "__main__":
    # ✅ 使用你的現有場景：大場景切片生成
    scene_paths = [
        "./nnn/nnn.xml",      # 第一個場景
        "./NYCU/NYCU.xml",    # NYCU場景  
    ]
    
    # 場景邊界座標(m) - 先用較小範圍測試
    scene_bounds = [
        [-1000, 1000, -1000, 1000],  # nnn場景 2km x 2km
        [-800, 800, -800, 800],      # NYCU場景 1.6km x 1.6km
    ]
    
    print("=== ✅ 修正版大場景切片生成 ===")
    print("關鍵改進：")
    print("- 使用0dBm發射功率，確保單位正確")
    print("- 3GPP TR38901真實天線方向圖")
    print("- 固定Z_PLANE=1.5m地面測量")
    print("- 正確的鏈路預算計算")
    print()
    
    for i, (path, bounds) in enumerate(zip(scene_paths, scene_bounds)):
        if not os.path.exists(path):
            print(f"⚠️ 場景檔案不存在: {path}")
            continue
            
        print(f"場景{i+1}: {path} → 邊界 {bounds}")
        x_range = bounds[1] - bounds[0]
        y_range = bounds[3] - bounds[2]
        n_tiles = len(generate_tile_centers(bounds))
        total_samples = n_tiles * len(AZ_LIST)
        print(f"  {x_range/1000:.1f}×{y_range/1000:.1f}km → {n_tiles} tiles → {total_samples} samples")
    
    print()
    
    # 選擇操作模式
    mode = input("選擇模式 [1=單位驗證, 2=生成資料集, 3=兩者都做]: ").strip()
    
    if mode in ["1", "3"]:
        # 單位驗證測試
        print("\n🔍 執行單位驗證...")
        for i, (path, bounds) in enumerate(zip(scene_paths, scene_bounds)):
            if os.path.exists(path):
                print(f"\n--- 驗證場景{i+1}: {path} ---")
                validate_single_tile(path, bounds, test_tile_id=0)
                break  # 只驗證第一個可用場景
    
    if mode in ["2", "3"]:
        # 生成完整資料集
        print("\n📊 執行資料集生成...")
        for i, (path, bounds) in enumerate(zip(scene_paths, scene_bounds)):
            if os.path.exists(path):
                output_dir = f"./geo2sigmap_tiles_scene{i+1}"
                print(f"\n--- 處理場景{i+1}: {path} → {output_dir} ---")
                try:
                    make_tiles(path, bounds, output_dir, stride=TILE)
                except Exception as e:
                    print(f"❌ 場景{i+1}生成失敗: {e}")
                    continue
    
    print("\n✅ 所有操作完成！")
    print("\n下一步:")
    print("1. 檢查生成的.npz檔案格式")
    print("2. 用dataset.py測試資料載入")
    print("3. 如果成功，可調大scene_bounds生成更多資料")
    print("4. 開始兩階段訓練 python train.py")