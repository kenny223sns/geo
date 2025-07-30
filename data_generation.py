import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, RadioMapSolver
import sionna

# GPU設置
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# ✅ 固定參數設置（基於Geo2SigMap論文 + 修正單位問題）
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
    
    參數:
        bounds: [x_min, x_max, y_min, y_max] 場景邊界(m)
        stride: tile步距(m)，TILE表示無重疊
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
    必要時可加NLoS懲罰
    
    參數:
        tx_pos: [x, y, z] BS位置
        centers_xy: (GRID, GRID, 2) 每像素中心座標
    """
    dx = centers_xy[..., 0] - tx_pos[0]
    dy = centers_xy[..., 1] - tx_pos[1]
    dz = Z_PLANE - tx_pos[2]
    d = np.sqrt(dx*dx + dy*dy + dz*dz) + 1e-6  # 避免除零
    
    # 路徑損(dB)：PL = PL0 + 10n*log10(d/d0)
    # 取n=2.3，d0=1m，PL0≈32.4 + 20*log10(f_GHz)
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

# ✅ 舊版相容性函數（簡化版）
def generate_single_sample(scene, tx_position, output_dir, sample_id):
    """
    生成單個訓練樣本（舊版相容性，建議使用make_tiles）
    """
    print(f"⚠️ 使用舊版API，建議改用make_tiles()進行大場景切片")
    
    # 簡化實現：在原點生成一個512x512的tile
    cx, cy = 0.0, 0.0
    out_dir = output_dir
    process_tile(scene, cx, cy, out_dir, sample_id)
    
def generate_dataset(num_samples=10, output_dir="./geo2sigmap_dataset"):
    """
    舊版生成函數（保留相容性）
    """
    print("⚠️ 建議使用make_tiles()進行大場景切片生成")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        scene = load_scene("./nnn/nnn.xml")
        print("載入場景成功")
        
        for i in range(num_samples):
            tx_position = [
                np.random.uniform(-200, 200),
                np.random.uniform(-200, 200), 
                np.random.uniform(30, 120)
            ]
            generate_single_sample(scene, tx_position, output_dir, i)
    except Exception as e:
        print(f"生成失敗: {e}")
        scene.remove(tx_name)
    for rx_name in list(scene.receivers):
        scene.remove(rx_name)
    
    # tile參數
    center_x, center_y = tile_center
    tile_center_3d = [center_x, center_y, RX_HEIGHT]
    
    # 動態計算TX高度（建築物最高點+5m）
    if random_tx_height:
        tx_height = np.random.uniform(30, 150)  # 簡化版，實際可用建築高度圖計算
    else:
        tx_height = TX_HEIGHT
    
    # TX位置（tile中心，可稍微偏移）
    tx_offset_x = np.random.uniform(-50, 50)
    tx_offset_y = np.random.uniform(-50, 50)  
    tx_position = [center_x + tx_offset_x, center_y + tx_offset_y, tx_height]
    
    # 添加接收器
    rx = Receiver(name="rx", position=tile_center_3d)
    scene.add(rx)
    
    # 更新RadioMapSolver參數用於此tile
    tile_rm_params = RM_PARAMS.copy()
    tile_rm_params["center"] = tile_center_3d
    tile_rm_params["size"] = [SCENE_SIZE, SCENE_SIZE]
    
    # === 1. 生成Piso（各向性） ===
    scene.tx_array = PlanarArray(**TX_ARRAY_ISO)
    scene.rx_array = PlanarArray(**RX_ARRAY_CONFIG)
    
    tx_iso = Transmitter(name="tx_iso", position=tx_position, power_dbm=30)
    scene.add(tx_iso)
    
    rm_solver = RadioMapSolver()
    rm_iso = rm_solver(scene, **tile_rm_params)
    
    piso_linear = rm_iso.rss[0].numpy()
    piso_db = 10 * np.log10(piso_linear / 1e-3) - 30
    scene.remove("tx_iso")
    
    # === 2. 生成多個Pdir（定向） ===
    scene.tx_array = PlanarArray(**TX_ARRAY_DIR)
    pdir_maps = []
    azimuths = np.linspace(0, 360, num_orientations, endpoint=False)
    
    for i, az in enumerate(azimuths):
        orientation = [0, 0, np.deg2rad(az)]
        tx_dir = Transmitter(
            name=f"tx_dir_{i}",
            position=tx_position,
            orientation=orientation, 
            power_dbm=30
        )
        scene.add(tx_dir)
        
        rm_dir = rm_solver(scene, **tile_rm_params)
        pdir_linear = rm_dir.rss[0].numpy()
        pdir_db = 10 * np.log10(pdir_linear / 1e-3) - 30
        pdir_maps.append(pdir_db)
        
        scene.remove(f"tx_dir_{i}")
    
    # 為每個方位生成樣本
    for orient_idx, pdir_db in enumerate(pdir_maps):
        # === 3. 鏈路預算合成完整SS地圖 ===
        ptx = np.random.uniform(10, 35)
        gtx = np.random.uniform(10, 20) 
        grx = np.random.uniform(10, 20)
        il = np.random.uniform(-10, 10)
        
        S = ptx + gtx + pdir_db + grx - il
        
        # === 4. 生成建築高度圖和P_UMa ===
        B = generate_building_height_map_for_tile(scene, tile_center, SCENE_SIZE)
        P_uma = compute_puma_map(tx_position, grid_size=GRID_SIZE, scene_size=SCENE_SIZE)
        
        # === 5. 儲存資料 ===
        sample_name = f"tile_{tile_id:04d}_orient_{orient_idx:02d}.npz"
        np.savez(
            os.path.join(output_dir, sample_name),
            B=B.astype(np.float32),
            Piso=piso_db.astype(np.float32),
            Pdir=pdir_db.astype(np.float32), 
            S=S.astype(np.float32),
            P_uma=P_uma.astype(np.float32),
            tx_position=np.array(tx_position),
            tile_center=np.array(tile_center),
            orientation=azimuths[orient_idx]
        )
    
    print(f"Tile {tile_id} 完成 ({num_orientations} 個方位)")

def generate_building_height_map_for_tile(scene, tile_center, tile_size):
    """
    為特定tile提取建築高度圖
    """
    # 實際實現中，你需要從scene mesh中提取此tile範圍的建築高度
    # 這裡用簡化版本
    height_map = np.zeros((GRID_SIZE, GRID_SIZE))
    
    # 根據tile位置生成不同的建築分佈（增加多樣性）
    np.random.seed(int(tile_center[0] + tile_center[1]))  # 基於位置的確定性隨機
    
    # 模擬不同區域類型
    distance_from_center = np.sqrt(tile_center[0]**2 + tile_center[1]**2)
    if distance_from_center < 1000:  # 市中心
        height_map = np.random.uniform(20, 80, (GRID_SIZE, GRID_SIZE))
    elif distance_from_center < 3000:  # 一般都市
        height_map = np.random.uniform(10, 40, (GRID_SIZE, GRID_SIZE))
    else:  # 郊區
        height_map = np.random.uniform(5, 20, (GRID_SIZE, GRID_SIZE))
    
    return height_map

def generate_dataset_from_large_scenes(scene_paths, output_dir="./geo2sigmap_dataset", 
                                     scene_bounds_list=None, tile_overlap=0):
    """
    從大場景生成完整訓練資料集
    
    參數:
        scene_paths: 大場景XML路徑列表
        output_dir: 輸出目錄
        scene_bounds_list: 每個場景的邊界 [x_min,x_max,y_min,y_max]
        tile_overlap: tile重疊距離(m)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    total_samples = 0
    
    for scene_idx, scene_path in enumerate(scene_paths):
        print(f"\n處理場景 {scene_idx+1}/{len(scene_paths)}: {scene_path}")
        
        try:
            scene = load_scene(scene_path)
        except Exception as e:
            print(f"無法載入場景 {scene_path}: {e}")
            continue
        
        # 使用提供的邊界或預設邊界
        if scene_bounds_list and scene_idx < len(scene_bounds_list):
            bounds = scene_bounds_list[scene_idx]
        else:
            # 預設邊界（需根據實際場景調整）
            bounds = [-2500, 2500, -2500, 2500]  # 5km x 5km場景
        
        # 生成tile網格
        tile_centers = generate_tile_grid(bounds, SCENE_SIZE, tile_overlap)
        
        # 為每個tile生成樣本
        for tile_idx, tile_center in enumerate(tile_centers):
            global_tile_id = total_samples + tile_idx
            
            try:
                generate_single_tile(
                    scene, tile_center, global_tile_id, output_dir,
                    num_orientations=4, random_tx_height=True
                )
            except Exception as e:
                print(f"Tile {global_tile_id} 生成失敗: {e}")
                continue
        
        total_samples += len(tile_centers) * 4  # 4個方位
        print(f"場景 {scene_idx+1} completed: {len(tile_centers)} tiles")
    
    print(f"\n資料集生成完成！總計 {total_samples} 個樣本")

def generate_dataset(num_samples=100, output_dir="./dataset"):
    """
    生成完整訓練資料集（舊版本，保留相容性）
    """
    print("警告：建議使用 generate_dataset_from_large_scenes() 進行批量生成")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 載入場景
    try:
        scene = load_scene("./nnn/nnn.xml")
        print("使用自定義場景")
    except:
        print("使用內建場景")
        # 這裡需要替換成你實際的場景載入方式
        
    # 生成多個位置的樣本
    for i in range(num_samples):
        # 隨機TX位置（在場景範圍內）
        tx_x = np.random.uniform(-200, 200)
        tx_y = np.random.uniform(-200, 200) 
        tx_position = [tx_x, tx_y, TX_HEIGHT]
        
        try:
            generate_single_sample(scene, tx_position, output_dir, i)
        except Exception as e:
            print(f"樣本 {i} 生成失敗: {e}")
            continue
    
    print(f"資料集生成完成，共 {num_samples} 個樣本")

if __name__ == "__main__":
    # === 使用你的現有場景：大場景切片生成 ===
    scene_paths = [
        "./nnn/nnn.xml",      # 第一個場景
        "./NYCU/NYCU.xml",    # NYCU場景  
    ]
    
    # 場景邊界座標(m) - 需要根據實際場景大小調整
    # 建議先用較小範圍測試，成功後再擴大
    scene_bounds = [
        [-1000, 1000, -1000, 1000],  # nnn場景 2km x 2km
        [-800, 800, -800, 800],      # NYCU場景 1.6km x 1.6km
    ]
    
    print("=== 開始大場景切片生成 ===")
    print(f"場景1: {scene_paths[0]} → 邊界 {scene_bounds[0]}")  
    print(f"場景2: {scene_paths[1]} → 邊界 {scene_bounds[1]}")
    
    # 預估計算
    for i, bounds in enumerate(scene_bounds):
        x_range = bounds[1] - bounds[0]  # x範圍
        y_range = bounds[3] - bounds[2]  # y範圍
        n_tiles_x = int(x_range / 512)
        n_tiles_y = int(y_range / 512) 
        total_tiles = n_tiles_x * n_tiles_y
        total_samples = total_tiles * 4  # 4個方位
        print(f"場景{i+1}: {x_range/1000:.1f}×{y_range/1000:.1f}km → {total_tiles} tiles → {total_samples} samples")
    
    # 執行生成
    try:
        generate_dataset_from_large_scenes(
            scene_paths=scene_paths,
            scene_bounds_list=scene_bounds,
            output_dir="./geo2sigmap_dataset_tiles",
            tile_overlap=0  # 第一次測試不重疊
        )
        
        print("\n✅ 數據集生成完成！")
        print(f"輸出目錄: ./geo2sigmap_dataset_tiles")
        print("\n下一步:")
        print("1. 檢查生成的.npz檔案")
        print("2. 運行: python example_usage.py 測試載入")
        print("3. 如果成功，可以增大scene_bounds範圍生成更多數據")
        
    except Exception as e:
        print(f"\n❌ 生成失敗: {e}")
        print("\n排錯建議:")
        print("1. 檢查XML檔案是否可正常載入")
        print("2. 確認GPU記憶體充足")
        print("3. 嘗試更小的scene_bounds範圍")
        print("4. 檢查Sionna環境是否正確安裝")
    
    print("\n=== 數據集配置 ===")
    print("每個tile: 512m × 512m")
    print("解析度: 4m/pixel (128×128 pixels)")
    print("天線方位: 4個 (0°, 90°, 180°, 270°)")
    print("數據增強: 在dataset.py中處理旋轉/鏡射")
    
    # === 如果需要快速測試單場景，取消註解下面這行 ===
    # generate_dataset(num_samples=5, output_dir="./geo2sigmap_dataset_test")