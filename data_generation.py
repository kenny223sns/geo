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

# ✅ 路徑A參數設置（對齊Geo2SigMap論文表I）
SCENE_SIZE = 512  # 場景大小 512m x 512m
RESOLUTION = 4    # 解析度 4m
GRID_SIZE = 128   # 網格大小 128x128
FREQUENCY = 3.66e9  # 3.66 GHz
TX_HEIGHT = 125   # BS高度（建築最高+5m，可調整）
RX_HEIGHT = 1.5   # 接收高度

# 天線配置
TX_ARRAY_ISO = {
    "num_rows": 1,
    "num_cols": 1,
    "vertical_spacing": 0.5,
    "horizontal_spacing": 0.5,
    "pattern": "iso",
    "polarization": "V"
}

# 定向天線配置（模擬蜂巢天線，6.3dBi增益）
TX_ARRAY_DIR = {
    "num_rows": 8,
    "num_cols": 1, 
    "vertical_spacing": 0.5,
    "horizontal_spacing": 0.5,
    "pattern": "iso",  # 用array factor產生方向性
    "polarization": "V"
}

RX_ARRAY_CONFIG = TX_ARRAY_ISO

# RadioMapSolver參數（表I）
RM_PARAMS = {
    "max_depth": 8,              # 最多8次反射/繞射
    "samples_per_tx": 7_000_000, # 7M條射線  
    "cell_size": (RESOLUTION, RESOLUTION),
    "center": [0, 0, RX_HEIGHT],
    "size": [SCENE_SIZE, SCENE_SIZE],  # ✅ 512m x 512m
    "orientation": [0, 0, 0],
    "specular_reflection": True,
    "diffuse_reflection": False,  # 論文主要用反射
    "diffraction": True,
    "refraction": False  # 簡化，可開啟
}

def generate_building_height_map(scene, grid_size=128):
    """
    從Sionna場景提取建築高度圖 B
    """
    # 這裡簡化為從場景物件提取高度資訊
    # 實際應用中可從OSM/CAD轉換
    height_map = np.zeros((grid_size, grid_size))
    
    # 如果你有建築物件，在此處理
    # 暫時用隨機高度模擬（實際使用時需替換）
    np.random.seed(42)
    height_map = np.random.uniform(0, 50, (grid_size, grid_size))
    
    return height_map

def compute_puma_map(tx_pos, grid_size=128, scene_size=512):
    """
    計算3GPP UMa路徑損模型的P_UMa地圖
    簡化版：使用對數距離模型
    """
    x = np.linspace(-scene_size/2, scene_size/2, grid_size)
    y = np.linspace(-scene_size/2, scene_size/2, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # 計算距離
    dx = X - tx_pos[0]
    dy = Y - tx_pos[1]
    dz = RX_HEIGHT - tx_pos[2]
    distance_3d = np.sqrt(dx**2 + dy**2 + dz**2)
    distance_2d = np.sqrt(dx**2 + dy**2)
    
    # 簡化UMa模型（可替換成完整的3GPP公式）
    # PL = 32.4 + 20*log10(fc) + 30*log10(d3d)  # LOS
    frequency_ghz = FREQUENCY / 1e9
    pl_los = 32.4 + 20*np.log10(frequency_ghz) + 20*np.log10(distance_3d)
    
    # 轉換為路徑增益（dB）
    path_gain = -pl_los
    
    return path_gain

def generate_single_sample(scene, tx_position, output_dir, sample_id):
    """
    生成單個訓練樣本
    """
    print(f"生成樣本 {sample_id}...")
    
    # 清除現有發射器和接收器
    for tx_name in list(scene.transmitters):
        scene.remove(tx_name)
    for rx_name in list(scene.receivers):
        scene.remove(rx_name)
    
    # 添加接收器（固定位置用於RadioMap）
    rx = Receiver(name="rx", position=[0, 0, RX_HEIGHT])
    scene.add(rx)
    
    # === 1. 生成Piso（各向性） ===
    scene.tx_array = PlanarArray(**TX_ARRAY_ISO)
    scene.rx_array = PlanarArray(**RX_ARRAY_CONFIG)
    
    tx_iso = Transmitter(name="tx_iso", position=tx_position, power_dbm=30)
    scene.add(tx_iso)
    
    rm_solver = RadioMapSolver()
    rm_iso = rm_solver(scene, **RM_PARAMS)
    
    # 提取Piso（dB）
    piso_linear = rm_iso.rss[0].numpy()  # 線性尺度
    piso_db = 10 * np.log10(piso_linear / 1e-3)  # 轉dBm再轉路徑增益
    piso_db = piso_db - 30  # 扣除發射功率得路徑增益
    
    scene.remove("tx_iso")
    
    # === 2. 生成多個Pdir（定向） ===
    scene.tx_array = PlanarArray(**TX_ARRAY_DIR)
    pdir_maps = []
    azimuths = [0, 90, 180, 270]  # 四個方位角
    
    for i, az in enumerate(azimuths):
        orientation = [0, 0, np.deg2rad(az)]  # 方位角轉弧度
        tx_dir = Transmitter(
            name=f"tx_dir_{i}", 
            position=tx_position, 
            orientation=orientation,
            power_dbm=30
        )
        scene.add(tx_dir)
        
        rm_dir = rm_solver(scene, **RM_PARAMS)
        pdir_linear = rm_dir.rss[0].numpy()
        pdir_db = 10 * np.log10(pdir_linear / 1e-3) - 30
        pdir_maps.append(pdir_db)
        
        scene.remove(f"tx_dir_{i}")
    
    # 選擇一個方位作為主要Pdir（或平均）
    pdir_db = pdir_maps[0]  # 使用第一個方位
    
    # === 3. 用鏈路預算合成完整SS地圖 ===
    # 隨機參數（增加多樣性）
    ptx = np.random.uniform(10, 35)   # dBm
    gtx = np.random.uniform(10, 20)   # dB
    grx = np.random.uniform(10, 20)   # dB
    il = np.random.uniform(-10, 10)   # dB
    
    S = ptx + gtx + pdir_db + grx - il  # 完整SS地圖（dBm）
    
    # === 4. 生成建築高度圖和P_UMa ===
    B = generate_building_height_map(scene)
    P_uma = compute_puma_map(tx_position)
    
    # === 5. 儲存資料 ===
    np.savez(
        os.path.join(output_dir, f"sample_{sample_id:06d}.npz"),
        B=B.astype(np.float32),
        Piso=piso_db.astype(np.float32), 
        Pdir=pdir_db.astype(np.float32),
        S=S.astype(np.float32),
        P_uma=P_uma.astype(np.float32),
        tx_position=np.array(tx_position)
    )
    
    print(f"樣本 {sample_id} 完成")
    return True

def generate_dataset(num_samples=100, output_dir="./dataset"):
    """
    生成完整訓練資料集
    """
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
    generate_dataset(num_samples=10, output_dir="./geo2sigmap_dataset")