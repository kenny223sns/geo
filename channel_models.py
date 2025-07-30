import numpy as np
import matplotlib.pyplot as plt

def compute_3gpp_uma_path_loss(tx_pos, rx_height=1.5, grid_size=128, scene_size=512, 
                               frequency=3.66e9, bs_height=25, los_probability=None):
    """
    實現3GPP UMa (Urban Macro) 通道模型
    
    參數:
        tx_pos: [x, y, z] 基地台位置
        rx_height: 接收器高度 (m)
        grid_size: 網格大小
        scene_size: 場景大小 (m)  
        frequency: 頻率 (Hz)
        bs_height: 基地台有效高度 (m)
        los_probability: LOS機率地圖 (可選)
    
    返回:
        path_gain_map: 路徑增益地圖 (dB)
    """
    
    # 創建座標網格
    x = np.linspace(-scene_size/2, scene_size/2, grid_size)
    y = np.linspace(-scene_size/2, scene_size/2, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # 計算2D和3D距離
    dx = X - tx_pos[0]
    dy = Y - tx_pos[1]
    d_2d = np.sqrt(dx**2 + dy**2)
    d_3d = np.sqrt(d_2d**2 + (tx_pos[2] - rx_height)**2)
    
    # 頻率 (GHz)
    fc = frequency / 1e9
    
    # 有效天線高度
    h_bs = bs_height
    h_ut = rx_height
    
    # === LOS/NLOS 機率計算 ===
    if los_probability is None:
        # 3GPP UMa LOS 機率模型
        # P_LOS = min(18/d_2d, 1) * (1 - exp(-d_2d/63)) + exp(-d_2d/63)
        p_los = np.minimum(18/np.maximum(d_2d, 1), 1) * (1 - np.exp(-d_2d/63)) + np.exp(-d_2d/63)
        p_los = np.clip(p_los, 0, 1)
    else:
        p_los = los_probability
    
    # === LOS 路徑損失 ===
    # 有效天線高度差
    h_e = 1.0  # 有效環境高度
    
    # Breaking point distance
    d_bp = 4 * h_bs * h_ut * fc / 3e8 * 1e9  # 轉換單位
    
    # LOS路徑損失
    pl_los = np.zeros_like(d_2d)
    
    # d_2d <= d_bp
    mask_close = d_2d <= d_bp
    pl_los[mask_close] = 32.4 + 21*np.log10(d_3d[mask_close]) + 20*np.log10(fc)
    
    # d_2d > d_bp  
    mask_far = d_2d > d_bp
    pl_los[mask_far] = (32.4 + 40*np.log10(d_3d[mask_far]) + 20*np.log10(fc) 
                        - 9.5*np.log10(d_bp**2 + (h_bs - h_ut)**2))
    
    # === NLOS 路徑損失 ===
    # NLOS路徑損失
    pl_nlos = (13.54 + 39.08*np.log10(d_3d) + 20*np.log10(fc) 
               - 0.6*(h_ut - 1.5))
    
    # 確保NLOS >= LOS
    pl_nlos = np.maximum(pl_nlos, pl_los)
    
    # === 陰影衰落 (可選) ===
    # sf_los = 4   # dB (LOS標準差)
    # sf_nlos = 6  # dB (NLOS標準差)
    
    # 合併LOS/NLOS路徑損失  
    path_loss = p_los * pl_los + (1 - p_los) * pl_nlos
    
    # 轉換為路徑增益 (dB)
    path_gain = -path_loss
    
    return path_gain

def compute_friis_path_loss(tx_pos, rx_height=1.5, grid_size=128, scene_size=512, 
                           frequency=3.66e9, path_loss_exponent=2.0):
    """
    簡化版：Friis + 對數距離模型
    用於快速計算或無3GPP需求的場合
    """
    # 創建座標網格
    x = np.linspace(-scene_size/2, scene_size/2, grid_size)
    y = np.linspace(-scene_size/2, scene_size/2, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # 計算3D距離
    dx = X - tx_pos[0]
    dy = Y - tx_pos[1]
    dz = tx_pos[2] - rx_height
    d_3d = np.sqrt(dx**2 + dy**2 + dz**2)
    
    # 避免除零
    d_3d = np.maximum(d_3d, 1.0)
    
    # Friis 自由空間路徑損失
    wavelength = 3e8 / frequency
    pl_free = 20*np.log10(4*np.pi*d_3d/wavelength)
    
    # 對數距離擴展 (path loss exponent)
    if path_loss_exponent != 2.0:
        pl_total = pl_free + 10*(path_loss_exponent-2)*np.log10(d_3d)
    else:
        pl_total = pl_free
    
    # 轉換為路徑增益
    path_gain = -pl_total
    
    return path_gain

def compute_antenna_gain_map(tx_pos, tx_orientation=[0,0,0], grid_size=128, scene_size=512,
                            antenna_pattern="3gpp_macro"):
    """
    計算天線方向圖增益地圖 G_bs
    
    參數:
        tx_pos: 基地台位置
        tx_orientation: [tilt, azimuth, 0] 天線指向
        antenna_pattern: 天線類型 
    """
    # 創建座標網格
    x = np.linspace(-scene_size/2, scene_size/2, grid_size)
    y = np.linspace(-scene_size/2, scene_size/2, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # 計算方向向量
    dx = X - tx_pos[0]
    dy = Y - tx_pos[1]
    dz = 1.5 - tx_pos[2]  # 接收高度
    
    # 方位角和仰角
    azimuth = np.arctan2(dy, dx)  # -π to π
    elevation = np.arctan2(dz, np.sqrt(dx**2 + dy**2))
    
    # 轉換為度數
    az_deg = np.rad2deg(azimuth)
    el_deg = np.rad2deg(elevation)
    
    if antenna_pattern == "3gpp_macro":
        # 3GPP 標準宏基地台天線方向圖
        # 水平方向圖
        phi_3db = 65  # 3dB波束寬度 (度)
        a_max = 30    # 最大衰減 (dB)
        
        # 相對於天線主波束的角度差
        phi_rel = az_deg - np.rad2deg(tx_orientation[1])  # 相對方位角
        phi_rel = np.mod(phi_rel + 180, 360) - 180  # 歸一化到 [-180, 180]
        
        # 水平方向圖
        a_h = -np.minimum(12 * (phi_rel / phi_3db)**2, a_max)
        
        # 垂直方向圖  
        theta_3db = 8   # 垂直3dB波束寬度
        theta_tilt = np.rad2deg(tx_orientation[0])  # 下傾角
        theta_rel = el_deg - theta_tilt
        
        # 垂直方向圖
        a_v = -np.minimum(12 * (theta_rel / theta_3db)**2, a_max)
        
        # 合併水平和垂直方向圖
        gain_db = -(a_h + a_v)  # 總增益
        
        # 限制在合理範圍
        gain_db = np.clip(gain_db, -30, 20)
        
    else:  # 各向性天線
        gain_db = np.zeros_like(X)
    
    return gain_db

def test_channel_models():
    """
    測試通道模型
    """
    tx_pos = [0, 0, 30]  # 基地台位置
    
    # 測試3GPP UMa
    print("計算 3GPP UMa 路徑增益...")
    pg_uma = compute_3gpp_uma_path_loss(tx_pos)
    
    # 測試Friis模型
    print("計算 Friis 路徑增益...")  
    pg_friis = compute_friis_path_loss(tx_pos)
    
    # 測試天線增益
    print("計算天線方向圖...")
    gain_map = compute_antenna_gain_map(tx_pos, tx_orientation=[np.deg2rad(-10), 0, 0])
    
    # 可視化
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    im1 = axes[0].imshow(pg_uma, cmap='viridis', origin='lower')
    axes[0].set_title('3GPP UMa Path Gain (dB)')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(pg_friis, cmap='viridis', origin='lower')
    axes[1].set_title('Friis Path Gain (dB)')
    plt.colorbar(im2, ax=axes[1])
    
    im3 = axes[2].imshow(gain_map, cmap='viridis', origin='lower')
    axes[2].set_title('Antenna Gain (dB)')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.show()
    
    return pg_uma, pg_friis, gain_map

if __name__ == "__main__":
    test_channel_models()