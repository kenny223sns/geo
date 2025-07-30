# ✅ 大場景切片數據生成 - 修正總結

基於你提供的詳細落地實現建議，已完成以下關鍵修正：

## 🔧 核心修正項目

### 1. ✅ 功率單位轉換修正（最重要！）
**問題**：原代碼使用錯誤的單位轉換，會導致訓練數據歪掉
```python
# ❌ 錯誤做法
piso_db = 10 * np.log10(piso_linear / 1e-3) - 30  # 會導致單位混亂
```

**修正**：使用0dBm發射功率 + 正確轉換函數
```python
# ✅ 正確做法
tx = Transmitter(position=tx_pos, power_dbm=0.0)  # 0 dBm!!
Piso_dB = w_to_dbm(rm.rss[0].numpy())  # 直接得到路徑增益(dB)

def w_to_dbm(p_w):
    p_mw = np.clip(p_w, 1e-18, None) * 1e3  # W → mW
    return 10.0 * np.log10(p_mw)
```

### 2. ✅ 3GPP TR38901真實天線方向圖
**問題**：原來使用簡化的iso天線模擬方向性
```python
# ❌ 舊版
TX_ARRAY_DIR = {"pattern": "iso", "num_rows": 8}  # 用array factor模擬
```

**修正**：使用Sionna內建的3GPP標準天線
```python
# ✅ 新版
TX_ARRAY_3GPP = {
    "pattern": "tr38901",     # 3GPP TR 38.901標準方向圖
    "polarization": "V",      # 單極化
    # θ_3dB=φ_3dB=65°, SLA_v=30dB, G_max=8dBi (內建參數)
}

# 加入電下傾
orientation = [
    np.deg2rad(az),                    # yaw (方位角)
    np.deg2rad(-elec_downtilt_deg),    # pitch (下傾)
    0.0                                # roll
]
```

### 3. ✅ 固定地面測量高度
**問題**：RX_HEIGHT參數不一致
```python
# ✅ 統一設定
Z_PLANE = 1.5  # 固定地面上方1.5m
rm_args = dict(RM_COMMON, center=[cx, cy, Z_PLANE])
```

### 4. ✅ 正確的鏈路預算計算
**問題**：在Sionna中重複加功率導致雙重計數
```python
# ❌ 錯誤：在Sionna裡已含TX power又再相加
S = ptx + gtx + (已含功率的rss轉dBm) + grx - il

# ✅ 正確：在numpy中做鏈路預算
Pdir_dB = w_to_dbm(rm.rss[0].numpy())  # 0dBm發射 → 純路徑增益
S = Ptx + Gtx + Pdir_dB + Grx - IL      # 全部dB/dBm單位一致
```

### 5. ✅ 大場景切片策略
**核心邏輯**：
- 大場景 → `generate_tile_centers()` → 512×512m tiles
- 每tile → `run_piso_pdir_for_tile()` → Piso + 4方位Pdir
- 每方位 → `link_budget_from_pdir()` → 隨機參數生成S
- 儲存格式：`tile_00001_az00.npz` (tile_id + 方位編號)

## 📁 新檔案結構

### `data_generation_clean.py` - 修正版主檔
- ✅ 所有單位轉換問題已修正
- ✅ 3GPP TR38901天線已實現
- ✅ 包含單位驗證函數 `validate_single_tile()`
- ✅ 互動式選單：驗證/生成/兩者皆有

### `test_corrected_pipeline.py` - 測試腳本
- 檢查場景檔案存在性
- 測試API導入
- 驗證與dataset.py兼容性

### `CORRECTIONS_SUMMARY.md` - 本文檔
- 詳細記錄所有修正項目
- 提供使用指南

## 🚀 使用指南

### 1. 運行修正版數據生成
```bash
python3 data_generation_clean.py
```
選擇模式：
- `1` = 單位驗證（檢查鏈路預算正確性）
- `2` = 生成資料集
- `3` = 兩者都做（推薦）

### 2. 預期輸出
- **nnn場景**: 2km×2km → ~16 tiles × 4方位 = **64樣本**  
- **NYCU場景**: 1.6km×1.6km → ~9 tiles × 4方位 = **36樣本**
- 總計：**~100樣本**（用於測試）

### 3. 檔案命名格式
```
geo2sigmap_tiles_scene1/
├── tile_00000_az00.npz  # tile 0, 方位 0° 
├── tile_00000_az01.npz  # tile 0, 方位 90°
├── tile_00000_az02.npz  # tile 0, 方位 180°
├── tile_00000_az03.npz  # tile 0, 方位 270°
├── tile_00001_az00.npz  # tile 1, 方位 0°
└── ...
```

### 4. NPZ檔案內容
每個檔案包含：
```python
{
    'B': (128,128),           # 建築高度圖
    'Piso': (128,128),        # 各向性路徑增益(dB)
    'Pdir': (128,128),        # 定向路徑增益(dB) 
    'S': (128,128),           # 完整信號強度(dBm)
    'P_uma': (128,128),       # UMa路徑模型粗先驗(dB)
    'tx_position': (3,),      # BS位置 [x,y,z]
    'tile_center': (2,),      # tile中心 [cx,cy]
    'z_plane': (1,),          # 測量高度 [1.5]
    'azimuth': (1,)           # 天線方位角(度)
}
```

## ⚠️ 單位驗證重要性

修正前後的差異可能很大！建議**必須**先運行單位驗證：
```python
validate_single_tile(scene_path, bounds, test_tile_id=0)
```

驗證邏輯：
1. 用鏈路預算：`S = 30 + 0 + Pdir + 0 - 0 = 30 + Pdir`
2. 直接30dBm：`Transmitter(power_dbm=30.0)` 
3. 比較兩者差異應 < 0.5dB

如果驗證失敗，表示單位轉換仍有問題。

## 🔄 與現有代碼兼容性

### dataset.py
已更新metadata欄位支援新格式：
```python
'metadata': {
    'file_name': file_name,
    'tx_position': data.get('tx_position', [0,0,0]),
    'tile_center': data.get('tile_center', [0,0]),      # 新增
    'orientation': data.get('orientation', 0.0)         # 新增
}
```

### train.py 
無需修改，輸入/輸出格式保持一致：
- Stage 1: `[B, P_uma] → Piso`
- Stage 2: `[B, Piso, S_sparse, mask] → S`

## 🎯 下一步建議

1. **先驗證**：運行單位驗證確保修正成功
2. **小規模測試**：用當前bounds生成~100樣本測試訓練
3. **擴大規模**：成功後調大scene_bounds生成更多數據
4. **優化參數**：調整SAMPLES_PER_TX (5M→7M) 提高精度

## 🔧 參數調整參考

```python
# 提高精度（更慢）
SAMPLES_PER_TX = 7_000_000  # 7M射線

# 增加數據量
scene_bounds = [
    [-2000, 2000, -2000, 2000],  # 4km×4km → ~256樣本
    [-1500, 1500, -1500, 1500],  # 3km×3km → ~144樣本  
]

# 增加重疊（更穩定邊界）
stride = 384  # 25%重疊 (vs TILE=512無重疊)
```

---

**總結**：這次修正解決了最關鍵的單位轉換問題，並實現了真實的3GPP天線模型。現在你可以用2個XML檔案生成高質量的訓練數據，無需手工製作更多場景！