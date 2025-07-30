# Geo2SigMap: 路徑A實作指南

基於論文《Geo2SigMap: High-Fidelity RF Signal Mapping Using Geographic Databases》的完整實作，支援稀疏RSSI地圖補全。

## 🎯 核心特點

- **兩階段U-Net架構**：Stage-1學習各向性路徑增益，Stage-2結合稀疏量測補全信號地圖
- **無線上Sionna依賴**：推論時只需3GPP UMa/Friis模型，不用射線追蹤
- **靈活稀疏採樣**：支援1-200個稀疏點，多種採樣模式（隨機/蛇形/聚類）
- **端到端訓練**：完整的資料生成→訓練→推論流水線

## 📁 檔案結構

```
geo/
├── data_generation.py    # 離線資料生成（用Sionna）
├── channel_models.py     # 3GPP UMa/Friis通道模型
├── dataset.py           # PyTorch Dataset和DataLoader
├── models.py            # 兩階段U-Net模型
├── train.py             # 訓練腳本
├── inference.py         # 推論腳本（不用Sionna）
├── config.json          # 訓練配置
└── README.md           # 使用說明
```

## 🚀 快速開始

### 1. 環境準備

```bash
# 安裝依賴
pip install torch torchvision numpy matplotlib scikit-learn tensorboard
pip install sionna tensorflow  # 僅資料生成階段需要
```

### 2. 生成訓練資料

```bash
# 修改data_generation.py中的場景路徑，然後執行
python data_generation.py

# 這會生成 ./geo2sigmap_dataset/ 目錄包含：
# - sample_000000.npz (B, Piso, Pdir, S, P_uma, tx_position)
# - sample_000001.npz
# - ...
```

### 3. 兩階段訓練

```bash
# Stage 1: [B, P_UMa] -> Piso
python train.py --stage 1 --epochs 100

# Stage 2: [B, Piso, S_sparse, mask] -> S  
python train.py --stage 2 --epochs 100

# 或一次執行兩階段
python train.py
```

### 4. 推論（不用Sionna）

```bash
# 示範模式
python inference.py \
    --stage1_model ./outputs/checkpoints/stage1_best.pth \
    --stage2_model ./outputs/checkpoints/stage2_best.pth \
    --norm_stats ./geo2sigmap_dataset/normalization_stats.pkl \
    --demo

# 自定義基地台位置
python inference.py \
    --stage1_model ./outputs/checkpoints/stage1_best.pth \
    --stage2_model ./outputs/checkpoints/stage2_best.pth \
    --tx_pos 100 -50 25 \
    --demo
```

## 📊 模型架構

### Stage 1: U-Net-Iso
- **輸入**: [建築高度圖 B, UMa路徑增益 P_UMa] (2通道)
- **輸出**: 各向性路徑增益 Piso (1通道)
- **作用**: 學習從地理環境到基礎傳播特性的映射

### Stage 2: U-Net-Dir  
- **輸入**: [B, Piso, 稀疏RSSI, mask] (4通道)
- **輸出**: 完整RSSI地圖 S (1通道)
- **作用**: 利用稀疏量測和環境先驗補全信號地圖

## 🔧 關鍵參數

### 資料生成 (data_generation.py)
```python
SCENE_SIZE = 512      # 場景大小 512m×512m
GRID_SIZE = 128       # 網格解析度 128×128
RESOLUTION = 4        # 4m/pixel
FREQUENCY = 3.66e9    # 3.66 GHz
```

### 訓練配置 (config.json)
```json
{
  "batch_size": 16,
  "learning_rate": 0.001,
  "stage1_epochs": 100,
  "stage2_epochs": 100,
  "base_features": 64
}
```

### 推論設定 (inference.py)
```python
inference_config = {
    "channel_model": "3gpp_uma",  # 或 "friis"
    "grid_size": 128,
    "scene_size": 512
}
```

## 📈 使用你的資料

### 替換建築高度圖
```python
# 在inference.py中修改generate_building_height_map()
def generate_building_height_map(self, method="file", **kwargs):
    if method == "file":
        # 載入你的建築高度資料
        height_map = np.load(kwargs["file_path"])
    # ... 其他方法
```

### 輸入稀疏RSSI量測
```python
# 你的UAV量測資料
rssi_measurements = [-65.2, -72.1, -68.9, ...]  # dBm
coordinates = [(10, 20), (50, 30), ...]          # (x,y) 座標

# 執行預測
predicted_map, _ = inferencer.predict_from_measurements(
    tx_position=[0, 0, 30],
    rssi_measurements=rssi_measurements,
    coordinates=coordinates
)
```

## 🎛️ 進階功能

### 自定義採樣模式
```python
# 在dataset.py的generate_sparse_mask()中
# 支援 "random", "snake", "cluster" 三種模式
mask = dataset.generate_sparse_mask(
    shape=(128, 128), 
    n_sparse=50, 
    pattern="snake"
)
```

### 天線方向圖
```python
# 可選：加入天線增益作為額外通道
from channel_models import compute_antenna_gain_map

gain_map = compute_antenna_gain_map(
    tx_pos=[0, 0, 30],
    tx_orientation=[np.deg2rad(-10), 0, 0],  # 下傾10度
    antenna_pattern="3gpp_macro"
)
```

### 自定義損失函數
```python
# 在models.py的LossFunction類中
# 支援MSE、MAE、組合損失、一致性損失
loss = LossFunction.combined_loss(
    pred, target, mask, 
    mse_weight=1.0, mae_weight=0.1
)
```

## 📋 評估指標

訓練完成後會自動計算：
- **RMSE**: 均方根誤差
- **MAE**: 平均絕對誤差  
- **一致性損失**: 稀疏點預測vs實際的誤差

```bash
# 模型評估
python train.py --eval --stage 1  # 評估Stage 1
python train.py --eval --stage 2  # 評估Stage 2
```

## 🚨 注意事項

1. **資料生成階段需要Sionna**：只有在創建訓練資料時需要，推論時不用
2. **標準化很重要**：確保載入正確的normalization_stats.pkl
3. **GPU記憶體**：128×128網格約需4GB顯存，可調整batch_size
4. **場景適應**：不同環境可能需要微調超參數

## 🔄 完整流程範例

```bash
# 1. 生成資料（需要Sionna環境）
python data_generation.py

# 2. 訓練兩階段模型
python train.py --stage 1 --epochs 50
python train.py --stage 2 --epochs 50

# 3. 推論測試
python inference.py \
    --stage1_model ./outputs/checkpoints/stage1_best.pth \
    --stage2_model ./outputs/checkpoints/stage2_best.pth \
    --norm_stats ./geo2sigmap_dataset/normalization_stats.pkl \
    --demo

# 4. 可視化結果會自動顯示6個子圖：
#    - 建築高度圖、P_UMa、Stage1預測、稀疏輸入、最終預測、採樣覆蓋
```

## 📚 論文對應

這個實作對應論文中的：
- **Table I**: Sionna射線追蹤參數 → `data_generation.py`
- **Figure 2**: 兩階段架構 → `models.py`  
- **Section III-B**: 稀疏採樣策略 → `dataset.py`
- **Section IV**: 訓練程序 → `train.py`

## 🤝 擴展建議

- 加入更多通道模型（mmWave、室內等）
- 支援多基地台場景
- 整合真實OSM建築資料
- 加入時域變化建模
- 支援不同頻段和天線配置

## 📞 問題排解

如果遇到問題，請檢查：
1. 資料路徑是否正確
2. 模型檔案是否存在  
3. GPU記憶體是否足夠
4. 標準化統計量是否載入正確