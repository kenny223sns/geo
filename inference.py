import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from datetime import datetime

from models import Geo2SigMapModel, UNetIso, UNetDir
from channel_models import compute_3gpp_uma_path_loss, compute_friis_path_loss, compute_antenna_gain_map
from dataset import Geo2SigMapDataset

class Geo2SigMapInference:
    """
    Geo2SigMap 推論器（線上部署用，不需要Sionna）
    """
    def __init__(self, stage1_model_path, stage2_model_path, config=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"推論設備: {self.device}")
        
        # 預設配置
        self.config = config or {
            "grid_size": 128,
            "scene_size": 512,
            "frequency": 3.66e9,
            "rx_height": 1.5,
            "channel_model": "3gpp_uma"  # 或 "friis"
        }
        
        # 載入模型
        self.load_models(stage1_model_path, stage2_model_path)
        
        # 標準化參數（需要從訓練資料獲得）
        self.norm_stats = None
        
    def load_models(self, stage1_path, stage2_path):
        """
        載入預訓練模型
        """
        # 載入Stage 1模型
        self.model_iso = UNetIso().to(self.device)
        stage1_checkpoint = torch.load(stage1_path, map_location=self.device)
        self.model_iso.load_state_dict(stage1_checkpoint['model_state_dict'])
        self.model_iso.eval()
        print(f"載入Stage 1模型: {stage1_path}")
        
        # 載入Stage 2模型
        self.model_dir = UNetDir().to(self.device)
        stage2_checkpoint = torch.load(stage2_path, map_location=self.device)
        self.model_dir.load_state_dict(stage2_checkpoint['model_state_dict'])
        self.model_dir.eval()
        print(f"載入Stage 2模型: {stage2_path}")
        
        # 創建完整模型
        self.full_model = Geo2SigMapModel().to(self.device)
        self.full_model.unet_iso = self.model_iso
        self.full_model.unet_dir = self.model_dir
        self.full_model.eval()
    
    def load_normalization_stats(self, stats_path):
        """
        載入標準化統計量
        """
        import pickle
        with open(stats_path, 'rb') as f:
            self.norm_stats = pickle.load(f)
        print(f"載入標準化統計量: {stats_path}")
    
    def normalize_data(self, data, key):
        """標準化資料"""
        if self.norm_stats is None:
            return data
        stats = self.norm_stats[key]
        return (data - stats["mean"]) / (stats["std"] + 1e-8)
    
    def denormalize_data(self, data, key):
        """反標準化"""
        if self.norm_stats is None:
            return data
        stats = self.norm_stats[key]
        return data * stats["std"] + stats["mean"]
    
    def generate_building_height_map(self, method="random", **kwargs):
        """
        生成或載入建築高度圖 B
        
        參數:
            method: "random", "osm", "file" 等
        """
        grid_size = self.config["grid_size"]
        
        if method == "random":
            # 模擬建築高度（實際應用中從OSM/CAD獲取）
            np.random.seed(kwargs.get("seed", 42))
            height_map = np.random.uniform(0, 50, (grid_size, grid_size))
            
        elif method == "file":
            # 從檔案載入
            file_path = kwargs.get("file_path")
            height_map = np.load(file_path)
            
        elif method == "osm":
            # 從OSM資料生成（需要額外實作）
            raise NotImplementedError("OSM建築高度提取尚未實作")
            
        else:
            # 平坦地形
            height_map = np.zeros((grid_size, grid_size))
        
        return height_map.astype(np.float32)
    
    def compute_puma_map(self, tx_position):
        """
        計算P_UMa地圖（即時計算，極快）
        """
        if self.config["channel_model"] == "3gpp_uma":
            return compute_3gpp_uma_path_loss(
                tx_position,
                rx_height=self.config["rx_height"],
                grid_size=self.config["grid_size"],
                scene_size=self.config["scene_size"],
                frequency=self.config["frequency"]
            )
        else:  # friis
            return compute_friis_path_loss(
                tx_position,
                rx_height=self.config["rx_height"],
                grid_size=self.config["grid_size"],
                scene_size=self.config["scene_size"],
                frequency=self.config["frequency"]
            )
    
    def create_sparse_rssi_map(self, rssi_measurements, coordinates, method="nearest"):
        """
        將稀疏RSSI量測投影到網格
        
        參數:
            rssi_measurements: RSSI數值列表 (dBm)
            coordinates: 座標列表 [(x, y), ...]
            method: 插值方法
        """
        grid_size = self.config["grid_size"]
        scene_size = self.config["scene_size"]
        
        # 初始化稀疏地圖和mask
        sparse_map = np.zeros((grid_size, grid_size), dtype=np.float32)
        mask = np.zeros((grid_size, grid_size), dtype=np.float32)
        
        # 座標轉換：實際座標 -> 網格索引
        for rssi, (x, y) in zip(rssi_measurements, coordinates):
            # 轉換為網格索引
            grid_x = int((x + scene_size/2) / scene_size * grid_size)
            grid_y = int((y + scene_size/2) / scene_size * grid_size)
            
            # 確保在範圍內
            if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                sparse_map[grid_y, grid_x] = rssi
                mask[grid_y, grid_x] = 1.0
        
        return sparse_map, mask
    
    def predict_rssi_map(self, tx_position, building_height_map, 
                        sparse_rssi_map, mask, return_intermediate=False):
        """
        執行完整的RSSI地圖預測
        
        參數:
            tx_position: 基地台位置 [x, y, z]
            building_height_map: 建築高度圖 (H, W)
            sparse_rssi_map: 稀疏RSSI地圖 (H, W)
            mask: 0/1 mask (H, W)
            return_intermediate: 是否返回中間結果
        
        返回:
            predicted_rssi: 完整RSSI地圖 (H, W)
            (可選) intermediate_results: Stage 1輸出等
        """
        with torch.no_grad():
            # === 1. 計算P_UMa ===
            P_uma = self.compute_puma_map(tx_position)
            
            # === 2. 資料預處理 ===
            # 截幅
            B = np.clip(building_height_map, 0, 200)
            P_uma_clipped = np.clip(P_uma, -140, -40)
            S_sparse_clipped = np.clip(sparse_rssi_map, -140, -40)
            
            # 標準化（如果有統計量）
            if self.norm_stats:
                B_norm = self.normalize_data(B, "B")
                P_uma_norm = self.normalize_data(P_uma_clipped, "P_uma")
                S_sparse_norm = self.normalize_data(S_sparse_clipped, "S")
            else:
                B_norm, P_uma_norm, S_sparse_norm = B, P_uma_clipped, S_sparse_clipped
            
            # 轉換為張量
            B_tensor = torch.from_numpy(B_norm).unsqueeze(0).unsqueeze(0).to(self.device)
            P_uma_tensor = torch.from_numpy(P_uma_norm).unsqueeze(0).unsqueeze(0).to(self.device)
            S_sparse_tensor = torch.from_numpy(S_sparse_norm).unsqueeze(0).unsqueeze(0).to(self.device)
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # === 3. Stage 1: [B, P_UMa] -> Piso ===
            stage1_input = torch.cat([B_tensor, P_uma_tensor], dim=1)
            Piso_pred = self.model_iso(stage1_input)
            
            # === 4. Stage 2: [B, Piso, S_sparse, mask] -> S ===
            stage2_input = torch.cat([B_tensor, Piso_pred, S_sparse_tensor, mask_tensor], dim=1)
            S_pred = self.model_dir(stage2_input)
            
            # === 5. 後處理 ===
            # 轉回numpy
            Piso_np = Piso_pred.cpu().numpy()[0, 0]
            S_np = S_pred.cpu().numpy()[0, 0]
            
            # 反標準化
            if self.norm_stats:
                S_final = self.denormalize_data(S_np, "S")
                Piso_final = self.denormalize_data(Piso_np, "Piso")
            else:
                S_final = S_np
                Piso_final = Piso_np
            
            if return_intermediate:
                intermediate = {
                    "P_uma": P_uma,
                    "Piso_pred": Piso_final,
                    "building_height": building_height_map,
                    "sparse_input": sparse_rssi_map,
                    "mask": mask
                }
                return S_final, intermediate
            else:
                return S_final
    
    def predict_from_measurements(self, tx_position, rssi_measurements, coordinates,
                                building_method="random", visualize=True, **kwargs):
        """
        從實際量測資料預測完整RSSI地圖
        
        參數:
            tx_position: 基地台位置
            rssi_measurements: RSSI量測值列表
            coordinates: 量測位置座標列表
            building_method: 建築高度圖生成方法
        """
        print(f"基地台位置: {tx_position}")
        print(f"稀疏量測點數: {len(rssi_measurements)}")
        
        # 生成建築高度圖
        B = self.generate_building_height_map(method=building_method, **kwargs)
        
        # 創建稀疏RSSI地圖
        S_sparse, mask = self.create_sparse_rssi_map(rssi_measurements, coordinates)
        
        # 預測
        S_pred, intermediate = self.predict_rssi_map(
            tx_position, B, S_sparse, mask, return_intermediate=True
        )
        
        # 可視化
        if visualize:
            self.visualize_prediction(S_pred, intermediate)
        
        return S_pred, intermediate
    
    def visualize_prediction(self, predicted_rssi, intermediate_results):
        """
        可視化預測結果
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 建築高度圖
        im1 = axes[0,0].imshow(intermediate_results["building_height"], cmap='viridis', origin='lower')
        axes[0,0].set_title('Building Height Map (m)')
        plt.colorbar(im1, ax=axes[0,0])
        
        # P_UMa
        im2 = axes[0,1].imshow(intermediate_results["P_uma"], cmap='viridis', origin='lower')
        axes[0,1].set_title('P_UMa Path Gain (dB)')
        plt.colorbar(im2, ax=axes[0,1])
        
        # Stage 1預測結果
        im3 = axes[0,2].imshow(intermediate_results["Piso_pred"], cmap='viridis', origin='lower')
        axes[0,2].set_title('Stage 1: Predicted Piso (dB)')
        plt.colorbar(im3, ax=axes[0,2])
        
        # 稀疏輸入
        sparse_masked = np.where(intermediate_results["mask"] > 0, 
                               intermediate_results["sparse_input"], np.nan)
        im4 = axes[1,0].imshow(sparse_masked, cmap='viridis', origin='lower')
        axes[1,0].set_title('Sparse RSSI Input (dBm)')
        plt.colorbar(im4, ax=axes[1,0])
        
        # 最終預測結果
        im5 = axes[1,1].imshow(predicted_rssi, cmap='viridis', origin='lower')
        axes[1,1].set_title('Final Predicted RSSI (dBm)')
        plt.colorbar(im5, ax=axes[1,1])
        
        # 採樣點覆蓋率
        coverage = intermediate_results["mask"]
        im6 = axes[1,2].imshow(coverage, cmap='gray', origin='lower')
        axes[1,2].set_title('Sampling Coverage')
        plt.colorbar(im6, ax=axes[1,2])
        
        plt.tight_layout()
        plt.show()
        
        # 統計資訊
        n_sparse = int(np.sum(intermediate_results["mask"]))
        total_pixels = intermediate_results["mask"].size
        coverage_ratio = n_sparse / total_pixels * 100
        
        print(f"\n=== 預測統計 ===")
        print(f"稀疏採樣點數: {n_sparse}")
        print(f"總像素數: {total_pixels}")
        print(f"覆蓋率: {coverage_ratio:.2f}%")
        print(f"預測RSSI範圍: [{np.min(predicted_rssi):.1f}, {np.max(predicted_rssi):.1f}] dBm")

def main():
    parser = argparse.ArgumentParser(description="Geo2SigMap 推論腳本")
    parser.add_argument("--stage1_model", type=str, required=True, help="Stage 1模型路徑")
    parser.add_argument("--stage2_model", type=str, required=True, help="Stage 2模型路徑")
    parser.add_argument("--norm_stats", type=str, help="標準化統計量路徑")
    parser.add_argument("--tx_pos", type=float, nargs=3, default=[0, 0, 30], 
                       help="基地台位置 [x, y, z]")
    parser.add_argument("--demo", action="store_true", help="執行示範")
    
    args = parser.parse_args()
    
    # 創建推論器
    inference_config = {
        "grid_size": 128,
        "scene_size": 512,
        "frequency": 3.66e9,
        "rx_height": 1.5,
        "channel_model": "3gpp_uma"
    }
    
    inferencer = Geo2SigMapInference(
        args.stage1_model, 
        args.stage2_model, 
        config=inference_config
    )
    
    # 載入標準化統計量
    if args.norm_stats and os.path.exists(args.norm_stats):
        inferencer.load_normalization_stats(args.norm_stats)
    
    if args.demo:
        # 示範模式：模擬稀疏量測
        print("=== 示範模式 ===")
        
        # 模擬UAV蛇形採樣軌跡
        np.random.seed(42)
        n_measurements = 50
        
        # 生成蛇形採樣座標
        measurements_x = np.linspace(-200, 200, n_measurements)
        measurements_y = np.linspace(-200, 200, n_measurements)
        coordinates = [(x, y) for x, y in zip(measurements_x, measurements_y)]
        
        # 模擬RSSI量測（加入噪聲）
        base_rssi = -60  # dBm
        rssi_measurements = []
        for x, y in coordinates:
            distance = np.sqrt(x**2 + y**2 + (args.tx_pos[2] - 1.5)**2)
            rssi = base_rssi - 20*np.log10(distance/10) + np.random.normal(0, 3)
            rssi_measurements.append(rssi)
        
        # 執行預測
        predicted_map, intermediate = inferencer.predict_from_measurements(
            args.tx_pos, rssi_measurements, coordinates,
            building_method="random", visualize=True
        )
        
        print("示範完成!")
    
    else:
        print("請提供 --demo 參數執行示範，或修改程式碼加入您的量測資料")

if __name__ == "__main__":
    main()