import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle

class Geo2SigMapDataset(Dataset):
    """
    Geo2SigMap 資料集
    支援兩階段訓練：
    - Stage 1: [B, P_uma] -> Piso
    - Stage 2: [B, Piso, S_down] -> S
    """
    
    def __init__(self, data_dir, stage="stage1", transform=None, 
                 sparse_range=(1, 200), normalize=True,
                 split="train", split_ratio=0.8):
        """
        參數:
            data_dir: 資料目錄路徑
            stage: "stage1" 或 "stage2"
            transform: 資料增強
            sparse_range: 稀疏採樣點數範圍 (min, max)
            normalize: 是否標準化
            split: "train", "val", "test"
            split_ratio: 訓練集比例
        """
        self.data_dir = data_dir
        self.stage = stage
        self.transform = transform
        self.sparse_range = sparse_range
        self.normalize = normalize
        
        # 載入檔案列表
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        self.file_list.sort()
        
        # 資料切分
        n_total = len(self.file_list)
        n_train = int(n_total * split_ratio)
        n_val = int(n_total * 0.1)
        
        if split == "train":
            self.file_list = self.file_list[:n_train]
        elif split == "val":
            self.file_list = self.file_list[n_train:n_train+n_val]
        else:  # test
            self.file_list = self.file_list[n_train+n_val:]
        
        print(f"{split} 集：{len(self.file_list)} 個樣本")
        
        # 載入或計算標準化參數
        if normalize:
            self.setup_normalization()
    
    def setup_normalization(self):
        """
        設置標準化參數
        """
        stats_path = os.path.join(self.data_dir, "normalization_stats.pkl")
        
        if os.path.exists(stats_path):
            # 載入已存在的統計量
            with open(stats_path, 'rb') as f:
                self.norm_stats = pickle.load(f)
            print("載入標準化統計量")
        else:
            # 計算統計量
            print("計算標準化統計量...")
            self.compute_normalization_stats()
            
            # 儲存統計量
            with open(stats_path, 'wb') as f:
                pickle.dump(self.norm_stats, f)
            print("儲存標準化統計量")
    
    def compute_normalization_stats(self):
        """
        計算各通道的均值和標準差
        """
        # 收集所有資料
        all_B, all_Piso, all_Pdir, all_S, all_P_uma = [], [], [], [], []
        
        for file_name in self.file_list[:min(100, len(self.file_list))]:  # 取樣計算
            data = np.load(os.path.join(self.data_dir, file_name))
            
            all_B.append(data['B'].flatten())
            all_Piso.append(data['Piso'].flatten())
            all_Pdir.append(data['Pdir'].flatten())
            all_S.append(data['S'].flatten())
            all_P_uma.append(data['P_uma'].flatten())
        
        # 計算統計量（截幅後）
        def compute_stats(data_list, clip_range=None):
            all_data = np.concatenate(data_list)
            if clip_range:
                all_data = np.clip(all_data, clip_range[0], clip_range[1])
            return {"mean": np.mean(all_data), "std": np.std(all_data)}
        
        self.norm_stats = {
            "B": compute_stats(all_B, (0, 200)),  # 建築高度 0-200m
            "Piso": compute_stats(all_Piso, (-140, -40)),  # dB範圍
            "Pdir": compute_stats(all_Pdir, (-140, -40)),
            "S": compute_stats(all_S, (-140, -40)),
            "P_uma": compute_stats(all_P_uma, (-140, -40))
        }
    
    def normalize_data(self, data, key):
        """
        標準化資料
        """
        if not self.normalize:
            return data
            
        stats = self.norm_stats[key]
        return (data - stats["mean"]) / (stats["std"] + 1e-8)
    
    def denormalize_data(self, data, key):
        """
        反標準化
        """
        if not self.normalize:
            return data
            
        stats = self.norm_stats[key]
        return data * stats["std"] + stats["mean"]
    
    def generate_sparse_mask(self, shape, n_sparse=None, pattern="random"):
        """
        生成稀疏採樣mask
        """
        if n_sparse is None:
            n_sparse = np.random.randint(self.sparse_range[0], self.sparse_range[1]+1)
        
        mask = np.zeros(shape, dtype=np.float32)
        
        if pattern == "random":
            # 隨機採樣
            total_pixels = shape[0] * shape[1]
            indices = np.random.choice(total_pixels, n_sparse, replace=False)
            mask.flat[indices] = 1.0
            
        elif pattern == "snake":
            # 蛇形採樣（類似你原本的實作）
            step_y = max(1, shape[0] // int(np.sqrt(n_sparse)))
            step_x = max(1, shape[1] // int(np.sqrt(n_sparse)))
            
            count = 0
            for y in range(0, shape[0], step_y):
                if count >= n_sparse:
                    break
                if (y // step_y) % 2 == 0:
                    x_range = range(0, shape[1], step_x)
                else:
                    x_range = range(shape[1]-1, -1, -step_x)
                    
                for x in x_range:
                    if count >= n_sparse:
                        break
                    mask[y, x] = 1.0
                    count += 1
                    
        elif pattern == "cluster":
            # 聚類採樣
            n_clusters = max(1, n_sparse // 10)
            points_per_cluster = n_sparse // n_clusters
            
            for _ in range(n_clusters):
                # 隨機聚類中心
                cy = np.random.randint(5, shape[0]-5)
                cx = np.random.randint(5, shape[1]-5)
                
                # 在聚類周圍採樣
                for _ in range(points_per_cluster):
                    y = np.clip(cy + np.random.randint(-3, 4), 0, shape[0]-1)
                    x = np.clip(cx + np.random.randint(-3, 4), 0, shape[1]-1)
                    mask[y, x] = 1.0
        
        return mask
    
    def apply_augmentation(self, data_dict):
        """
        資料增強：旋轉和鏡射
        """
        if self.transform is None:
            return data_dict
        
        # 隨機選擇變換
        rotate_k = np.random.randint(0, 4)  # 0, 90, 180, 270度
        flip_h = np.random.choice([True, False])
        flip_v = np.random.choice([True, False])
        
        # 對所有2D陣列應用相同變換
        for key in data_dict:
            if len(data_dict[key].shape) == 2:
                # 旋轉
                data_dict[key] = np.rot90(data_dict[key], rotate_k)
                
                # 鏡射
                if flip_h:
                    data_dict[key] = np.fliplr(data_dict[key])
                if flip_v:
                    data_dict[key] = np.flipud(data_dict[key])
        
        return data_dict
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # 載入資料
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        data = np.load(file_path)
        
        # 提取陣列
        B = data['B'].astype(np.float32)
        Piso = data['Piso'].astype(np.float32) 
        Pdir = data['Pdir'].astype(np.float32)
        S = data['S'].astype(np.float32)
        P_uma = data['P_uma'].astype(np.float32)
        
        # 截幅
        B = np.clip(B, 0, 200)
        Piso = np.clip(Piso, -140, -40)
        Pdir = np.clip(Pdir, -140, -40)
        S = np.clip(S, -140, -40)
        P_uma = np.clip(P_uma, -140, -40)
        
        # 組織資料字典
        data_dict = {
            'B': B, 'Piso': Piso, 'Pdir': Pdir, 
            'S': S, 'P_uma': P_uma
        }
        
        # 資料增強
        if self.transform:
            data_dict = self.apply_augmentation(data_dict)
        
        # 標準化
        for key in data_dict:
            data_dict[key] = self.normalize_data(data_dict[key], key)
        
        if self.stage == "stage1":
            # Stage 1: [B, P_uma] -> Piso
            input_data = np.stack([data_dict['B'], data_dict['P_uma']], axis=0)
            target_data = data_dict['Piso'][np.newaxis, ...]  # 增加通道維度
            
        elif self.stage == "stage2":
            # Stage 2: 生成稀疏採樣
            mask = self.generate_sparse_mask(S.shape, pattern="random")
            S_sparse = data_dict['S'] * mask
            
            # [B, Piso, S_sparse, mask] -> S
            input_data = np.stack([
                data_dict['B'], 
                data_dict['Piso'],
                S_sparse,
                mask
            ], axis=0)
            target_data = data_dict['S'][np.newaxis, ...]
            
        else:
            raise ValueError(f"Unknown stage: {self.stage}")
        
        return {
            'input': torch.from_numpy(input_data),
            'target': torch.from_numpy(target_data),
            'metadata': {
                'file_name': self.file_list[idx],
                'tx_position': data.get('tx_position', [0, 0, 0])
            }
        }

def create_dataloaders(data_dir, batch_size=64, num_workers=4, 
                      stage="stage1", enable_augmentation=True):
    """
    創建資料載入器
    """
    # 資料增強
    transform = enable_augmentation if enable_augmentation else None
    
    # 創建資料集
    train_dataset = Geo2SigMapDataset(
        data_dir, stage=stage, transform=transform, split="train"
    )
    val_dataset = Geo2SigMapDataset(
        data_dir, stage=stage, transform=None, split="val"
    )
    test_dataset = Geo2SigMapDataset(
        data_dir, stage=stage, transform=None, split="test"
    )
    
    # 創建載入器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def visualize_sample(dataset, idx=0):
    """
    可視化資料樣本
    """
    sample = dataset[idx]
    input_data = sample['input'].numpy()
    target_data = sample['target'].numpy()
    
    if dataset.stage == "stage1":
        # Stage 1: [B, P_uma] -> Piso
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        im1 = axes[0].imshow(input_data[0], cmap='viridis')  # B
        axes[0].set_title('Building Height (B)')
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(input_data[1], cmap='viridis')  # P_uma
        axes[1].set_title('P_UMa')
        plt.colorbar(im2, ax=axes[1])
        
        im3 = axes[2].imshow(target_data[0], cmap='viridis')  # Piso
        axes[2].set_title('Target Piso')
        plt.colorbar(im3, ax=axes[2])
        
    elif dataset.stage == "stage2":
        # Stage 2: [B, Piso, S_sparse, mask] -> S
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        im1 = axes[0,0].imshow(input_data[0], cmap='viridis')  # B
        axes[0,0].set_title('Building Height (B)')
        plt.colorbar(im1, ax=axes[0,0])
        
        im2 = axes[0,1].imshow(input_data[1], cmap='viridis')  # Piso
        axes[0,1].set_title('Piso')
        plt.colorbar(im2, ax=axes[0,1])
        
        im3 = axes[0,2].imshow(input_data[2], cmap='viridis')  # S_sparse
        axes[0,2].set_title('Sparse S')
        plt.colorbar(im3, ax=axes[0,2])
        
        im4 = axes[1,0].imshow(input_data[3], cmap='gray')     # mask
        axes[1,0].set_title('Mask')
        plt.colorbar(im4, ax=axes[1,0])
        
        im5 = axes[1,1].imshow(target_data[0], cmap='viridis') # S target
        axes[1,1].set_title('Target S')
        plt.colorbar(im5, ax=axes[1,1])
        
        axes[1,2].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 測試資料集
    if os.path.exists("./geo2sigmap_dataset"):
        # Stage 1 測試
        dataset_s1 = Geo2SigMapDataset("./geo2sigmap_dataset", stage="stage1")
        print(f"Stage 1 資料集大小: {len(dataset_s1)}")
        
        if len(dataset_s1) > 0:
            visualize_sample(dataset_s1, 0)
        
        # Stage 2 測試
        dataset_s2 = Geo2SigMapDataset("./geo2sigmap_dataset", stage="stage2")
        print(f"Stage 2 資料集大小: {len(dataset_s2)}")
        
        if len(dataset_s2) > 0:
            visualize_sample(dataset_s2, 0)
    else:
        print("請先執行 data_generation.py 生成資料集")