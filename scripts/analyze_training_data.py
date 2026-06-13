import torch
import os
import sys
import matplotlib
matplotlib.use('Agg')  # 非交互式环境使用Agg后端
import matplotlib.pyplot as plt
import numpy as np

# 添加项目根目录到路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print(f"项目根目录: {PROJECT_ROOT}")
print(f"Python版本: {sys.version}")
print(f"PyTorch版本: {torch.__version__}")
print(f"是否有CUDA: {torch.cuda.is_available()}")

from data.spacenet_dataset import SpaceNetRoadDataset
from torch.utils.data import DataLoader
import yaml

def get_train_loader(config):
    """获取训练数据加载器"""
    aoi_dir = config["data"]["aoi_dir"]
    labels_dir = config["data"]["labels_dir"]
    image_size = tuple(config["data"].get("image_size", [224, 224]))
    batch_size = config["training"].get("batch_size", 2)
    num_workers = config["data"].get("num_workers", 0)
    
    dataset = SpaceNetRoadDataset(
        aoi_dir=aoi_dir,
        labels_dir=labels_dir,
        image_size=image_size,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True
    )
    
    return loader

def analyze_training_data():
    """分析训练数据"""
    print("\n=== 开始分析训练数据 ===")
    # 加载配置
    try:
        with open("configs/config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        print("✓ 成功加载配置文件")
    except Exception as e:
        print(f"❌ 加载配置文件失败: {e}")
        return None
    
    # 获取数据加载器
    try:
        loader = get_train_loader(config)
        print(f"✓ 成功创建数据加载器，批次大小: {config['training'].get('batch_size', 2)}")
    except Exception as e:
        print(f"❌ 创建数据加载器失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 获取一个批次的数据
    print("\n正在加载数据...")
    try:
        batch = next(iter(loader))
        print("✓ 成功加载数据批次")
    except Exception as e:
        print(f"❌ 加载数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 提取数据
    try:
        images = batch["image"]
        masks = batch["mask"]
        heatmaps = batch["intersection"]
        orients = batch["orientation"]
        sample_ids = batch["sample_id"]
        
        print(f"批次大小: {images.shape[0]}")
        print(f"图像形状: {images.shape}")
        print(f"掩码形状: {masks.shape}")
        print(f"热图形状: {heatmaps.shape}")
        print(f"方向场形状: {orients.shape}")
        print(f"样本ID: {sample_ids}")
        
        # 分析第一个样本
        print("\n=== 分析第一个样本 ===")
        
        # 1. 图像分析
        img_np = images[0].permute(1, 2, 0).cpu().numpy()
        print(f"图像均值: {img_np.mean():.3f}")
        print(f"图像最大值: {img_np.max():.3f}")
        print(f"图像最小值: {img_np.min():.3f}")
        
        # 2. 掩码分析
        mask_np = masks[0, 0].cpu().numpy()
        road_ratio = (mask_np > 0.5).sum() / mask_np.size
        print(f"掩码中道路像素占比: {road_ratio:.3%}")
        print(f"掩码最大值: {mask_np.max():.3f}")
        print(f"掩码最小值: {mask_np.min():.3f}")
        
        # 3. 热图分析
        heatmap_np = heatmaps[0, 0].cpu().numpy()
        print(f"热图最大值: {heatmap_np.max():.3f}")
        print(f"热图均值: {heatmap_np.mean():.3f}")
        
        # 4. 方向场分析
        orient_np = orients[0].cpu().numpy()
        dx = orient_np[0]
        dy = orient_np[1]
        conf = orient_np[2]
        norms = np.sqrt(dx**2 + dy**2)
        print(f"方向场向量模长均值: {norms.mean():.3f}")
        print(f"方向场置信度均值: {conf.mean():.3f}")
        
        # 5. 可视化
        print("\n正在生成可视化结果...")
        os.makedirs("logs/debug", exist_ok=True)
        
        # 图像
        plt.figure(figsize=(12, 3))
        
        # 原始图像
        plt.subplot(1, 4, 1)
        img_np = np.clip(img_np, 0, 1)
        plt.imshow(img_np)
        plt.title('Image')
        plt.axis('off')
        
        # 真实掩码
        plt.subplot(1, 4, 2)
        plt.imshow(mask_np, cmap='gray', vmin=0, vmax=1)
        plt.title('Mask (GT Road)')
        plt.axis('off')
        
        # 热图
        plt.subplot(1, 4, 3)
        plt.imshow(heatmap_np, cmap='hot', vmin=0, vmax=1)
        plt.title('Intersection Heatmap')
        plt.axis('off')
        
        # 方向场向量
        plt.subplot(1, 4, 4)
        # 只在道路区域显示方向场
        road_mask = mask_np > 0.5
        if road_mask.sum() > 0:
            # 每隔几个像素绘制一个向量
            step = 10
            y, x = np.where(road_mask)
            y = y[::step]
            x = x[::step]
            dx_sub = dx[y, x]
            dy_sub = dy[y, x]
            plt.imshow(img_np)
            plt.quiver(x, y, dx_sub, dy_sub, color='red', scale=20, headwidth=3, headlength=3)
            plt.title('Direction Field')
        else:
            plt.imshow(img_np)
            plt.title('No Road Pixels')
        plt.axis('off')
        
        plt.tight_layout()
        save_path = "logs/debug/data_analysis.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"可视化结果已保存到: {save_path}")
        
        # 诊断问题
        print("\n=== 诊断结果 ===")
        if img_np.mean() < 0.1 or img_np.mean() > 0.9:
            print("⚠️ 警告: 图像均值异常，可能存在归一化问题")
        if road_ratio < 0.01:
            print("⚠️ 警告: 道路像素占比过低，可能导致模型学习困难")
        if heatmap_np.max() < 0.1:
            print("⚠️ 警告: 交叉口热图值过低，可能没有检测到交叉口")
        if norms.mean() < 0.9:
            print("⚠️ 警告: 方向场向量模长异常，可能存在归一化问题")
        
        return {
            "image_mean": img_np.mean(),
            "mask_road_ratio": road_ratio,
            "heatmap_max": heatmap_np.max(),
            "orientation_norm_mean": norms.mean()
        }
        
    except Exception as e:
        print(f"❌ 分析数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    try:
        result = analyze_training_data()
        if result:
            print("\n脚本执行完成！")
            print(f"分析结果: {result}")
        else:
            print("\n脚本执行失败！")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)