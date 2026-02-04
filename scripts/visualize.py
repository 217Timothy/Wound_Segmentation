import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt

current_root = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_root)
sys.path.append(project_root)

def get_args():
    parser = argparse.ArgumentParser()
    
    # 必要參數：模型版本與資料集
    parser.add_argument("--version", type=str, required=True,
                        help="要使用的模型版本")
    parser.add_argument("--run_name", type=str, required=True,
                        help="第幾次跑")
    
    # 輸入與輸出根目錄
    parser.add_argument("--in_root", type=str, default="data/processed",
                        help="輸入圖片的資料夾路徑")
    parser.add_argument("--out_root", type=str, default="results/runs",
                        help="輸出結果的根目錄")
    
    return parser.parse_args()


def highlight_max(ax, x_data, y_data, color="lightcoral"):
    max_val = y_data.max()
    max_idx = y_data.idxmax()
    max_epoch = x_data[max_idx]
    
    ax.scatter(max_epoch, max_val, color=color, s=120, zorder=5, edgecolors='white', linewidth=2)
    
    ax.annotate(f'Max: {max_val:.4f}\n(Epoch {max_epoch})', 
                xy=(max_epoch, max_val), 
                xytext=(max_epoch, max_val - (max_val * 0.1)), # 文字稍微往下移一點，避免擋住線
                color='black',
                fontsize=10,
                fontweight='bold',
                arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=8),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.9))


def highlight_min(ax, x_data, y_data, color="lightcoral"):
    mim_val = y_data.min()
    min_idx = y_data.idxmin()
    min_epoch = x_data[min_idx]
    
    ax.scatter(min_epoch, mim_val, color=color, s=120, zorder=5, edgecolors='white', linewidth=2)
    
    ax.annotate(f'Min: {mim_val:.4f}\n(Epoch {min_epoch})', 
                xy=(min_epoch, mim_val), 
                xytext=(min_epoch, mim_val - (mim_val * 0.1)), # 文字稍微往下移一點，避免擋住線
                color='black',
                fontsize=10,
                fontweight='bold',
                arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=8),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.9))


def main():
    args = get_args()
    
    out_fig_dir = os.path.join(args.out_root, args.version, args.run_name)
    log_path = os.path.join("logs", args.version, f"{args.run_name}.csv")
    if not os.path.exists(log_path):
        print(f"[Error] 找不到 Log 檔案：{log_path}")
        return
    os.makedirs(out_fig_dir, exist_ok=True)
    
    print(f"[INFO] Reading log from: {log_path}")
    try:
        df = pd.read_csv(log_path)
    except Exception as e:
        print(f"[Error] 讀取 CSV 失敗: {e}")
        return
    
    plt.style.use("ggplot")
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 6))
    
    epochs = df["epoch"]
    
    # 1. Val Dice Score Curve
    ax1.plot(epochs, df["val_dice"], label="Val Dice", color="royalblue", linewidth=1, linestyle="-", marker="o")
    ax1.set_title(f"Validation Dice per Epoch ({args.version} - {args.run_name})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Dice")
    ax1.legend()
    ax1.grid(True)
    
    highlight_max(ax1, epochs, df["val_dice"])
    
    # 2. Val IoU Score Curve
    ax2.plot(epochs, df["val_iou"], label="Val IoU", color="royalblue", linewidth=1, linestyle="-", marker="o")
    ax2.set_title(f"Validation IoU per Epoch ({args.version} - {args.run_name})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("IoU")
    ax2.legend()
    ax2.grid(True)
    
    highlight_max(ax2, epochs, df["val_iou"])
    
    # 3. Val Recall Score Curve
    ax3.plot(epochs, df["val_recall"], label="Val Recall", color="royalblue", linewidth=1, linestyle="-", marker="o")
    ax3.set_title(f"Validation Recall per Epoch ({args.version} - {args.run_name})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Recall")
    ax2.legend()
    ax2.grid(True)
    
    highlight_max(ax3, epochs, df["val_recall"])
    
    # 4. Loss Curve
    ax4.plot(epochs, df["train_loss"], label="Train Loss", color="tab:orange", linewidth=2, linestyle="--")
    ax4.plot(epochs, df["val_loss"], label="Val Loss", color="royalblue", linewidth=2, linestyle="-", marker="o", markersize=4)
    ax4.set_title(f"Loss Curve ({args.version} - {args.run_name})")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Loss")
    ax4.legend()
    ax4.grid(True)
    
    highlight_min(ax4, epochs, df["val_loss"])
    
    
    save_path = os.path.join(out_fig_dir, f"plot")
    plt.savefig(save_path, dpi=300)
    print(f"✅ Plot saved to: {save_path}")


if __name__ == "__main__":
    main()