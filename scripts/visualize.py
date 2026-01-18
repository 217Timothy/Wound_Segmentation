import os
import sys
import argparse
import pandas as pd
import matplotlib as plt

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
    parser.add_argument("--out_root", type=str, default="results/run",
                        help="輸出結果的根目錄")
    
    return parser.parse_args()


def main():
    args = get_args()


if __name__ == "__main__":
    main()