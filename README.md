# ITRI 多類別傷口分割專案

多類別傷口語意分割。目標：每一個傷口類別的 Dice 都 ≥ 0.80。

支援三種訓練模式，共用同一套 `woundseg` 函式庫：

| 模式 | 入口腳本 | 說明 |
|------|----------|------|
| 一般訓練 | `scripts/train.py` | 在大型分割資料集上從頭預訓練 |
| 多類別傷口微調 | `scripts/finetune_wound.py` | 載入預訓練權重，微調多類別傷口 |
| TKR 傷口微調 | `scripts/finetune_tkr.py` | 在 TKR 膝關節資料集上微調 |

## 專案結構

```
ITRI_Segmentation_Project/
├── woundseg/              # 核心函式庫（可被 import）
│   ├── config.py          #   設定載入：預設值 + YAML + 命令列
│   ├── data/              #   Dataset 與資料增強
│   ├── models/            #   UNet / ResUnet / EfficientUnet + build_model 工廠
│   ├── losses/            #   Dice / Tversky / Focal-Tversky + build_loss 工廠
│   ├── metrics/           #   Dice / IoU / Recall / Precision
│   ├── engine/            #   訓練迴圈、驗證、推論（run_training 在 loop.py）
│   ├── postprocess.py     #   預測遮罩的形態學後處理
│   └── utils/             #   裝置、隨機種子、checkpoint、CSV log、視覺化
│
├── scripts/               # 精簡的入口腳本（不含商業邏輯）
│   ├── train.py
│   ├── finetune_wound.py
│   ├── finetune_tkr.py
│   ├── predict.py         #   批次推論
│   ├── evaluate.py        #   逐類別 Dice/IoU 評估
│   └── visualize.py       #   訓練曲線繪圖
│
├── configs/               # 每個模式一份 YAML 設定檔
├── tools/                 # 一次性資料工具
│   ├── preprocess/         #   原始資料 -> 處理後資料
│   ├── convert_heic_to_jpg.py
│   ├── rename_images.py
│   └── make_finetune_split.py
│
├── data/
│   ├── raw/                # 原始資料
│   │   ├── segmentation/   #   （原 data_raw）
│   │   └── multiclass/     #   （原 data_raw_cla）
│   └── processed/          # 處理後、可直接訓練的資料
│       ├── wound/          #   多類別傷口（原 data）
│       └── tkr/            #   TKR（原 data_tkr）
│
├── outputs/                # 所有產出物
│   ├── checkpoints/<version>/<run_name>/{last,best}.pt
│   ├── logs/<version>/<run_name>.csv
│   ├── runs/<version>/<run_name>/{config.yaml,metrics_*.json,curves.png}
│   ├── predictions/  visualizations/
│   └── legacy/             # 重構前的舊產出（results / results_cla / ...）
│
└── _deprecated/            # 重構前的舊程式碼備份（確認無誤後可刪除）
```

## 安裝

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .            # 讓 `import woundseg` 在任何地方都能用（選用）
```

## 使用方式

每個腳本都吃一份 YAML 設定檔，任何欄位都能用命令列覆蓋。

```bash
# 1) 一般預訓練
python scripts/train.py --config configs/train.yaml

# 2) 多類別傷口微調（每類 Dice ≥ 0.80 的目標）
python scripts/finetune_wound.py --config configs/finetune_wound.yaml

# 3) TKR 微調
python scripts/finetune_tkr.py --config configs/finetune_tkr.yaml

# 命令列覆蓋設定（方便做實驗）
python scripts/train.py --config configs/train.yaml --epochs 120 --lr 5e-4 --run_name run2

# 評估：印出每一類的 Dice，並標示未達 0.80 的類別
python scripts/evaluate.py --config configs/evaluate.yaml --split val

# 批次推論
python scripts/predict.py --config configs/predict.yaml

# 繪製訓練曲線
python scripts/visualize.py --version v3 --run_name run1
```

也可以用 `run.sh` 當作捷徑：`./run.sh train`、`./run.sh evaluate`、`./run.sh report v3 run1`。

## 設計重點

- **設定分層**：模式預設值 → YAML → 命令列，三層由上往下覆蓋，做大量實驗時不必改程式碼。
- **train 與 finetune 完全分離**：三個入口腳本各自獨立，不再互相 `import`；共用邏輯都在 `woundseg.engine.run_training`。
- **checkpoint 自帶架構資訊**：每個 checkpoint 內含 `model_cfg`，推論時自動重建正確架構，不需再用 run 名稱猜測。
- **逐類別指標**：驗證會回傳每個資料集（類別）的 Dice，`mean_dice` 為各類別的巨觀平均——這正是「每類 ≥ 0.80」要追蹤的指標。微調以 `mean_dice` 選最佳權重。

## 實驗建議（朝每類 Dice ≥ 0.80）

1. 先用 `evaluate.py` 找出目前低於 0.80 的類別。
2. 樣本數少的類別容易偏低：用 `tools/make_finetune_split.py` 平衡各類別數量，或加強資料增強。
3. 針對漏抓（recall 低）的類別，把 `loss_name` 換成 `focal_tversky` 並調高 Tversky 的 beta（懲罰 false negative）。
4. 每次實驗用不同的 `run_name`，產出物會分開存放，方便用 `visualize.py` 比較。
