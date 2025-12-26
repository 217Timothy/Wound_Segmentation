import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.wound_dataset import WoundDataset
from models.unet.unet import UNet
from utils.checkpoint import save_checkpoint, load_checkpoint


def main():
    # -----------------------------
    # 基本設定
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")
    
    # -----------------------------
    # path
    # -----------------------------
    img_dir = "data/processed/images"
    mask_dir = "data/processed/masks"
    
    run_name = "unet"
    ckpt_dir = f"outputs/checkpoints/{run_name}"
    log_dir = f"outputs/logs/{run_name}"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    last_ckpt_path = os.path.join(ckpt_dir, "last.pt")
    best_ckpt_path = os.path.join(ckpt_dir, "best.pt")
    log_path = os.path.join(log_dir, "train_log.csv")
    
    # -----------------------------
    # Hyperparams
    # -----------------------------
    batch_size = 2
    num_epochs = 2
    lr = 1e-4
    num_workers = 4
    
    # -----------------------------
    # Dataset / DataLoader
    # -----------------------------
    dataset = WoundDataset(
        images_dir=img_dir,
        masks_dir=mask_dir,
        transform=None
    )
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"Dataset Size: {len(dataset)}")
    
    # -----------------------------
    # Model / Loss / Optimizer
    # -----------------------------
    model = UNet(in_channels=3, out_channels=1)
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # -----------------------------
    # Resume (optional)
    # -----------------------------
    start_epoch = 0
    best_loss = float("inf")

    if os.path.exists(last_ckpt_path):
        print(f"[INFO] Resuming from {last_ckpt_path}")
        start_epoch, best_loss_loaded = load_checkpoint(
            last_ckpt_path, model, optimizer, map_location=device
        )
        if best_loss_loaded is not None:
            best_loss = best_loss_loaded
        print(f"[INFO] start_epoch={start_epoch}, best_loss={best_loss}")
    
    # -----------------------------
    # Training
    # -----------------------------
    model.train()
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0

        for step, (images, masks) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)  # (B,3,H,W)
            masks = masks.to(device, non_blocking=True)    # (B,1,H,W) 0/1

            outputs = model(images)                        # logits (B,1,H,W)
            loss = criterion(outputs, masks)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if step % 20 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] "
                    f"Step [{step}/{len(dataloader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}] Average Loss: {avg_loss:.6f}")

        # log
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, f"{avg_loss:.6f}"])

        # save last
        save_checkpoint(
            last_ckpt_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            best_metric=best_loss
        )

        # save best (by train loss for now; 之後我們會換成 val dice)
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                best_ckpt_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                best_metric=best_loss
            )
            print(f"✅ Best updated: {best_loss:.6f}")

    print("Training finished.")
    print("Last checkpoint:", last_ckpt_path)
    print("Best checkpoint:", best_ckpt_path)
    print("Log:", log_path)


if __name__ == "__main__":
    main()