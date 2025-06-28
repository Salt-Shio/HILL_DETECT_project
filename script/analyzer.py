import yaml
import os
import pandas as pd
import matplotlib.pyplot as plt

model = "yolov8m-pose"
model_paths = [f"./{model}/{train_path}" 
               for train_path in os.listdir(f"./{model}")]
data_type = ["result.png", "args.yaml", "val_batch0_pred.jpg", "result.csv"]

print(f"[model: {model}]")
for train_path in sorted(os.listdir(f"./{model}"), key = lambda x: int(x.replace("train", ""))):
    print(f"    model: {model}, train_path: {train_path}")
    args_path = f"{model}/{train_path}/args.yaml"
    result_csv_path = f"{model}/{train_path}/results.csv"
    result_img_path = f"{model}/{train_path}/results.png"
    one_pred_path = f"{model}/{train_path}/val_batch0_pred.jpg"
    
    with open(args_path, 'r') as stream: 
        train_args = yaml.load(stream, Loader = yaml.FullLoader)
        epochs = train_args["epochs"]
        batch = train_args["batch"]
        dataset = train_args["data"]
        train_name = train_args["name"]
        lr0 = train_args["lr0"]

    csv = pd.read_csv(result_csv_path)
    
    print("=" * 40)
    print(f"    lr0: {lr0}")
    print(f"    epochs: {epochs}")
    print(f"    batch: {batch}")
    print(f"    dataset: {dataset.split('/')[2]}\n")
    print(csv.tail(5).loc[:, ["train/box_loss", "train/pose_loss"]].mean()) # 最後面 5 比的 loss 平均
    print("=" * 40)

    # 繪製圖表分兩張圖，box_loss 和 pose_loss
    plt.figure(figsize=(10, 5))
    plt.suptitle(f"Model {model}: {train_name}")

    plt.subplot(1, 2, 1)
    plt.plot(csv["epoch"], csv["train/box_loss"], label="Train Box Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Box Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(csv["epoch"], csv["train/pose_loss"], label="Train Pose Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Pose Loss")
    plt.legend()

    plt.tight_layout() # 自動調整
    plt.savefig(f"{model}/{train_path}/loss.png")
    plt.show()