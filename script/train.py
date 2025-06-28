
# https://docs.ultralytics.com/modes/train/#train-settings
from ultralytics import YOLO
from os.path import exists as path_exist

def train_model(model_path, dataset):
    if not path_exist(model_path): print("下載新的模型做訓練")
    model = YOLO(model_path)
    
    dataset_path = f"./data/{dataset}"
    results = model.train(data = f"{dataset_path}/data.yaml", 
                          epochs = 50,
                          batch = 4,
                          imgsz = 640, # 圖片大小
                          lr0 = 0.01, # 初始學習率
                          device = 'cuda',
                          optimizer = 'Adam', # 較常用的優化器
                          val = False, # 不做驗證
                          amp=False, # 必須禁用，否則會自動下載 yolov11 來跑
                          plots = True # 繪出訓練統計的資料         
                        )  

if __name__ == "__main__":
    train_model("./yolov8m-pose.pt", 
                "BlackSoldierFly_Lableling_COPY.v3i.yolov8")

