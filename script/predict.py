from ultralytics import YOLO
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

model_type = "yolov8m-pose"
weight_type = "best"
train = 12
model_weights = f"./{model_type}/train{train}/weights/{weight_type}.pt"

dataset_type = "test"
image_path = f"./data/BlackSoldierFly_Lableling_COPY.v3i.yolov8/{dataset_type}/images/"
label_path = f"./data/BlackSoldierFly_Lableling_COPY.v3i.yolov8/{dataset_type}/labels/"

model = YOLO(model_weights)

# 把點串成線長度
def get_length(keypoint_xy):
    length = 0
    cur = keypoint_xy[0]

    for nxt in keypoint_xy[1:]:
        length += np.linalg.norm(nxt - cur)
        cur = nxt
    return int(length)

# 把標籤數據轉換成點
def label_keypoints_transfer(label_file):
    keypoints = []
    with open(label_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip().split()
        keypoint_set = []
        visible = []
        for i in range(4):
            visible.append(int(line[7 + i * 3]))
            keypoint_set.append([float(line[5 + i * 3]), float(line[6 + i * 3])])
        
        if "0" in visible: continue
        keypoints.append(keypoint_set)
    
    return np.array(keypoints, dtype=np.float32) * 640

# 要求使用者輸入標記 尺的 長度
def measure_distance_on_image(image): # from gpt
    img = image.copy()
    original = img.copy()
    total_dist = None
    cm = None
    points = []

    def draw_all(): # 畫線
        img[:] = original.copy()
        for pt in points:
            cv2.circle(img, pt, 4, (0, 0, 255), -1)

        for i in range(1, len(points)):
            cv2.line(img, points[i - 1], points[i], (0, 255, 0), 2)

    def mouse_callback(event, x, y, flags, param): # 滑鼠點擊事件
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            draw_all()

    print("[滑鼠左鍵] 加點, [u] 回上一步, [c] 計算長度, [ESC] 結束")

    while True:
        if cv2.getWindowProperty("Measure", cv2.WND_PROP_VISIBLE) < 1:
            cv2.namedWindow("Measure")
            cv2.setMouseCallback("Measure", mouse_callback)
            print("退出不要按 (X) 按 ESC")

        cv2.imshow("Measure", img)
        key = cv2.waitKey(1)

        if key == 27: 
            break
        elif key == ord('u') and points: # 返回上一步
            points.pop()
            draw_all()
        elif key == ord('c') and len(points) >= 2: # 計算長度, 要求輸入 cm
            total_dist = get_length(np.array(points))

            print(f"總像素距離 = {total_dist:.2f} px")
            
            try:
                cm = float(input("輸入這兩點的實際長度（公分）："))
                print(f"像素距離 = {total_dist:.2f} px, 每像素 = {cm / total_dist:.4f} 公分")
                print("按下 ESC 完成設定")
            except:
                print("輸入無效，請輸入數字")


    cv2.destroyWindow("Measure")
    
    if total_dist is not None and cm is not None:
        return total_dist, cm
    
    return None, None


# 開始執行與量測
for (image_file, label_file) in zip(sorted(os.listdir(image_path)), 
                                    sorted(os.listdir(label_path))):
    
    # 路徑設定與資料前處理
    label_full_path = os.path.join(label_path, label_file)
    label_keypoints = label_keypoints_transfer(label_full_path)

    image_full_path = os.path.join(image_path, image_file)
    image = cv2.imread(image_full_path)
    
    # 執行模型
    results = model(image)
    boxes = results[0].boxes
    keypoints = results[0].keypoints
    predict_lengths = []

    # 畫框與骨架
    if boxes is not None and keypoints is not None:
        for i, (box, kps, conf) in enumerate(zip(boxes.xyxy, keypoints.xy, boxes.conf), 1):
            if conf < 0.6: break 
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 繪製骨架點
            for kp in kps:
                x, y = map(int, kp[:2])
                cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
            
            predict_lengths.append(get_length(kps.cpu().numpy()))
            
    # 輸入標記 尺 的長度
    while True:
        total_dist, cm = measure_distance_on_image(image)
        if total_dist is not None and cm is not None:
            break
        else:
            if input("沒有標記尺長度，是否不做標記? [Y] ") == 'Y':
                break
    
    if total_dist is None or cm is None:
        print("跳過分布圖")
        continue
    
    
    # 像素轉換長度
    cm_pre_pixel = cm / total_dist
    label_lengths = np.array([get_length(kps) for kps in label_keypoints]) * cm_pre_pixel
    predict_lengths = np.array(predict_lengths) * cm_pre_pixel

    # 畫分布圖
    plt.figure(figsize=(15, 6))
    plt.hist(predict_lengths, bins=20, edgecolor='black')
    plt.xlabel("Length")
    plt.ylabel("Frequency")
    plt.title("Length Distribution")


    plt.text(1.01, 0.99, f"Predict", transform=plt.gca().transAxes, ha='left', va='top')
    plt.text(1.01, 0.95, f"Total {len(predict_lengths)}", transform=plt.gca().transAxes, ha='left', va='top')
    plt.text(1.01, 0.90, f"Unit: CM", transform=plt.gca().transAxes, ha='left', va='top')
    plt.text(1.01, 0.85, f"Length Avg: {predict_lengths.mean():.2f}", transform=plt.gca().transAxes, ha='left', va='top')
    plt.text(1.01, 0.80, f"Length Std: {predict_lengths.std():.2f}", transform=plt.gca().transAxes, ha='left', va='top')
    plt.text(1.01, 0.75, f"Length Max: {predict_lengths.max():.2f}", transform=plt.gca().transAxes, ha='left', va='top')
    plt.text(1.01, 0.70, f"Length Min: {predict_lengths.min():.2f}", transform=plt.gca().transAxes, ha='left', va='top')

    plt.text(1.01, 0.55, f"Label", transform=plt.gca().transAxes, ha='left', va='top')
    plt.text(1.01, 0.50, f"Total {len(label_keypoints)}", transform=plt.gca().transAxes, ha='left', va='top')
    plt.text(1.01, 0.45, f"Unit: CM", transform=plt.gca().transAxes, ha='left', va='top')
    plt.text(1.01, 0.40, f"Length Avg: {label_lengths.mean():.2f}", transform=plt.gca().transAxes, ha='left', va='top')
    plt.text(1.01, 0.35, f"Length Std: {label_lengths.std():.2f}", transform=plt.gca().transAxes, ha='left', va='top')
    plt.text(1.01, 0.30, f"Length Max: {label_lengths.max():.2f}", transform=plt.gca().transAxes, ha='left', va='top')
    plt.text(1.01, 0.25, f"Length Min: {label_lengths.min():.2f}", transform=plt.gca().transAxes, ha='left', va='top')

    plt.show()
        

cv2.destroyAllWindows()
