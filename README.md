# Yolo-pose

這個專案嘗試使用 yolo-pose 模型處理問題

## Problem 問題

假設我要量測物體的長度，最後做統計分析，常用的流程是: `拍照(有比例尺) -> 電腦標記 -> 計算長度`

問題點在於 `電腦標記` 標得非常累，所以希望有個電腦視覺的輔助工具

測量目標: 黑水虻幼蟲

## Method 方法

1. yolo-pose 標記點位
2. 透過點位計算長度(像素單位)
3. 手動標記尺的長度(公分單位)
4. 轉換長度資訊並統計

## Project introduce 專案介紹

1. data : 放著訓練資料
2. roboflow_download : 放著從 roboflow 下載的訓練資料壓縮檔
3. script
   * train.py 訓練
   * predict.py 預測
   * analyzer.py 分析損失
4. utils
   * dataset_fix_tool.py 用於 dataset_fixer.py
5. yolov8m-pose 執行 train.py 後會產生 runs 資料夾，裡面會有訓練的紀錄與結果，可以把這些結果搬來 yolov8m-pose 資料夾
   * train12 某一次的訓練結果，這個是目前覺得訓練的最好的，所以只保留這個
6. dataset_fixer.py 用來調整從 roboflow 下載的 yaml 檔案(因為在 roboflow 有多餘的操作，但是 data 資料夾下的檔案已經處理過了，可以不必再執行)

## Future 展望

1. 尺也可以用 yolo-pose 標記，這樣甚至不需要手東標記尺的位置
2. 執行 analyzer.py 可以看到還有下降的趨勢，也許可以跑更多圈
3. 訓練資料很少且單一

