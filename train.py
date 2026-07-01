from ultralytics import YOLO
from swanlab.integration.ultralytics import add_swanlab_callback
import swanlab

# 修复中文/特殊符号编码问题

def main():
    # 本地模式（数据保存在本地）
    swanlab.login(api_key="YbgtGCmAlbx8lmnAsYScX")
    swanlab.init(
        project="ultralytics-main",
        experiment_name="YOLOv8n",
    )

    # 加载模型
    model = YOLO("yolov8n.pt")
    add_swanlab_callback(model)

    # 训练模型（使用绝对路径更可靠！）
    model.train(
       data = r"D:\\python project\\ultralytics-main\\hand_gesture_dataset\\hand.yaml",
        epochs=5,
        imgsz=512,
        batch=32
    )

if __name__ == "__main__":
    main()