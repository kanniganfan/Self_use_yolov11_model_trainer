import os
import requests
from tqdm import tqdm

def download_file(url, filename):
    """下载文件并显示进度条"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def main():
    """下载 YOLOv8 模型文件"""
    models = {
        "yolo11n.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "yolo11s.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt",
        "yolo11m.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt",
        "yolo11l.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt",
        "yolo11x.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt"
    }
    
    os.makedirs("weights", exist_ok=True)
    
    print("开始下载 YOLO 模型文件...")
    for model_name, url in models.items():
        filename = os.path.join("weights", model_name)
        if not os.path.exists(filename):
            print(f"\n下载 {model_name}...")
            download_file(url, filename)
        else:
            print(f"\n{model_name} 已存在，跳过下载")
    
    print("\n所有模型下载完成！")

if __name__ == "__main__":
    main() 