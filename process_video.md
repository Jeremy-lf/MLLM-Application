## Read Video

```
import os
from decord import VideoReader
import cv2  # 用于保存图片（也可以用PIL等其他库）

def save_frames_at_fps(video_file, target_fps, output_folder):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 读取视频信息
    vr = VideoReader(video_file, num_threads=4)
    total_frames = len(vr)
    avg_fps = vr.get_avg_fps()
    
    # 计算帧间隔（向上取整确保不丢帧）
    frame_interval = max(1, round(avg_fps / target_fps))
    
    # 按间隔保存帧
    saved_count = 0
    for i in range(0, total_frames, frame_interval):
        frame = vr[i].asnumpy()  # 获取帧数据（Decord返回的是numpy数组）
        
        # 如果是RGB格式，转换为BGR（OpenCV默认格式）
        if frame.shape[2] == 3:  # 3通道（RGB）
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # 保存图片
        output_path = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
        cv2.imwrite(output_path, frame)
        saved_count += 1
    
    print(f"Saved {saved_count} frames at {target_fps} FPS to {output_folder}")

# 使用示例
video_file = "input.mp4"
target_fps = 2  # 目标保存FPS（例如每秒保存2帧）
output_folder = "output_frames"
save_frames_at_fps(video_file, target_fps, output_folder)
```
### 帧间隔计算：
frame_interval = round(avg_fps / target_fps), 例如：视频原始FPS=30，目标FPS=2 → 每隔15帧保存1帧。

### 格式转换：
Decord返回的帧是RGB格式，而OpenCV的imwrite需要BGR格式，因此需要转换。

### 边界情况处理：
使用max(1, ...)确保间隔至少为1（避免除零或跳过所有帧）。如果视频时长不能被整除，最后一帧可能略晚于严格的时间间隔。

### 性能优化：
Decord的VideoReader支持多线程（通过num_threads参数）。直接通过索引访问帧（vr[i]）比逐帧读取更高效。
```
import cv2

def save_frames_opencv(video_file, target_fps, output_folder):
    cap = cv2.VideoCapture(video_file)
    avg_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, round(avg_fps / target_fps))
    
    saved_count = 0
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % frame_interval == 0:
            cv2.imwrite(f"{output_folder}/frame_{saved_count:04d}.jpg", frame)
            saved_count += 1
        frame_id += 1
    
    cap.release()
    print(f"Saved {saved_count} frames at {target_fps} FPS")
```
