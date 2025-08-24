## Processor

self.video_processor调用的类是Qwen2VLVideoProcessor(BaseVideoProcessor), `BaseVideoProcessor` 位置在transformers/video_processing_utils.py 
```
class BaseVideoProcessor(BaseImageProcessorFast):
  def __init__(self,..):
      pass
  def __call__(self, videos):
      return self.preprocess(videos, **kwargs)

  def preprocess(self, xxx):
      self.sample_frames(xxx) # 用于从视频张量中均匀采样帧
      sefl._preprocess(xxx)  # 用于对视频数据进行预处理，包括帧采样、设备转移、尺寸调整、颜色空间转换、裁剪、缩放、归一化等操作。
      return data={"pixel_values_videos": pixel_values_videos, "video_grid_thw": video_grid_thw}
```

---

self.image_processor调用的是` Qwen2VLImageProcessor(BaseImageProcessor)`, 位置在transformers/models/qwen2_vl/image_processing_qwen2_vl.py

```
# 用于动态resize图像、预处理（归一化、裁剪、resize）
class Qwen2VLImageProcessor(BaseImageProcessor):
  def __init__(self,..):
      pass
  def _preprocess(self, xx):
      # 处理batch图像，包括Qwen2.5-VL的原生分辨率，返回reshape后的特征

      # 如果视频帧数不是 temporal_patch_size 的整数倍，通过复制最后一帧填充不足部分。
      if patches.shape[0] % temporal_patch_size != 0:
          repeats = np.repeat(
              patches[-1][np.newaxis], temporal_patch_size - (patches.shape[0] % temporal_patch_size), axis=0
          )
          patches = np.concatenate([patches, repeats], axis=0)
      channel = patches.shape[1]
      grid_t = patches.shape[0] // temporal_patch_size 
      grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
      patches = patches.reshape(
            grid_t,
            temporal_patch_size,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
      patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
      flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size
        ) # 特征总数 * channel

        return flatten_patches, (grid_t, grid_h, grid_w)

    def preprocess(self, xxx):
        self._preprocess(xxx)
        return data.update({"pixel_values": pixel_values, "image_grid_thw": vision_grid_thws})

```
