## HuggingFace Transformer

#### 1.`config.json`

config一般包含了`"architectures"`, `model_type`类型，从而能够加载对应的类

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(


#### `2.preprocessor_config.json`

AutoProcessor会检查模型仓库中的配置文件（如preprocessor_config.json）。
```
processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
        )
print(processor)
```

```
# process包括内容如下

Qwen2_5_VLProcessor:
- image_processor: Qwen2VLImageProcessor {
  "do_convert_rgb": true,
  "do_normalize": true,
  "do_rescale": true,
  "do_resize": true,
  "image_mean": [
    0.48145466,
    0.4578275,
    0.40821073
  ],
  "image_processor_type": "Qwen2VLImageProcessor",
  "image_std": [
    0.26862954,
    0.26130258,
    0.27577711
  ],
  "max_pixels": 12845056,
  "merge_size": 2,
  "min_pixels": 3136,
  "patch_size": 14,
  "processor_class": "Qwen2_5_VLProcessor",
  "resample": 3,
  "rescale_factor": 0.00392156862745098,
  "size": {
    "longest_edge": 12845056,
    "shortest_edge": 3136
  },
  "temporal_patch_size": 2
}

- tokenizer: Qwen2TokenizerFast(name_or_path='Qwen2.5-VL-7B-Instruct', vocab_size=151643, model_max_length=131072, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'eos_token': '<|im_end|>', 'pad_token': '<|endoftext|>', 'additional_special_tokens': ['<|im_start|>', '<|im_end|>', '<|object_ref_start|>', '<|object_ref_end|>', '<|box_start|>', '<|box_end|>', '<|quad_start|>', '<|quad_end|>', '<|vision_start|>', '<|vision_end|>', '<|vision_pad|>', '<|image_pad|>', '<|video_pad|>']}, clean_up_tokenization_spaces=False, added_tokens_decoder={
        151643: AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        151644: AddedToken("<|im_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        151645: AddedToken("<|im_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        151646: AddedToken("<|object_ref_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        151647: AddedToken("<|object_ref_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        151648: AddedToken("<|box_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        151649: AddedToken("<|box_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        151650: AddedToken("<|quad_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        151651: AddedToken("<|quad_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        151652: AddedToken("<|vision_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        151653: AddedToken("<|vision_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        151654: AddedToken("<|vision_pad|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        151655: AddedToken("<|image_pad|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        151656: AddedToken("<|video_pad|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        151657: AddedToken("<tool_call>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
        151658: AddedToken("</tool_call>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
        151659: AddedToken("<|fim_prefix|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
        151660: AddedToken("<|fim_middle|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
        151661: AddedToken("<|fim_suffix|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
        151662: AddedToken("<|fim_pad|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
        151663: AddedToken("<|repo_name|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
        151664: AddedToken("<|file_sep|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
}
)
- video_processor: Qwen2VLVideoProcessor {
  "crop_size": null,
  "data_format": "channels_first",
  "default_to_square": true,
  "device": null,
  "do_center_crop": null,
  "do_convert_rgb": true,
  "do_normalize": true,
  "do_pad": null,
  "do_rescale": true,
  "do_resize": true,
  "do_sample_frames": false,
  "fps": null,
  "image_mean": [
    0.48145466,
    0.4578275,
    0.40821073
  ],
  "image_std": [
    0.26862954,
    0.26130258,
    0.27577711
  ],
  "input_data_format": null,
  "max_frames": 768,
  "max_pixels": 12845056,
  "merge_size": 2,
  "min_frames": 4,
  "min_pixels": 3136,
  "num_frames": null,
  "patch_size": 14,
  "processor_class": "Qwen2_5_VLProcessor",
  "resample": 3,
  "rescale_factor": 0.00392156862745098,
  "size": {
    "longest_edge": 12845056,
    "shortest_edge": 3136
  },
  "size_divisor": null,
  "temporal_patch_size": 2,
  "video_metadata": null,
  "video_processor_type": "Qwen2VLVideoProcessor"
}


{
  "processor_class": "Qwen2_5_VLProcessor"
}
```

然后，可以根据processor类型处理多种模态的数据
```
# 准备输入（文本+图像）
url = "/root/paddlejob/workspace/env_run/output/lvfeng/InternVL/1749439861564568.jpg"
image = Image.open(url)
text = "描述这张图片的内容。"
 
# 处理器自动处理文本和图像
inputs = processor(text=text, images=image, return_tensors="pt")
print(inputs)

# 只处理图像
processor = AutoProcessor.from_pretrained("Qwen2.5-VL-7B-Instruct").image_processor
# inputs = processor(images=image, return_tensors="pt")
# print(inputs)
```

```
# 输出结果
{'input_ids': tensor([[ 53481, 108893,  45930, 104597,   1773]]), 'attention_mask': tensor([[1, 1, 1, 1, 1]]), 'pixel_values': tensor([[ 0.9960,  0.9522,  1.0690,  ...,  1.2074, -1.1247, -1.3522],
        [ 1.0836, -1.3397, -1.5733,  ..., -1.2811, -1.3096, -1.2811],
        [ 0.7771,  0.7771,  0.7771,  ..., -0.6839, -0.7408, -0.6981],
        ...,
        [-1.0185, -0.0696,  0.7041,  ...,  0.5675,  0.5248,  0.5248],
        [ 0.3975,  0.3245,  0.3391,  ...,  0.5817,  0.5959,  0.6670],
        [ 0.2953,  0.5873,  0.7917,  ..., -0.1720,  0.8377,  1.0367]]), 'image_grid_thw': tensor([[  1,  78, 138]])}
```

[Transformers：AutoProcessor代码分析](https://blog.csdn.net/fydw_715/article/details/146260757)


```
# transformers/utils/__init__.py
# 存放预定义的各种配置文件

WEIGHTS_NAME = "pytorch_model.bin"
WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"
TF2_WEIGHTS_NAME = "tf_model.h5"
TF2_WEIGHTS_INDEX_NAME = "tf_model.h5.index.json"
TF_WEIGHTS_NAME = "model.ckpt"
FLAX_WEIGHTS_NAME = "flax_model.msgpack"
FLAX_WEIGHTS_INDEX_NAME = "flax_model.msgpack.index.json"
SAFE_WEIGHTS_NAME = "model.safetensors"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
CONFIG_NAME = "config.json"
FEATURE_EXTRACTOR_NAME = "preprocessor_config.json"
IMAGE_PROCESSOR_NAME = "preprocessor_config.json"
VIDEO_PROCESSOR_NAME = "video_preprocessor_config.json"
AUDIO_TOKENIZER_NAME = "audio_tokenizer_config.json"
PROCESSOR_NAME = "processor_config.json"
GENERATION_CONFIG_NAME = "generation_config.json"
MODEL_CARD_NAME = "modelcard.json"
```

```
# transformers/models/auto/processing_auto.py
# 包含了类型与处理类的映射PROCESSOR_MAPPING_NAMES与processor_class_from_name函数
# 定义了AutoProcessor类以及from_pretrained函数


```
