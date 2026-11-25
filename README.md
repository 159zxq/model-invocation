# V100 数字人模型服务

这个仓库把数字人链路拆成四个独立服务，并针对单张 NVIDIA V100 32GB
做了显存和进程隔离：

```text
照片 + 身高/体重/年龄
        │
        ▼
视觉引擎 8001：RealVisXL V5 + IP-Adapter-FaceID + ControlNet OpenPose
        │ 全身图 / 视频参考图
        ▼
数字人格 8000：硅基流动 Qwen/Qwen3-8B
        │ 回复文本 / 情绪 / 动作 / 视频提示词
        ▼
语音引擎 8002：Fun-CosyVoice3-0.5B，零样本音色克隆
        │ 克隆音色 WAV
        ▼
视频引擎 8003：EchoMimicV3 Flash Pro
        │
        ▼
最终 MP4
```

语音链路不包含 ASR。用户上传的参考音频只用于音色克隆和语气控制，
不需要提供逐字稿。

## 选型与 V100 适配

| 模块 | 模型 | 部署方式 | 说明 |
| --- | --- | --- | --- |
| 视觉 | `SG161222/RealVisXL_V5.0` | 本地 GPU | SDXL 写实底模，默认 `832x1216` |
| 后备视觉 | `stabilityai/stable-diffusion-xl-base-1.0` | 本地 GPU | RealVis 文件缺失时自动回退 |
| 身份保持 | `h94/IP-Adapter-FaceID` | 本地 GPU | 使用 `ip-adapter-faceid_sdxl.bin`，直接输入 InsightFace 512 维身份 embedding |
| 姿态控制 | `xinsir/controlnet-openpose-sdxl-1.0` | 本地 GPU | 生成头到脚的站姿骨架 |
| 人脸检测 | InsightFace `buffalo_l` | CPU | 只提取身份 embedding，不占用显存 |
| 数字人格 | `Qwen/Qwen3-8B` | 硅基流动 API | 不占本地显存；关闭思考模式并强制 JSON |
| 语音 | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | 本地 GPU | 零样本音色克隆、情绪和语速控制 |
| 视频 | `BadToBest/EchoMimicV3` Flash Pro | 本地 GPU | 81 帧一段，默认最多 3 段 |
| 视频音频编码 | `TencentGameMate/chinese-wav2vec2-base` | CPU | EchoMimic Flash 必需组件 |

所有本地 GPU 推理固定使用 `float16`。V100 的计算能力是 `sm70`，不支持
BF16、FP8 和 FlashAttention 3，因此没有启用这些优化，也没有使用 vLLM 或
TensorRT。三个 GPU 服务通过 `runtime/gpu.lock` 串行执行，完整编排任务还会
持有 `runtime/job.lock`，避免多条任务交叉换模型。

这套组合的共同约束是“一个 32GB V100、一个任务、模型按需换入换出”：

- RealVisXL 负责写实全身图，IP-Adapter-FaceID 保持身份，OpenPose 约束站姿。
- CosyVoice3 只有 0.5B，适合在视觉模型卸载后再加载，避免常驻挤占显存。
- EchoMimicV3 Flash Pro 使用 Wan2.1-Fun 1.3B，作为独立子进程生成视频，
  进程退出后 CUDA 显存会直接归还。
- Qwen3-8B 通过硅基流动调用，不进入本机显存。

身高、体重和 BMI 会改变提示词与 OpenPose 的肩宽、躯干和腿部比例，但它只能
控制“大致体型”，不能从一张正面照片恢复精确的人体测量数据。

默认策略是：

- 视觉推理完成后卸载 SDXL，降低与视频模型同时驻留的风险。
- CosyVoice3 默认使用后卸载。确认显存余量后可以设 `TTS_KEEP_WARM=1`。
- EchoMimic 每次通过独立 Python 子进程运行，进程退出后显存立即释放。
- 视频默认 `81` 帧、`25 FPS`，也就是每段约 `3.24` 秒；最多生成 3 段并
  在服务内拼接，防止一个任务无限占用 GPU。

## 目录

```text
configs/                 环境和模型版本配置
scripts/                 CPU 安装、下载、GPU 启停和验证
services/common/         统一错误、媒体校验、任务目录、GPU 锁
services/vision/         全身图生成
services/tts/            音色克隆与语音合成
services/video/          EchoMimic 视频生成
services/orchestrator/   数字人格和端到端编排
runtime/                 模型、仓库、venv、日志、任务和锁
```

每个服务使用独立 venv：

```text
runtime/venvs/vision-service
runtime/venvs/tts-service
runtime/venvs/video-service
runtime/venvs/orchestrator
runtime/venvs/download-tools
```

Python 固定为 `3.10.12`。uv 管理的解释器、venv、模型和仓库都放在
`runtime/` 下，因此 Cloud Studio 的 CPU 实例和 GPU 实例只要挂载同一个
工作区，就不需要重新安装。

## 第一阶段：CPU 实例准备

先把仓库放到 Cloud Studio 的持久化工作区，例如：

```bash
cd /workspace
git clone <your-git-url> model-invocation
cd model-invocation
cp .env.example .env
```

编辑 `.env`，至少设置：

```dotenv
PROJECT_ROOT=/workspace/model-invocation
SILICONFLOW_API_KEY=sk-...
MODELS_ROOT=/workspace/model-invocation/runtime/models
REPOS_ROOT=/workspace/model-invocation/runtime/repos
VENVS_ROOT=/workspace/model-invocation/runtime/venvs
```

如果需要下载 gated 模型，在 `.env` 中增加 `HF_TOKEN`。在中国大陆网络环境
可以尝试在 `.env` 中设置：

```dotenv
HF_ENDPOINT=https://hf-mirror.com
```

然后执行：

```bash
bash scripts/setup_all.sh
```

该脚本会：

1. 安装 Git、FFmpeg、SoX、OpenCV 运行库等系统依赖。
2. 把 Python 3.10.12、uv 和全部 venv 安装到 `runtime/`。
3. 先安装 CUDA 12.1 的 `torch==2.5.1`、`torchvision==0.20.1` 和
   `torchaudio==2.5.1`，再安装服务依赖，并重新校验 Torch/CUDA 版本。
4. 下载 RealVisXL、IP-Adapter-FaceID、ControlNet、CosyVoice3、
   EchoMimicV3、Wan2.1 和 wav2vec2 权重。
5. 下载 InsightFace `buffalo_l`。

模型和环境下载量很大，建议至少预留 80GB 磁盘。可分开执行：

```bash
bash scripts/setup_python.sh
bash scripts/setup_vision.sh
bash scripts/setup_tts.sh
bash scripts/setup_video.sh
bash scripts/setup_orchestrator.sh
bash scripts/download_models.sh
```

如果 Cloud Studio 的 CPU 和 GPU 实例不共享磁盘，需要在 GPU 实例重新执行
`setup_all.sh`，或者自行同步整个 `runtime/` 目录。

## 第二阶段：V100 GPU 实例启动

确认 GPU 实例能看到同一工作区和 `.env`，然后执行：

```bash
cd /workspace/model-invocation
bash scripts/start_all.sh
bash scripts/status.sh
```

服务地址：

| 服务 | 地址 |
| --- | --- |
| 编排 API 和 Swagger | `http://127.0.0.1:8000/docs` |
| 视觉 | `http://127.0.0.1:8001/docs` |
| 语音 | `http://127.0.0.1:8002/docs` |
| 视频 | `http://127.0.0.1:8003/docs` |

日志和 PID：

```text
runtime/logs/*.log
runtime/pids/*.pid
```

常用命令：

```bash
bash scripts/status.sh
bash scripts/verify.sh
bash scripts/stop_all.sh
tail -f runtime/logs/video.log
```

模型采用首次请求时懒加载，所以 `/health` 可访问不代表模型已经加载完成。
第一次请求会明显慢于后续请求。

## 本地测试

测试环境只安装 API 和媒体处理依赖，不下载模型，也不安装 CUDA Torch：

```bash
bash scripts/setup_dev.sh
runtime/venvs/dev-tests/bin/python -m pytest -q
runtime/venvs/dev-tests/bin/python -m ruff check services scripts tests
```

测试覆盖四个服务的健康检查、上传和参数边界、视频命令构造、端到端阶段顺序、
失败结果落盘以及 Qwen JSON 输出归一化。安装顺序和模型 revision 也有静态
回归测试。

## 端到端调用

```bash
curl -X POST http://127.0.0.1:8000/v1/jobs \
  -F "photo=@/path/to/front-photo.jpg" \
  -F "reference_audio=@/path/to/voice.wav" \
  -F "height_cm=172" \
  -F "weight_kg=63" \
  -F "age=28" \
  -F "gender=female" \
  -F "style=realistic" \
  -F "user_text=请用亲切自然的语气向第一次见面的用户打招呼" \
  -F "persona=友好、耐心、说话简洁的数字人助手" \
  -F "scene=产品介绍视频" \
  -F "seed=2026"
```

完成后返回：

```json
{
  "job_id": "avatar-...",
  "status": "completed",
  "personality": {
    "source": "siliconflow",
    "plan": {
      "text": "你好，很高兴认识你。",
      "emotion": "happy",
      "gesture": "greet",
      "speed": 1.0,
      "video_prompt": "A person is speaking naturally and greeting the viewer warmly."
    }
  },
  "artifacts": {
    "avatar_image": "/workspace/.../output/fullbody.png",
    "video_reference": "/workspace/.../output/video_reference.png",
    "pose": "/workspace/.../output/pose.png",
    "speech": "/workspace/.../output/speech.wav",
    "video": "/workspace/.../output/avatar.mp4"
  }
}
```

完整任务状态和失败原因也会原子写入：

```text
runtime/jobs/<job_id>/result.json
```

## 单独调用

### 全身图

```bash
curl -X POST http://127.0.0.1:8001/v1/avatar/image \
  -F "photo=@front.jpg" \
  -F "height_cm=175" \
  -F "weight_kg=68" \
  -F "age=30" \
  -F "gender=male" \
  -F "style=business"
```

### 音色克隆与语音

```bash
curl -X POST http://127.0.0.1:8002/v1/avatar/speech \
  -F "reference_audio=@voice.wav" \
  -F "text=你好，很高兴认识你。" \
  -F "emotion=happy" \
  -F "speed=1.0"
```

参考音频建议 3 到 15 秒、单人说话、无明显底噪。服务会统一转换为
24kHz 单声道 WAV。

### 视频

```bash
curl -X POST http://127.0.0.1:8003/v1/avatar/video \
  -F "image=@video_reference.png" \
  -F "audio=@speech.wav" \
  -F "prompt=A person is speaking naturally, full body visible head to toe." \
  -F "seed=43"
```

EchoMimic 主要是音频驱动的口型和头部动作模型，文本 prompt 用来约束画面，
不代表可以精确控制任意手势。

## 数字人格配置

编排服务调用硅基流动 OpenAI-compatible Chat Completions：

```text
POST https://api.siliconflow.cn/v1/chat/completions
model: Qwen/Qwen3-8B
```

`Qwen3-8B` 在上下文中只负责生成回复和视频规划，不进入本地 GPU。默认关闭
thinking，并要求只返回以下 JSON：

```json
{
  "text": "你好，很高兴认识你。",
  "emotion": "happy",
  "gesture": "greet",
  "speed": 1.0,
  "video_prompt": "A person is speaking naturally and greeting the viewer warmly."
}
```

如果 `SILICONFLOW_API_KEY` 未配置或 API 暂时失败，
`PERSONALITY_ALLOW_FALLBACK=1` 会使用用户输入生成安全的后备计划；生产环境
也可以设为 `0`，让任务直接失败并暴露上游问题。

## 显存和并发

推荐初始参数：

```dotenv
VISION_WIDTH=832
VISION_HEIGHT=1216
VISION_STEPS=28
VISION_CPU_OFFLOAD=0
VISION_ATTENTION_SLICING=1
VISION_KEEP_WARM=0

TTS_FP16=1
TTS_KEEP_WARM=0

VIDEO_MAX_FRAMES=81
VIDEO_MAX_SEGMENTS=3
VIDEO_STEPS=8
VIDEO_SAMPLE_SIZE=768
```

如果视觉首次请求发生 OOM，依次尝试：

1. 设 `VISION_ATTENTION_SLICING=1`。
2. 把 `VISION_WIDTH=768`、`VISION_HEIGHT=1152`。
3. 设 `VISION_CPU_OFFLOAD=1`，速度会变慢。
4. 把 `VISION_STEPS` 降到 `22` 到 `24`。

不要同时调高 `TTS_KEEP_WARM`、视觉常驻和视频段数。V100 的 32GB 显存需要
给 CUDA context、VAE 临时张量和视频编码留余量。

## 安全和合规

- 只处理已取得本人授权的人脸和声音素材。
- `InsightFace` 的预训练模型和 `IP-Adapter-FaceID` 含有仅适合研究/非商业使用的
  授权限制。正式商用前请核对模型卡、InsightFace 预训练模型许可和所在地区的
  肖像权要求，并替换为已获授权的身份编码方案。
- 不要把生成内容用于冒充真人、诈骗、虚假身份或未经同意的商业代言。
- 建议在最终产品中明确提示内容是 AI 生成。
- 当前 API 没有鉴权，Cloud Studio 对外暴露时应放在反向代理、鉴权网关或
  内网之后。
- `.env`、模型权重、venv、任务产物和日志不会进入 Git。

## 已知限制

- 当前开发机没有 V100，已验证代码结构和接口契约，但不能代替真机 CUDA、
  显存峰值、CosyVoice3 和 EchoMimicV3 的实机推理验证。
- `configs/models.yaml` 中没有填写视频显存峰值。必须通过首次请求时的
  `nvidia-smi --query-gpu=memory.used --format=csv -l 1` 实测后再调整段数和分辨率。
- 全身身份一致性依赖正面、清晰、单人照片。侧脸、遮挡、低分辨率或多人照片
  会明显降低效果。
- SDXL 生成全身图时手脚仍是主要失败点。正式产品需要增加自动重试、人体检测
  和质量评分。
- 长视频通过分段生成再拼接，段与段之间可能有轻微表情或姿态跳变。
- 上游仓库和模型更新后 API 可能变化。本仓库已在 `configs/download.env`
  固定 Git commit 和 Hugging Face snapshot revision；升级时必须同步重新测试。
