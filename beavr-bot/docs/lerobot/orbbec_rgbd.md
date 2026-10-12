
## 无损 MP4 模式（当前 `record_fa_rgbd_mp4.sh`）

在大脑终端 1 启动 FA 遥操作后，终端 2 执行：

```bash
cd /mnt/public_data/lkw_data_files/beavr-bot
bash scripts/record_fa_rgbd_mp4.sh
```

当前脚本使用 `lossless_video=true, video=true, depth_video=false, keep_image_files=false`。
`depth_video` 是旧的有损预览开关，在此模式必须关闭。六路图像（3 RGB + 3 深度）
均为 MP4 视频字段；Parquet 保存状态、动作、时间等数据，不再嵌入 PNG。
录制期间仍使用临时 PNG 缓冲；所有编码完成且逐帧逐像素验证通过后删除。
出现编码或验证失败时保留缓冲图像供恢复，不以删除数据来满足“无 PNG”。

RGB 使用 `libx264rgb`、`rgb24`、CRF 0，不做 YUV420 色度抽样。
深度打包约定 v1：R=毫米 uint16 的低字节，G=高字节，B=0；使用同样无损编码。
恢复公式为 `depth_mm = uint16(R) | (uint16(G) << 8)`，0 仍表示无效。
当前相机在对齐后保留 1～1000 mm，超过 1000 mm 置零；视频存储保证这些输入值不变。
“无损”不撤销设备 MJPG 压缩、ISP、深度对齐或用户要求的范围过滤。
打包深度视频不是灰度预览，普通视频播放器显示的颜色没有距离含义。

编码器在发布每个 MP4 前用 PyAV 解码全部帧，与临时 PNG 比较；像素不一致或帧数
不一致均报错。无损编码及校验增加 CPU、磁盘和结束时等待时间，不保证 30 Hz 实际采集。
腕部 USB2 原生 10 FPS 的限制仍然存在。

`meta/info.json` 中的 `lossless_encoding` 指定自定义读取协议。
本仓库 v2.1 `LeRobotDataset` 已自动选择对应解码器，返回 RGB float32 [0,1] 和
深度 float32 毫米，分别为 C,H,W（深度 C=1），不对深度除以 255。
v3.0 视频可按 episode 元数据读取：

```python
from beavr.lerobot.common.datasets.lossless_video import read_v30_lossless_video
root = "datasets/fa_rgbd_mp4_实际时间"
rgb = read_v30_lossless_video(root, 0, "observation.images.head", [0.0])
depth_mm = read_v30_lossless_video(root, 0, "observation.images.head_depth", [0.0])
# rgb: (1,3,720,1280); depth_mm: (1,1,720,1280)
```

其他 LeRobot 版本、pi0.5 或 AHA-WAM 训练读取器不能把打包深度当作普通 RGB 视频；
必须接入此解码器或实现上述字节协议。不要对打包视频做 resize、颜色转换或有损转码
再解包；应先解包得到毫米值再进行模型预处理。

## 16 位灰度深度 MKV（最新采集默认）

无损采集新增 `--control.depth_video_format=gray16_mkv`，当前为默认值。
该模式要求相机配置保留毫米单位、上限 1 m。RGB 仍为无损 RGB24 H.264 MP4，
深度改为 FFV1/gray16le MKV；`packed_mp4` 可读取/采集旧式打包格式。

深度像素写入公式 `gray16=round(depth_mm*65535/1000)`，读取公式
`depth_mm=round(gray16*1000/65535)`。0 无效/黑色，1000 mm 为白色；
1～1000 mm 每个整数均可精确恢复，保存值已经占用整个 16 位亮度范围。
这表示保存分辨率为 1 mm，不表示相机测量误差≤1 mm。
每次编码同时验证解码灰度像素与源映射一致、恢复毫米值与采集毫米值一致，再删除临时 PNG。

路径示例：
- RGB：`videos/observation.images.head/chunk-000/file-000.mp4`
- 深度：`videos/observation.images.head_depth/chunk-000/file-000.mkv`

v2.1 路径解析、v3.0 转换和 `read_v30_lossless_video` 已支持混合容器；
应依据每个 feature 的 `lossless_encoding` 解析后缀，不能假设全部视频均为 MP4。
其余读取示例不变，深度输出仍为单通道毫米 float32，RGB 为正常 RGB float32 [0,1]。

RGB 正确读取使用 PyAV 软件解码并明确转成 `rgb24`，不使用浏览器的预览输出。
这不会改变原始无损 RGB 编码，不能保证 VS Code 的播放器消除原有偏绿兼容性问题。
如需普通播放器可观看副本，结束采集后执行（输出目录必须不存在）：

```bash
python -m beavr.scripts.preview_lossless_rgbd \
  --root datasets/实际数据集目录 \
  --output datasets/实际数据集目录/previews_compatible
```

预览 RGB 为有损兼容 MP4，深度为 0～1 m 的 8 位灰度预览；只用于观看，
不替换无损训练文件。MKV 已直接存储全范围灰度，但播放器仍需支持 FFV1/gray16。
此更新不会自动改写已采集的旧数据集。

## 按 data_100 保存查看版本，同时保留无损原件

实查参考 `/mnt/public_data/lkw_data_files/data_100`：LeRobot v3.0，RGB AV1/YUV420
MP4，原参考尺寸 640×480，没有深度。根据本次确认，新增版本只复用其 RGB 编码方式；
保留当前相机名称、原生尺寸、RGB-D 对齐和 0～1 m 灰度 uint16 深度 MKV，
不猜左右、不缩放、不改成 PNG。

在上述无损采集命令中增加：

```bash
--control.export_data100=true
```

必须同时使用 `lossless_video=true, depth_video_format=gray16_mkv, dataset_format=v3.0`。
录制结束、无损编码校验及 v3.0 转换成功后生成同级目录：

- `<root>/`：原始无损 RGB MP4 和 uint16 灰度深度 MKV。
- `<root>_data100/`：正式查看版本；RGB 是 AV1/YUV420 MP4，深度 MKV 逐字节复制。

两份包含相同的动作、状态、时间、相机配置和原生空间尺寸。查看版本的 RGB 是有损副本，
不能称为像素无损；深度保持相同精度。MKV 的播放器支持要求没有改变。
`meta/export_provenance.json` 记录无损源路径、RGB 转换方式和帧数；
查看版本仍保留源数据的编码前统计值，符合采集器原有统计习惯，RGB 压缩会改变部分像素。

转换已有数据（输出目录必须不存在）：

```bash
python -m beavr.scripts.export_data100_rgbd \
  --source datasets/fa_rgbd_mkv_20260921_193146 \
  --output datasets/fa_rgbd_mkv_20260921_193146_data100
```

转换阶段写入新 `.partial` 目录，全部视频解码帧数验证通过后才发布最终目录。
失败时不改源数据；检查 `.partial` 和错误日志后恢复或选择新输出目录。
RGB 读取依然使用显式软件 `rgb24` 解码，深度继续逆映射为毫米。

## 一次采集，同时保存无损数据和 RGB/深度 MP4 预览

在无损 v3.0 命令中设置 `--control.export_previews=true`，并设置
`--control.export_data100=false`。同一个数据集内自动生成：

```text
data/chunk-000/file-000.parquet
meta/info.json
meta/episodes/...
videos/observation.images.head/chunk-000/file-000.mp4        # 无损 RGB
videos/observation.images.head_depth/chunk-000/file-000.mkv  # 无损 16 位灰度深度
previews/observation.images.head/chunk-000/file-000.mp4      # 兼容播放 RGB
previews/observation.images.head_depth/chunk-000/file-000.mp4 # 黑白深度预览
previews/manifest.json
```

两台腕部相机同样保存。`videos/` 是训练字段引用的无损文件，`previews/` 是不计入
`meta/info.json` 视频特征和数量的观看副本。深度单通道和 MKV 扩展仍需本仓库配套读取器，
不保证未适配的上游 LeRobot 版本能直接加载深度。预览采用 H.264/YUV420，
深度先恢复毫米再按 0～1000 mm 映射成黑白图，0 无效为黑色；预览是有损的。

生成流程：临时 PNG -> 无损编码和像素校验 -> 清理 PNG -> v3.0 转换 ->
软件 RGB 解码/深度恢复 -> 六路 MP4 预览 -> 逐路解码验证帧数、记录 manifest。
全部预览完成才将 `previews.partial` 发布为 `previews`；失败时训练数据仍保留。
无需额外手动执行预览导出，也不需要另外的 `_data100` 数据集。
