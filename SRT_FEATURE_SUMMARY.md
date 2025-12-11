# SRT 字幕功能实现总结

## 实现的功能

### 1. 自动生成 SRT 字幕文件

- 为每个音频片段生成独立的 SRT 字幕文件
- 生成合并的 SRT 字幕文件，与合并音频完全同步
- 字幕时间轴精确到毫秒

### 2. 与音频生成完全同步

- 在音频生成时自动记录每个片段的时长
- 字幕时间轴考虑静音间隔
- 单独字幕和合并字幕都能正确生成

### 3. 灵活的生成模式

- `./gen.sh` 或 `./gen.sh both` - 完整流程（生成音频+字幕+合并）
- `./gen.sh generate` - 仅生成音频
- `./gen.sh merge` - 合并音频并生成所有字幕
- `./gen.sh srt` - 仅生成字幕文件

## 修改的文件

### 1. `incremental_tts_generator.py`

**新增功能：**

- `get_audio_duration()` - 获取音频文件时长
- `generate_srt_files()` - 生成 SRT 字幕文件
- `format_srt_time()` - 格式化 SRT 时间戳

**修改功能：**

- `generate_single_audio()` - 添加时长记录
- `process_text_file()` - 为跳过的文件也记录时长
- `process_direct_text()` - 为跳过的文件也记录时长
- `main()` - 添加 `--srt_only` 参数
- `merge_only` 模式现在同时生成 SRT 字幕

### 2. `gen/gen.sh`

**新增功能：**

- 添加 `srt` 步骤选项
- 更新使用说明

**修改功能：**

- `merge` 步骤现在同时生成 SRT 字幕
- 更新帮助文档

### 3. 文档文件

**新增：**

- `gen/SRT_USAGE.md` - SRT 功能详细使用说明
- `gen/example_subtitle.srt` - SRT 格式示例
- `tests/test_srt_generation.py` - SRT 功能测试

**更新：**

- `gen/README.md` - 添加 SRT 功能说明

## 技术实现细节

### 1. 时长计算

使用 Python 的 `wave` 模块精确计算音频时长：

```python
with wave.open(audio_path, 'rb') as audio_file:
    frames = audio_file.getnframes()
    rate = audio_file.getframerate()
    duration = frames / float(rate)
```

### 2. SRT 时间格式

标准 SRT 格式：`HH:MM:SS,mmm`

```python
def format_srt_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
```

### 3. 时间轴同步

- 单独的 SRT 文件：从 0 开始到音频时长结束
- 合并的 SRT 文件：累加时长，包含静音间隔

```python
current_time = 0.0
for entry in subtitle_entries:
    start_time = format_srt_time(current_time)
    end_time = format_srt_time(current_time + entry['duration'])
    # 写入 SRT 条目
    current_time += entry['duration'] + gap_seconds
```

### 4. 数据流

```
音频生成 → 记录时长 → generation_report.json
                              ↓
                    读取报告 + 音频文件
                              ↓
                    生成单独 SRT 文件
                              ↓
                    生成合并 SRT 文件
```

## 使用示例

### 完整流程

```bash
# 1. 编辑文本
vim gen/speech.txt

# 2. 生成音频和字幕
./gen/gen.sh

# 输出：
# gen/audio/001_xxx.wav
# gen/audio/001_xxx.srt
# gen/audio/002_xxx.wav
# gen/audio/002_xxx.srt
# ...
# gen/audio/merged_audio_xxx.wav
# gen/audio/merged_subtitles.srt
```

### 仅生成字幕

```bash
# 如果已有音频文件和 generation_report.json
./gen/gen.sh srt
```

### Python 命令行

```bash
# 仅生成 SRT
python incremental_tts_generator.py --srt_only --output_dir gen/audio

# 合并音频并生成 SRT
python incremental_tts_generator.py --merge_only --output_dir gen/audio --merge_gap 10
```

## 测试验证

运行测试：

```bash
python tests/test_srt_generation.py
```

测试覆盖：

- SRT 时间格式化（多种时长）
- SRT 内容格式（标准格式验证）

## 注意事项

1. **依赖关系**：SRT 生成依赖 `generation_report.json` 文件
2. **编码格式**：所有 SRT 文件使用 UTF-8 编码
3. **时间精度**：时间精确到毫秒（1/1000 秒）
4. **静音间隔**：合并字幕的时间轴包含静音间隔
5. **文件命名**：SRT 文件名与对应的 WAV 文件名一致（扩展名不同）

## 兼容性

- ✅ 支持中文文本
- ✅ 支持英文文本
- ✅ 支持混合语言
- ✅ 标准 SRT 格式，兼容所有主流播放器
- ✅ UTF-8 编码，支持所有 Unicode 字符

## 未来改进方向

1. 支持更多字幕格式（ASS、VTT 等）
2. 支持字幕样式自定义
3. 支持多语言字幕生成
4. 支持字幕时间轴微调
