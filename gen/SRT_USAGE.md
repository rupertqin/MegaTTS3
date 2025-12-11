# SRT 字幕生成功能使用说明

## 功能概述

现在系统支持自动生成 SRT 字幕文件，与音频生成完全同步。

## 生成流程

### 1. 生成音频和字幕（完整流程）

```bash
./gen/gen.sh
# 或
./gen/gen.sh both
```

这将：

- 逐句生成音频文件（001_xxx.wav, 002_xxx.wav, ...）
- 为每个音频生成对应的 SRT 字幕文件（001_xxx.srt, 002_xxx.srt, ...）
- 合并所有音频为 merged_audio_xxx.wav
- 合并所有字幕为 merged_subtitles.srt

### 2. 仅生成音频

```bash
./gen/gen.sh generate
```

### 3. 仅合并音频和生成字幕

```bash
./gen/gen.sh merge
```

这将：

- 根据 generation_report.json 合并音频
- 同时生成所有 SRT 字幕文件（单独的和合并的）

### 4. 仅生成 SRT 字幕

```bash
./gen/gen.sh srt
```

这将：

- 根据 generation_report.json 和已有音频文件
- 为每个音频片段生成独立的 SRT 文件
- 生成合并的 merged_subtitles.srt

## 输出文件结构

```
gen/audio/
├── 001_最近有个传闻在网上闹得沸沸扬扬.wav
├── 001_最近有个传闻在网上闹得沸沸扬扬.srt
├── 002_说康熙皇帝可能是个汉人.wav
├── 002_说康熙皇帝可能是个汉人.srt
├── ...
├── merged_audio_1234567890.wav
├── merged_subtitles.srt
└── generation_report.json
```

## SRT 文件格式

### 单独的 SRT 文件（例如 001_xxx.srt）

```
1
00:00:00,000 --> 00:00:03,500
最近有个传闻在网上闹得沸沸扬扬，说康熙皇帝可能是个汉人。
```

### 合并的 SRT 文件（merged_subtitles.srt）

```
1
00:00:00,000 --> 00:00:03,500
最近有个传闻在网上闹得沸沸扬扬，说康熙皇帝可能是个汉人。

2
00:00:03,510 --> 00:00:07,200
这事儿不论真假，它其实暴露了一个极具讽刺意味的底层逻辑。

3
00:00:07,210 --> 00:00:12,800
哪怕是坐拥天下、有着最森严后宫管理的封建帝王，依然无法在生物学上彻底豁免"喜当爹"的风险。
```

## Python 命令行使用

### 生成音频时自动记录时长

```bash
python incremental_tts_generator.py \
  --input_wav "assets/Chinese_prompt.wav" \
  --input_text "你的文本内容" \
  --output_dir "gen/audio"
```

### 仅生成 SRT 字幕

```bash
python incremental_tts_generator.py \
  --srt_only \
  --output_dir "gen/audio" \
  --merge_gap 10
```

### 合并音频和生成字幕

```bash
python incremental_tts_generator.py \
  --merge_only \
  --output_dir "gen/audio" \
  --merge_gap 10
```

## 参数说明

- `--merge_gap`: 音频片段之间的静音间隔（毫秒），默认 500ms
  - 这个间隔也会反映在合并的 SRT 字幕时间轴中

## 技术细节

1. **时长计算**：每个音频文件的时长通过 wave 模块精确计算
2. **时间同步**：SRT 时间轴与音频完全同步，包括静音间隔
3. **格式标准**：SRT 格式符合标准规范（HH:MM:SS,mmm）
4. **编码支持**：所有 SRT 文件使用 UTF-8 编码，支持中英文

## 注意事项

- 必须先运行音频生成步骤，才能生成 SRT 字幕
- SRT 字幕依赖 generation_report.json 文件
- 如果音频文件被删除或移动，需要重新生成
- 合并的 SRT 字幕时间轴会自动考虑静音间隔
