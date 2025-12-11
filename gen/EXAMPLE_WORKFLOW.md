# 使用示例工作流

## 场景 1：首次使用

```bash
# 1. 编辑文本文件
vim gen/speech.txt

# 2. 运行完整流程
./gen/gen.sh

# 输出结果：
# ✅ gen/audio/001_xxx.wav + 001_xxx.srt
# ✅ gen/audio/002_xxx.wav + 002_xxx.srt
# ✅ ...
# ✅ gen/audio/merged_audio_xxx.wav
# ✅ gen/audio/merged_subtitles.srt
```

## 场景 2：分批生成

```bash
# 1. 先生成音频（可以随时中断）
./gen/gen.sh generate

# 2. 检查音频质量
# 播放 gen/audio/001_xxx.wav 等文件

# 3. 满意后合并并生成字幕
./gen/gen.sh merge
```

## 场景 3：重新生成字幕

```bash
# 如果只想重新生成字幕（音频已存在）
./gen/gen.sh srt

# 这会：
# - 读取 generation_report.json
# - 为每个音频生成新的 SRT 文件
# - 生成新的 merged_subtitles.srt
```

## 场景 4：调整参数后重新合并

```bash
# 1. 编辑 gen.sh，修改 --merge_gap 参数
vim gen/gen.sh
# 将 --merge_gap 10 改为 --merge_gap 500

# 2. 重新合并（不会重新生成音频）
./gen/gen.sh merge

# 这会生成新的合并文件，使用新的间隔时间
```

## 场景 5：使用 Python 直接调用

```bash
# 生成音频和字幕
python incremental_tts_generator.py \
  --input_wav "assets/Chinese_prompt.wav" \
  --input_text "你好，这是测试文本。" \
  --output_dir "gen/audio" \
  --p_w 1.0 \
  --t_w 2.5

# 合并音频和生成字幕
python incremental_tts_generator.py \
  --merge_only \
  --output_dir "gen/audio" \
  --merge_gap 500

# 仅生成字幕
python incremental_tts_generator.py \
  --srt_only \
  --output_dir "gen/audio"
```

## 场景 6：查看生成报告

```bash
# 查看 JSON 格式的详细报告
cat gen/audio/generation_report.json

# 或使用 jq 美化输出
cat gen/audio/generation_report.json | jq .

# 查看摘要信息
cat gen/audio/generation_report.json | jq '.summary'
```

## 场景 7：播放测试

```bash
# 使用 VLC 播放器测试（macOS）
open -a VLC gen/audio/merged_audio_xxx.wav

# 同时打开字幕文件
open gen/audio/merged_subtitles.srt

# 或使用支持字幕的播放器
# 将 merged_audio_xxx.wav 和 merged_subtitles.srt 放在同一目录
# 播放器会自动加载字幕
```

## 输出文件说明

### 单独文件

- `001_句子内容.wav` - 第一句音频
- `001_句子内容.srt` - 第一句字幕（时间从 00:00:00,000 开始）

### 合并文件

- `merged_audio_1234567890.wav` - 所有音频合并（带时间戳）
- `merged_subtitles.srt` - 所有字幕合并（时间轴连续）

### 报告文件

- `generation_report.json` - 详细的生成报告
  - 包含每个句子的文本
  - 对应的文件名
  - 音频时长
  - 生成状态

## 常见操作

### 清理输出目录

```bash
# 删除所有生成的文件（保留 speech.txt）
rm -rf gen/audio/*

# 重新开始
./gen/gen.sh
```

### 只保留合并文件

```bash
# 删除单独的音频和字幕片段
cd gen/audio
rm [0-9][0-9][0-9]_*.wav
rm [0-9][0-9][0-9]_*.srt

# 保留 merged_* 和 generation_report.json
```

### 批量处理多个文本

```bash
# 创建多个文本文件
echo "文本1" > gen/speech1.txt
echo "文本2" > gen/speech2.txt

# 分别处理
for i in 1 2; do
  python incremental_tts_generator.py \
    --input_wav "assets/Chinese_prompt.wav" \
    --text_file "gen/speech${i}.txt" \
    --output_dir "gen/audio${i}"
done
```

## 提示

1. **增量生成**：已存在的文件会自动跳过，可以安全地重复运行
2. **时间戳**：合并文件名包含时间戳，不会覆盖旧文件
3. **报告文件**：generation_report.json 是关键文件，不要删除
4. **字幕编码**：所有 SRT 文件使用 UTF-8 编码
5. **播放器兼容**：标准 SRT 格式，支持 VLC、PotPlayer、MPC-HC 等
