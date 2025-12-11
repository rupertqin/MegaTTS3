# 快速参考

## 命令速查

```bash
# 完整流程（生成音频+字幕+合并）
./gen.sh

# 仅生成音频
./gen.sh generate

# 合并音频并生成字幕
./gen.sh merge

# 仅生成字幕
./gen.sh srt
```

## 字幕分句规则

- **最大长度**: 15 个字
- **最小长度**: 10 个字
- **自动合并**: 少于 10 字的片段会自动合并
- **去除标点**: 自动去除末尾标点符号

## 输出文件

```
gen/audio/
├── 001_句子内容.wav          # 音频片段
├── 001_句子内容.srt          # 对应字幕
├── 002_句子内容.wav
├── 002_句子内容.srt
├── ...
├── merged_audio_xxx.wav      # 合并音频
├── merged_subtitles.srt      # 合并字幕
└── generation_report.json    # 生成报告
```

## SRT 格式示例

### 单独字幕（001_xxx.srt）

```srt
1
00:00:00,000 --> 00:00:03,500
这是第一句话的内容。
```

### 合并字幕（merged_subtitles.srt）

```srt
1
00:00:00,000 --> 00:00:03,500
这是第一句话的内容。

2
00:00:03,510 --> 00:00:07,200
这是第二句话的内容。

3
00:00:07,210 --> 00:00:10,800
这是第三句话的内容。
```

## 参数调整

编辑 `gen.sh` 中的参数：

```bash
# 清晰度权重（1.0-3.0）
--p_w 1.0

# 相似度权重（2.0-5.0）
--t_w 2.5

# 静音间隔（毫秒）
--merge_gap 10
```

## Python 直接调用

```bash
# 生成音频
python incremental_tts_generator.py \
  --input_wav "assets/Chinese_prompt.wav" \
  --input_text "你的文本" \
  --output_dir "gen/audio"

# 合并音频和生成字幕
python incremental_tts_generator.py \
  --merge_only \
  --output_dir "gen/audio" \
  --merge_gap 10

# 仅生成字幕
python incremental_tts_generator.py \
  --srt_only \
  --output_dir "gen/audio"
```

## 常见问题

**Q: 如何重新生成字幕？**

```bash
./gen.sh srt
```

**Q: 如何调整字幕时间间隔？**
修改 `gen.sh` 中的 `--merge_gap` 参数（单位：毫秒）

**Q: 字幕文件在哪里？**
与音频文件在同一目录，扩展名为 `.srt`

**Q: 如何查看生成报告？**

```bash
cat gen/audio/generation_report.json
```

**Q: 支持哪些播放器？**
所有支持标准 SRT 格式的播放器（VLC、PotPlayer、MPC-HC 等）
