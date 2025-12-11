#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SPEECH_FILE="$SCRIPT_DIR/speech.txt"
SAMPLE_TEXT="你好，这是一个示例文本。\nThis is a sample text.\n"

# 默认执行步骤：both（生成+合并）
STEP="${1:-both}"

# 显示使用说明
show_usage() {
    echo "用法: $0 [步骤]"
    echo ""
    echo "步骤选项:"
    echo "  generate  - 仅生成音频文件"
    echo "  merge     - 仅合并已生成的音频文件和生成 SRT 字幕"
    echo "  srt       - 仅生成 SRT 字幕文件"
    echo "  both      - 生成并合并（默认）"
    echo ""
    echo "示例:"
    echo "  $0           # 生成并合并"
    echo "  $0 generate  # 仅生成"
    echo "  $0 merge     # 仅合并音频和生成字幕"
    echo "  $0 srt       # 仅生成字幕"
}

# 检查参数
if [[ "$STEP" != "generate" && "$STEP" != "merge" && "$STEP" != "both" && "$STEP" != "srt" ]]; then
    echo "❌ 错误: 无效的步骤参数 '$STEP'"
    echo ""
    show_usage
    exit 1
fi

if [[ ! -f "$SPEECH_FILE" ]]; then
    mkdir -p "$SCRIPT_DIR"
    echo "$SAMPLE_TEXT" > "$SPEECH_FILE"
    echo "Created a sample speech file at: $SPEECH_FILE"
    echo "Edit the file with the text you want to synthesize, then run ./gen.sh again."
    exit 1
fi

mkdir -p "$SCRIPT_DIR/audio"

# 切换到项目根目录，确保路径正确
cd "$REPO_ROOT"

# 步骤1: 生成音频文件
if [[ "$STEP" == "generate" || "$STEP" == "both" ]]; then
    echo "🎵 步骤1: 生成音频文件"
    echo "================================"

    # 读取并预处理文本内容，去除多余的空格和换行符
    INPUT_TEXT=$(cat "$SPEECH_FILE" | tr -d '\n' | sed 's/  */ /g' | sed 's/^ *//;s/ *$//')

    # 默认值: p_w = 1.6，t_w = 2.5
    # p_w（intelligibility weight）：约 1.0–3.0（有时可试到 0.5–5.0）；越大语音越"标准化"。
    # t_w（similarity weight）：约 2.0–5.0；越大更偏向说话人相似性。
    PYTHONPATH="$REPO_ROOT" python "$REPO_ROOT/incremental_tts_generator.py" --input_wav "assets/Chinese_prompt.wav" \
      --input_text "$INPUT_TEXT" --output_dir "gen/audio" \
      --p_w 1.0 --t_w 2.5

    echo ""
fi

# 步骤2: 合并音频文件和生成 SRT 字幕
if [[ "$STEP" == "merge" || "$STEP" == "both" ]]; then
    echo "🔗 步骤2: 合并音频文件和生成 SRT 字幕"
    echo "================================"

    PYTHONPATH="$REPO_ROOT" python "$REPO_ROOT/incremental_tts_generator.py" \
      --merge_only --output_dir "gen/audio" --merge_gap 10

    echo ""
fi

# 步骤3: 仅生成 SRT 字幕
if [[ "$STEP" == "srt" ]]; then
    echo "📝 步骤: 生成 SRT 字幕文件"
    echo "================================"

    PYTHONPATH="$REPO_ROOT" python "$REPO_ROOT/incremental_tts_generator.py" \
      --srt_only --output_dir "gen/audio" --merge_gap 10

    echo ""
fi

echo "✅ 完成!"
