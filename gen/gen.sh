#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SPEECH_FILE="$SCRIPT_DIR/speech.txt"
SAMPLE_TEXT="你好，这是一个示例文本。\nThis is a sample text.\n"

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

# 读取并预处理文本内容，去除多余的空格和换行符
INPUT_TEXT=$(cat "$SPEECH_FILE" | tr -d '\n' | sed 's/  */ /g' | sed 's/^ *//;s/ *$//')

# 默认值: p_w = 1.6，t_w = 2.5
# p_w（intelligibility weight）：约 1.0–3.0（有时可试到 0.5–5.0）；越大语音越"标准化"。
# t_w（similarity weight）：约 2.0–5.0；越大更偏向说话人相似性。
PYTHONPATH="$REPO_ROOT" python "$REPO_ROOT/tts/infer_cli.py" --input_wav "assets/Chinese_prompt.wav" \
  --input_text "$INPUT_TEXT" --output_dir "gen/audio" \
  --p_w 1.0 --t_w 2.5
