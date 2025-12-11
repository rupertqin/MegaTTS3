#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 SRT 字幕生成功能
"""

def split_subtitle_text(text, max_chars=15, min_chars=10):
    """将长文本分割成多个字幕行"""
    import re

    # 按标点符号分割
    segments = re.split(r'[，。！？；：、,\.!?;:]', text)

    # 去除空白片段
    segments = [s.strip() for s in segments if s.strip()]

    if not segments:
        return []

    # 第一步：合并过短的片段（向后合并）
    merged = []
    i = 0
    while i < len(segments):
        current = segments[i]

        # 如果当前片段少于 min_chars，尝试与下一个合并
        while len(current) < min_chars and i + 1 < len(segments):
            # 检查合并后是否会超过 max_chars
            if len(current + segments[i + 1]) <= max_chars:
                i += 1
                current = current + segments[i]
            else:
                # 如果合并会超过限制，就保持当前状态
                break

        merged.append(current)
        i += 1

    # 第二步：处理超长片段
    split_lines = []
    for part in merged:
        if len(part) <= max_chars:
            split_lines.append(part)
        else:
            # 超长片段按 max_chars 分割
            for j in range(0, len(part), max_chars):
                chunk = part[j:j+max_chars]
                if chunk.strip():
                    split_lines.append(chunk.strip())

    # 第三步：处理仍然过短的片段（向前合并）
    final_lines = []
    for part in split_lines:
        if len(part) < min_chars and final_lines:
            # 尝试与前一行合并
            if len(final_lines[-1] + part) <= max_chars:
                final_lines[-1] = final_lines[-1] + part
            else:
                # 无法合并，保留
                final_lines.append(part)
        else:
            final_lines.append(part)

    return final_lines

def format_srt_time(seconds):
    """将秒数转换为 SRT 时间格式 (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def test_split_subtitle_text():
    """测试字幕文本分割功能"""
    print("测试字幕文本分割...")

    # 测试用例
    test_cases = [
        ("最近有个传闻在网上闹得沸沸扬扬，说康熙皇帝可能是个汉人。", 15, 10),
        ("这是一个很长的句子，需要被分割成多个部分，每个部分不超过十五个字。", 15, 10),
        ("短句子", 15, 10),
        ("这是第一句，这是第二句，这是第三句。", 15, 10),
    ]

    all_passed = True
    for text, max_chars, min_chars in test_cases:
        result = split_subtitle_text(text, max_chars, min_chars)
        print(f"\n原文: {text}")
        print(f"分割结果 ({len(result)} 行):")
        for i, line in enumerate(result, 1):
            length = len(line)
            # 检查：不超过max_chars，且尽量不少于min_chars（除非是短文本或最后一行）
            if length <= max_chars:
                status = "✅"
            else:
                status = "❌"
                all_passed = False
            print(f"  {status} 第{i}行 ({length}字): {line}")

    return all_passed

def test_format_srt_time():
    """测试 SRT 时间格式化函数"""
    print("\n测试 SRT 时间格式化...")

    # 测试用例
    test_cases = [
        (0, "00:00:00,000"),
        (1.5, "00:00:01,500"),
        (65.123, "00:01:05,123"),
        (3661.456, "01:01:01,456"),
        (0.001, "00:00:00,001"),
        (59.999, "00:00:59,999"),
    ]

    all_passed = True
    for seconds, expected in test_cases:
        result = format_srt_time(seconds)
        if result == expected:
            print(f"✅ {seconds}s -> {result}")
        else:
            print(f"❌ {seconds}s -> {result} (期望: {expected})")
            all_passed = False

    if all_passed:
        print("\n✅ 所有测试通过!")
    else:
        print("\n❌ 部分测试失败!")

    return all_passed

def test_srt_content_format():
    """测试 SRT 内容格式"""
    print("\n测试 SRT 内容格式...")

    # 模拟一个 SRT 条目
    index = 1
    start_time = format_srt_time(0)
    end_time = format_srt_time(3.5)
    text = "这是一个测试字幕"

    srt_content = f"{index}\n{start_time} --> {end_time}\n{text}\n"

    expected = "1\n00:00:00,000 --> 00:00:03,500\n这是一个测试字幕\n"

    if srt_content == expected:
        print("✅ SRT 内容格式正确")
        print(f"生成的内容:\n{srt_content}")
        return True
    else:
        print("❌ SRT 内容格式错误")
        print(f"生成的内容:\n{srt_content}")
        print(f"期望的内容:\n{expected}")
        return False

if __name__ == '__main__':
    print("="*60)
    print("SRT 字幕生成功能测试")
    print("="*60)

    test1 = test_split_subtitle_text()
    test2 = test_format_srt_time()
    test3 = test_srt_content_format()

    print("\n" + "="*60)
    if test1 and test2 and test3:
        print("✅ 所有测试通过!")
    else:
        print("❌ 部分测试失败!")
    print("="*60)
