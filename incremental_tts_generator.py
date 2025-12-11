#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增量式逐句音频生成器
按句子逐个生成音频，支持跳过已存在的文件
"""

import os
import sys
import json
import hashlib
import argparse
import time
from pathlib import Path
from tqdm import tqdm
import wave

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入核心模块
from tts.infer_cli import MegaTTS3DiTInfer
from tts.utils.text_utils.split_text import chunk_text_chinesev2, chunk_text_english
from langdetect import detect as classify_language
import torch

class IncrementalTTSGenerator:
    def __init__(self, reference_wav, reference_npy=None, device=None):
        self.reference_wav = reference_wav
        self.reference_npy = reference_npy or reference_wav.replace('.wav', '.npy')
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")

        # 初始化模型
        print(f"🔄 初始化模型，设备: {self.device}")
        self.infer_pipe = MegaTTS3DiTInfer(device=self.device)

        # 加载参考音频
        print("🔄 加载参考音频...")
        with open(self.reference_wav, 'rb') as file:
            file_content = file.read()
        self.resource_context = self.infer_pipe.preprocess(file_content, latent_file=self.reference_npy)
        print("✅ 模型初始化完成")

        # 导入必要的函数
        from tts.frontend_function import g2p, dur_pred, prepare_inputs_for_dit
        from tts.utils.audio_utils.io import to_wav_bytes
        from tn.chinese.normalizer import Normalizer as ZhNormalizer
        from tn.english.normalizer import Normalizer as EnNormalizer
        import pyloudnorm as pyln
        import numpy as np

        self.g2p = g2p
        self.dur_pred = dur_pred
        self.prepare_inputs_for_dit = prepare_inputs_for_dit
        self.to_wav_bytes = to_wav_bytes
        self.zh_normalizer = ZhNormalizer()
        self.en_normalizer = EnNormalizer()
        self.pyln = pyln
        self.np = np

    def generate_filename(self, text, index=None):
        """生成音频文件名，使用完整文本内容（去除标点符号）"""
        import re

        # 去除所有标点符号和特殊字符，只保留字母、数字、中文
        # 保留空格用于后续替换
        safe_text = re.sub(r'[^\w\s]', '', text)
        # 去除多余空格
        safe_text = re.sub(r'\s+', '_', safe_text.strip())

        # 如果文本太长，截断但保留完整性
        max_length = 200  # 文件名最大长度
        if len(safe_text) > max_length:
            safe_text = safe_text[:max_length]

        if index is not None:
            filename = f"{index:03d}_{safe_text}.wav"
        else:
            filename = f"{safe_text}.wav"

        return filename

    def generate_single_sentence(self, text, seg_i, total_segs, time_step, p_w, t_w, dur_disturb=0.1, dur_alpha=1.0):
        """生成单个句子的音频（底层实现，不调用forward）"""
        device = self.device

        # 从 resource_context 获取参考数据
        ph_ref = self.resource_context['ph_ref'].to(device)
        tone_ref = self.resource_context['tone_ref'].to(device)
        mel2ph_ref = self.resource_context['mel2ph_ref'].to(device)
        vae_latent = self.resource_context['vae_latent'].to(device)
        ctx_dur_tokens = self.resource_context['ctx_dur_tokens'].to(device)
        incremental_state_dur_prompt = self.resource_context['incremental_state_dur_prompt']

        with torch.inference_mode():
            # G2P
            ph_pred, tone_pred = self.g2p(self.infer_pipe, text)

            # Duration Prediction
            mel2ph_pred = self.dur_pred(
                self.infer_pipe, ctx_dur_tokens, incremental_state_dur_prompt,
                ph_pred, tone_pred, seg_i, dur_disturb, dur_alpha,
                is_first=(seg_i==0), is_final=(seg_i==total_segs-1)
            )

            # Prepare inputs
            inputs = self.prepare_inputs_for_dit(
                self.infer_pipe, mel2ph_ref, mel2ph_pred,
                ph_ref, tone_ref, ph_pred, tone_pred, vae_latent
            )

            # DiT inference
            with torch.cuda.amp.autocast(dtype=self.infer_pipe.precision, enabled=True):
                x = self.infer_pipe.dit.inference(inputs, timesteps=time_step, seq_cfg_w=[p_w, t_w]).float()

            # WavVAE decode
            x[:, :vae_latent.size(1)] = vae_latent
            wav_pred = self.infer_pipe.wavvae.decode(x)[0,0].to(torch.float32)

            # Post-processing
            wav_pred = wav_pred[vae_latent.size(1)*self.infer_pipe.vae_stride*self.infer_pipe.hop_size:].cpu().numpy()

            # Normalize loudness
            meter = self.pyln.Meter(self.infer_pipe.sr)
            loudness_pred = self.infer_pipe.loudness_meter.integrated_loudness(wav_pred.astype(float))
            wav_pred = self.pyln.normalize.loudness(wav_pred, loudness_pred, self.infer_pipe.loudness_prompt)
            if self.np.abs(wav_pred).max() >= 1:
                wav_pred = wav_pred / self.np.abs(wav_pred).max() * 0.95

            return wav_pred

    def generate_single_audio(self, text, output_dir, index=None, total_segs=1, **kwargs):
        """生成单个音频文件"""
        # 生成文件名
        filename = self.generate_filename(text, index)
        output_path = os.path.join(output_dir, filename)

        try:
            print(f"   📝 文本: {text[:50]}{'...' if len(text) > 50 else ''}")

            # 生成音频
            start_time = time.time()
            seg_i = (index - 1) if index else 0
            wav_pred = self.generate_single_sentence(
                text, seg_i, total_segs,
                kwargs.get('time_step', 16),
                kwargs.get('p_w', 1.6),
                kwargs.get('t_w', 2.5),
                kwargs.get('dur_disturb', 0.1),
                kwargs.get('dur_alpha', 1.0)
            )

            # 转换为 wav bytes 并保存
            wav_bytes = self.to_wav_bytes(wav_pred, self.infer_pipe.sr)
            from tts.utils.audio_utils.io import save_wav
            save_wav(wav_bytes, output_path)

            generation_time = time.time() - start_time

            print(f"   ✅ 完成 - 耗时: {generation_time:.1f}秒")

            # 获取音频时长
            duration = self.get_audio_duration(output_path)

            return {
                'status': 'success',
                'text': text,
                'output_path': output_path,
                'filename': filename,
                'index': index,
                'generation_time': generation_time,
                'duration': duration
            }

        except Exception as e:
            print(f"   ❌ 生成失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'status': 'failed',
                'text': text,
                'output_path': None,
                'filename': filename,
                'index': index,
                'error': str(e)
            }

    def get_audio_duration(self, audio_path):
        """获取音频文件时长（秒）"""
        try:
            with wave.open(audio_path, 'rb') as audio_file:
                frames = audio_file.getnframes()
                rate = audio_file.getframerate()
                duration = frames / float(rate)
                return duration
        except Exception as e:
            print(f"⚠️  获取音频时长失败: {e}")
            return 0.0

    def normalize_text_for_matching(self, text):
        """标准化文本用于匹配：去除所有标点和空格"""
        import re
        return re.sub(r'[^\w]', '', text)

    def load_existing_report(self, output_dir):
        """加载已有的生成报告"""
        report_path = os.path.join(output_dir, 'generation_report.json')
        if os.path.exists(report_path):
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    report = json.load(f)
                return report
            except Exception as e:
                print(f"⚠️  读取报告失败: {e}")
        return None

    def build_existing_sentences_map(self, existing_report):
        """构建已处理句子的映射表（使用标准化文本作为key）"""
        existing_sentences = {}
        if existing_report:
            for result in existing_report.get('results', []):
                if result['status'] in ['success', 'skipped']:
                    # 标准化文本：去除所有标点和空格，用于匹配
                    normalized_text = self.normalize_text_for_matching(result['text'])
                    existing_sentences[normalized_text] = result
        return existing_sentences

    def process_text_file(self, text_file, output_dir, force_regenerate=False, **kwargs):
        """处理文本文件，逐句生成音频"""
        # 读取文本文件
        with open(text_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        # 检测语言并分句
        language_type = classify_language(content)

        if language_type == 'en':
            print("🌍 检测到英文文本")
            content = self.en_normalizer.normalize(content)
            text_segs = chunk_text_english(content, max_chars=130)
        else:
            print("🌍 检测到中文文本")
            content = self.zh_normalizer.normalize(content)
            text_segs = chunk_text_chinesev2(content, limit=60)

        print(f"📝 文本已分句，共 {len(text_segs)} 句")
        print(f"📁 输出目录: {output_dir}")

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 处理每个句子
        results = []
        for i, sentence in enumerate(text_segs):
            print(f"\n正在处理第 {i+1}/{len(text_segs)} 句")

            # 生成文件名，检查文件是否已存在
            filename = self.generate_filename(sentence, i+1)
            output_path = os.path.join(output_dir, filename)

            if os.path.exists(output_path) and not force_regenerate:
                print(f"⏭️  跳过已存在的文件: {filename}")
                print(f"   📝 文本: {sentence[:50]}{'...' if len(sentence) > 50 else ''}")
                # 获取已存在文件的时长
                duration = self.get_audio_duration(output_path)
                results.append({
                    'status': 'skipped',
                    'text': sentence,
                    'output_path': output_path,
                    'filename': filename,
                    'index': i+1,
                    'duration': duration
                })
                continue

            result = self.generate_single_audio(
                sentence,
                output_dir,
                index=i+1,
                total_segs=len(text_segs),
                **kwargs
            )
            results.append(result)

        # 生成报告
        self.generate_report(results, output_dir, text_file)

        return results

    def process_direct_text(self, input_text, output_dir, force_regenerate=False, **kwargs):
        """处理直接输入的文本，逐句生成音频"""
        # 检测语言并分句
        language_type = classify_language(input_text)

        if language_type == 'en':
            print("🌍 检测到英文文本")
            input_text = self.en_normalizer.normalize(input_text)
            text_segs = chunk_text_english(input_text, max_chars=130)
        else:
            print("🌍 检测到中文文本")
            input_text = self.zh_normalizer.normalize(input_text)
            text_segs = chunk_text_chinesev2(input_text, limit=60)

        print(f"📝 文本已分句，共 {len(text_segs)} 句")
        print(f"📁 输出目录: {output_dir}")

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 处理每个句子
        results = []
        for i, sentence in enumerate(text_segs):
            print(f"\n正在处理第 {i+1}/{len(text_segs)} 句")

            # 生成文件名，检查文件是否已存在
            filename = self.generate_filename(sentence, i+1)
            output_path = os.path.join(output_dir, filename)

            if os.path.exists(output_path) and not force_regenerate:
                print(f"⏭️  跳过已存在的文件: {filename}")
                print(f"   📝 文本: {sentence[:50]}{'...' if len(sentence) > 50 else ''}")
                # 获取已存在文件的时长
                duration = self.get_audio_duration(output_path)
                results.append({
                    'status': 'skipped',
                    'text': sentence,
                    'output_path': output_path,
                    'filename': filename,
                    'index': i+1,
                    'duration': duration
                })
                continue

            result = self.generate_single_audio(
                sentence,
                output_dir,
                index=i+1,
                total_segs=len(text_segs),
                **kwargs
            )
            results.append(result)

        # 生成报告
        self.generate_report(results, output_dir, "直接输入文本")

        return results

    def generate_report(self, results, output_dir, text_file):
        """生成处理报告"""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'source_file': str(text_file),
            'output_directory': str(output_dir),
            'reference_audio': self.reference_wav,
            'device': self.device,
            'summary': {
                'total': len(results),
                'success': len([r for r in results if r['status'] == 'success']),
                'skipped': len([r for r in results if r['status'] == 'skipped']),
                'failed': len([r for r in results if r['status'] == 'failed'])
            },
            'results': results
        }

        # 保存报告
        report_path = os.path.join(output_dir, 'generation_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n📊 处理报告:")
        print(f"   📝 总句数: {report['summary']['total']}")
        print(f"   ✅ 成功: {report['summary']['success']}")
        print(f"   ⏭️  跳过: {report['summary']['skipped']}")
        print(f"   ❌ 失败: {report['summary']['failed']}")
        print(f"   📄 报告保存: {report_path}")

def generate_srt_files(output_dir, gap_ms=500):
    """
    根据 generation_report.json 生成 SRT 字幕文件
    为每个音频片段生成独立的 SRT 文件，并生成合并的 SRT 文件
    """
    print("📝 开始生成 SRT 字幕文件...")

    # 读取 generation_report.json
    report_path = os.path.join(output_dir, 'generation_report.json')
    if not os.path.exists(report_path):
        print(f"❌ 未找到生成报告: {report_path}")
        print("   请先运行生成步骤")
        return None

    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
    except Exception as e:
        print(f"❌ 读取报告文件失败: {e}")
        return None

    # 从报告中提取成功生成的音频文件
    subtitle_entries = []
    for result in report.get('results', []):
        if result['status'] in ['success', 'skipped'] and result.get('output_path'):
            file_path = result['output_path']
            if os.path.exists(file_path):
                # 优先使用报告中的 duration，如果没有则动态计算
                duration = result.get('duration')
                if duration is None or duration == 0.0:
                    # 动态计算音频时长
                    try:
                        with wave.open(file_path, 'rb') as audio_file:
                            frames = audio_file.getnframes()
                            rate = audio_file.getframerate()
                            duration = frames / float(rate)
                    except Exception as e:
                        print(f"⚠️  无法获取音频时长: {result['filename']}, 错误: {e}")
                        duration = 3.0  # 默认 3 秒

                subtitle_entries.append({
                    'index': result['index'],
                    'text': result['text'],
                    'filename': result['filename'],
                    'duration': duration
                })
            else:
                print(f"⚠️  文件不存在，跳过: {result['filename']}")

    if not subtitle_entries:
        print("❌ 没有找到可生成字幕的音频文件")
        return None

    print(f"📁 从报告中找到 {len(subtitle_entries)} 个音频文件")

    # 生成单独的 SRT 文件
    for entry in subtitle_entries:
        srt_filename = entry['filename'].replace('.wav', '.srt')
        srt_path = os.path.join(output_dir, srt_filename)

        # 将文本分割成多行（每行不超过15个字）
        subtitle_lines = split_subtitle_text(entry['text'], max_chars=15)

        # 计算每行的时长（平均分配）
        total_duration = entry['duration']
        line_duration = total_duration / len(subtitle_lines) if subtitle_lines else total_duration

        # 生成 SRT 内容
        srt_content = ""
        for i, line in enumerate(subtitle_lines, start=1):
            start_time = format_srt_time((i - 1) * line_duration)
            end_time = format_srt_time(i * line_duration)
            srt_content += f"{i}\n{start_time} --> {end_time}\n{line}\n\n"

        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)

        print(f"✅ 生成字幕: {srt_filename} ({len(subtitle_lines)} 行)")

    # 生成合并的 SRT 文件
    merged_srt_path = os.path.join(output_dir, "merged_subtitles.srt")
    gap_seconds = gap_ms / 1000.0
    current_time = 0.0
    subtitle_index = 1

    with open(merged_srt_path, 'w', encoding='utf-8') as f:
        for entry in subtitle_entries:
            # 将文本分割成多行（每行不超过15个字）
            subtitle_lines = split_subtitle_text(entry['text'], max_chars=15)

            # 计算每行的时长（平均分配）
            total_duration = entry['duration']
            line_duration = total_duration / len(subtitle_lines) if subtitle_lines else total_duration

            # 为每一行生成字幕条目
            for line in subtitle_lines:
                start_time = format_srt_time(current_time)
                end_time = format_srt_time(current_time + line_duration)

                f.write(f"{subtitle_index}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{line}\n\n")

                subtitle_index += 1
                current_time += line_duration

            # 添加间隔时间（在句子之间）
            current_time += gap_seconds

    print(f"\n✅ SRT 字幕生成完成!")
    print(f"📁 合并字幕文件: {merged_srt_path}")
    print(f"📊 音频片段数: {len(subtitle_entries)}")
    print(f"📊 字幕条目数: {subtitle_index - 1}")

    return merged_srt_path

def count_chars_without_punctuation(text):
    """计算文本字符数，不包括标点符号"""
    import re
    clean_text = re.sub(r'[，。！？；：、,\.!?;:"""''「」『』（）\(\)\s]', '', text)
    return len(clean_text)

def split_subtitle_text(text, max_chars=15, min_chars=10):
    """
    将长文本分割成多个字幕行
    - 按标点符号分割（不包括引号）
    - 每行不超过 max_chars 个字符（不计标点）
    - 优先在"的"、"地"、"得"等助词处断开
    - 引号不计入字符数
    """
    import re

    # 只按这些标点分割，不包括引号
    segments = re.split(r'[，。！？；：、,\.!?;:]', text)

    # 去除空白片段
    segments = [s.strip() for s in segments if s.strip()]

    if not segments:
        return []

    # 第一步：合并过短的片段
    merged = []
    i = 0
    while i < len(segments):
        current = segments[i]
        current_len = count_chars_without_punctuation(current)

        while current_len < min_chars and i + 1 < len(segments):
            next_text = segments[i + 1]
            combined = current + next_text
            combined_len = count_chars_without_punctuation(combined)

            if combined_len <= max_chars:
                i += 1
                current = combined
                current_len = combined_len
            else:
                break

        merged.append(current)
        i += 1

    # 第二步：智能分割超长片段
    def find_best_split_point(text_part, target_len, min_len):
        """找到最佳分割点"""
        priority_chars = ['的', '地', '得', '了', '着', '过']
        secondary_chars = ['是', '在', '和', '与', '或', '及', '把', '被']

        # 计算每个位置的实际字符数
        positions = []
        count = 0
        for c in text_part:
            if not re.match(r'[，。！？；：、,\.!?;:"""''「」『』（）\(\)\s]', c):
                count += 1
            positions.append(count)

        best_pos = -1
        best_score = -1000

        for i in range(len(text_part)):
            curr_len = positions[i]
            remain_len = count - curr_len

            if curr_len < min_len or curr_len > target_len:
                continue

            score = 0

            # 下一个字符是优先字符，加分
            if i + 1 < len(text_part) and text_part[i + 1] in priority_chars:
                score += 15
            elif i + 1 < len(text_part) and text_part[i + 1] in secondary_chars:
                score += 8

            # 越接近目标长度越好
            score -= abs(curr_len - target_len)

            # 剩余部分的长度评分
            if remain_len >= min_len:
                score += 5
            elif remain_len > 0:
                score -= (min_len - remain_len) * 3

            if score > best_score:
                best_score = score
                best_pos = i + 1

        return best_pos if best_pos > 0 else min(target_len, len(text_part))

    split_lines = []
    for part in merged:
        part_len = count_chars_without_punctuation(part)

        if part_len <= max_chars:
            split_lines.append(part)
        else:
            # 超长片段需要智能分割
            remaining = part

            while remaining:
                remaining_len = count_chars_without_punctuation(remaining)

                if remaining_len <= max_chars:
                    split_lines.append(remaining)
                    break

                # 找到最佳分割点
                best_pos = find_best_split_point(remaining, max_chars, min_chars)

                if best_pos > 0 and best_pos < len(remaining):
                    split_lines.append(remaining[:best_pos].strip())
                    remaining = remaining[best_pos:].strip()
                else:
                    split_lines.append(remaining)
                    break

    # 第三步：合并过短的片段
    final_lines = []
    for part in split_lines:
        part_len = count_chars_without_punctuation(part)

        if part_len < min_chars and final_lines:
            prev_len = count_chars_without_punctuation(final_lines[-1])
            combined_len = prev_len + part_len

            if combined_len <= max_chars:
                final_lines[-1] = final_lines[-1] + part
            else:
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

def merge_audio_files(output_dir, output_filename="merged_audio.wav", gap_ms=500):
    """
    根据 generation_report.json 合并音频文件
    只合并 JSON 中记录的成功生成的文件，按顺序合并
    """
    print("🎵 开始音频合并...")

    # 读取 generation_report.json
    report_path = os.path.join(output_dir, 'generation_report.json')
    if not os.path.exists(report_path):
        print(f"❌ 未找到生成报告: {report_path}")
        print("   请先运行生成步骤")
        return None

    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
    except Exception as e:
        print(f"❌ 读取报告文件失败: {e}")
        return None

    # 从报告中提取成功生成的音频文件
    audio_files = []
    for result in report.get('results', []):
        if result['status'] in ['success', 'skipped'] and result.get('output_path'):
            file_path = result['output_path']
            if os.path.exists(file_path):
                audio_files.append((file_path, result['filename'], result.get('text', '')[:30]))
            else:
                print(f"⚠️  文件不存在，跳过: {result['filename']}")

    if not audio_files:
        print("❌ 没有找到可合并的音频文件")
        return None

    print(f"📁 从报告中找到 {len(audio_files)} 个音频文件")

    try:
        from pydub import AudioSegment

        # 加载第一个音频文件
        merged = AudioSegment.from_wav(audio_files[0][0])
        print(f"🔄 [1/{len(audio_files)}] {audio_files[0][1]}")
        print(f"   📝 {audio_files[0][2]}...")

        # 合并其他音频文件
        for i, (file_path, file_name, text_preview) in enumerate(audio_files[1:], start=2):
            # 添加静音间隔
            silence = AudioSegment.silent(duration=gap_ms)
            merged += silence

            # 添加音频文件
            audio = AudioSegment.from_wav(file_path)
            merged += audio
            print(f"🔄 [{i}/{len(audio_files)}] {file_name}")
            print(f"   📝 {text_preview}...")

        # 保存合并后的音频
        output_path = os.path.join(output_dir, output_filename)
        merged.export(output_path, format="wav")

        # 计算总时长
        duration_seconds = len(merged) / 1000.0
        duration_minutes = duration_seconds / 60.0

        print(f"\n✅ 音频合并完成!")
        print(f"📁 合并文件: {output_path}")
        print(f"⏱️  总时长: {duration_minutes:.2f} 分钟 ({duration_seconds:.1f} 秒)")
        print(f"🔇 静音间隔: {gap_ms}ms")
        print(f"📊 合并文件数: {len(audio_files)}")

        return output_path

    except ImportError:
        print("❌ 缺少 pydub 库，请安装: pip install pydub")
        return None
    except Exception as e:
        print(f"❌ 音频合并失败: {e}")
        return None

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='增量式逐句音频生成器 - 按句子分段生成和合并音频')

    parser.add_argument('--input_wav', help='参考音频文件')
    parser.add_argument('--input_npy', help='参考特征文件')
    parser.add_argument('--text_file', help='文本文件路径')
    parser.add_argument('--input_text', help='直接输入的文本内容')
    parser.add_argument('--output_dir', default='./output', help='输出目录')
    parser.add_argument('--time_step', type=int, default=16, help='推理步数')
    parser.add_argument('--p_w', type=float, default=1.6, help='清晰度权重 (1.0-3.0)')
    parser.add_argument('--t_w', type=float, default=2.5, help='相似度权重 (2.0-5.0)')
    parser.add_argument('--force', action='store_true', help='强制重新生成已存在的文件')
    parser.add_argument('--merge_only', action='store_true', help='仅执行音频合并（根据generation_report.json）')
    parser.add_argument('--merge_gap', type=int, default=500, help='合并时的静音间隔(ms)')
    parser.add_argument('--srt_only', action='store_true', help='仅生成 SRT 字幕文件（根据generation_report.json）')

    # 解析参数
    args = parser.parse_args()

    if args.srt_only:
        # 仅生成 SRT 字幕
        if not args.output_dir:
            parser.error("--output_dir 在 SRT 生成模式下是必需的")

        result = generate_srt_files(args.output_dir, args.merge_gap)
        return

    if args.merge_only:
        # 仅执行音频合并
        if not args.output_dir:
            parser.error("--output_dir 在合并模式下是必需的")

        output_filename = f"merged_audio_{int(time.time())}.wav"
        audio_result = merge_audio_files(args.output_dir, output_filename, args.merge_gap)

        # 同时生成 SRT 字幕
        print("\n" + "="*80)
        srt_result = generate_srt_files(args.output_dir, args.merge_gap)
        return

    # 检查必需的参数
    if not args.input_wav:
        parser.error("--input_wav 是必需的")

    # 检查文本输入方式
    if not args.text_file and not args.input_text:
        parser.error("需要提供 --text_file 或 --input_text 参数之一")

    if args.text_file and args.input_text:
        parser.error("不能同时提供 --text_file 和 --input_text 参数")

    # 生成参数
    kwargs = {
        'time_step': args.time_step,
        'p_w': args.p_w,
        't_w': args.t_w
    }

    print("="*80)
    print("🚀 增量式逐句音频生成器")
    print("="*80)
    if args.text_file:
        print(f"📁 文本文件: {args.text_file}")
    else:
        print("📝 直接输入文本")
    print(f"🎵 参考音频: {args.input_wav}")
    print(f"📂 输出目录: {args.output_dir}")
    print(f"🔧 推理步数: {args.time_step}")
    print(f"⚙️  清晰度权重(p_w): {args.p_w}")
    print(f"⚙️  相似度权重(t_w): {args.t_w}")
    print(f"⚡ 强制重生成: {args.force}")
    print("="*80)

    # 初始化生成器
    generator = IncrementalTTSGenerator(
        reference_wav=args.input_wav,
        reference_npy=args.input_npy
    )

    # 处理文本
    if args.text_file:
        results = generator.process_text_file(
            text_file=args.text_file,
            output_dir=args.output_dir,
            force_regenerate=args.force,
            **kwargs
        )
    else:
        results = generator.process_direct_text(
            input_text=args.input_text,
            output_dir=args.output_dir,
            **kwargs
        )

    # 提示合并命令
    success_count = sum(1 for r in results if r['status'] == 'success')
    skipped_count = sum(1 for r in results if r['status'] == 'skipped')
    total_audio = success_count + skipped_count

    if total_audio > 0:
        print(f"\n💡 提示: 共有 {total_audio} 个音频文件可合并")
        print("   运行以下命令合并音频和生成字幕:")
        print(f"   python {__file__} --merge_only --output_dir {args.output_dir}")
        print("   或使用脚本:")
        print(f"   ./gen/gen.sh merge")
        print("\n   仅生成 SRT 字幕:")
        print(f"   python {__file__} --srt_only --output_dir {args.output_dir}")

if __name__ == '__main__':
    main()
