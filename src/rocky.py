"""
Rocky Voice Synthesizer - 伊甸人外星语音合成

基于《挽救计划》中 Rocky 的声音特征进行程序化合成。
Rocky 是伊甸人(Eridians)，用肢体敲击身体产生声波进行交流。

核心特性：确定性随机数生成，相同文本始终产生完全相同的输出。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable

import numpy as np
import soundfile as sf
from scipy.signal import lfilter


# =============================================================================
# 声音预设
# =============================================================================

# 每个预设定义一套独特的声学特征参数
# - f0_curve: 基频曲线类型，影响音高变化模式
# - f0_hz_range: 基频(声音高低)范围
# - f1/f2/f3_range: 共振峰频率范围，决定音色特征
# - duration_range: 每个音节的持续时间范围
# - noise_level: 噪声成分比例，影响声音清晰度
# - speech_rate: 语速系数，控制每秒音节数量

VOICE_PRESET: dict[str, dict] = {
    # --- Rocky / 伊甸人 (Eridian) ---
    # 基于《挽救计划》小说描述：肢体敲击身体产生声波交流
    # 声音特征：低沉钟/鼓质感，像大钟被敲响，共鸣丰富但高频泛音快速衰减
    #
    # 声学参考：
    # - 基频极低 (~40-80 Hz)：像大钟/低音鼓的深沉嗡鸣
    # - 泛音结构：钟/乐器特有的非整数泛音，高频泛音快速衰减
    # - 敲击质感：短促冲击 + 长共鸣尾音
    # - 温暖饱满：低频共鸣丰富，中频圆润，高频不刺耳
    #
    # 关键调整说明：
    # - harmonic_amplitudes: 基频占主导，高次泛音衰减更快，避免刺耳
    # - inharmonic_ratios: 非整数泛音比，产生钟/乐器质感（区别于整数泛音的机械感）
    # - formant范围下调：让声音更温暖低沉
    # - noise_level极低：敲击声纯净，几乎无噪声
    "f0_curve": "rocky",
    "f0_hz_range": (40, 80),           # 极低基频，像大钟/低音鼓
    "f1_range": (80, 160),             # 第一共振峰下调：温暖低沉的嗡鸣核心
    "f2_range": (180, 350),            # 第二共振峰：中低频圆润泛音
    "f3_range": (400, 700),            # 第三共振峰：轻泛音点缀，克制不刺耳
    "duration_range": (0.40, 0.80),
    "noise_level": 0.0003,
    "speech_rate": 0.4,
    "harmonic_decay": 0.5,
    "harmonic_amplitudes": [1.0, 0.25, 0.12, 0.06, 0.03],
    "inharmonic_ratios": [1.0, 2.1, 3.3, 4.7, 6.2],
    "f1_bw_range": (20, 50),
    "f2_bw_range": (40, 80),
    "f3_bw_range": (60, 120),
    "attack_ms": 8,
    "hold_ms": 100,
    "decay_ms": 500,
}


# =============================================================================
# 基频曲线函数 (F0 Curve Functions)
# =============================================================================

# 基频曲线定义声音的音高变化模式
# t: 时间数组, f0: 基频值, 返回: 对应的频率信号
# 这些函数生成调制基频的方式，影响外星语音的"语调"特征

def f0_curve_random_jump(t: np.ndarray, f0: float, rng: np.random.Generator) -> np.ndarray:
    """
    随机跳变基频曲线 - 模拟外星语言的不规则语调
    在短时间窗口内随机跳跃基频，产生外星语言的"机器感"
    """
    f0_signal = np.full_like(t, f0, dtype=np.float64)
    window_size = int(0.05 * len(t))
    jump_interval = max(window_size, rng.integers(low=50, high=200))
    num_jumps = len(t) // jump_interval
    for i in range(num_jumps):
        start = i * jump_interval
        end = min(start + jump_interval, len(t))
        jump = 1.0 + rng.uniform(-0.4, 0.4)
        f0_signal[start:end] *= jump
    return f0_signal


def f0_curve_rocky(t: np.ndarray, f0: float, rng: np.random.Generator) -> np.ndarray:
    """
    Rocky 的颤鸣基频曲线

    特征：稳定、极低频的嗡鸣感
    - 几乎无颤音（乐器质感）
    - 轻微的频率波动模拟自然振动
    """
    base_freq = f0

    # 几乎没有颤音（像钟/乐器的声音）
    vibrato_rate = rng.uniform(1.5, 2.5)
    vibrato_depth = rng.uniform(0.001, 0.003)  # 极轻微
    vibrato = 1.0 + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)

    # 极缓慢的漂移
    drift_freq = rng.uniform(0.2, 0.4)
    drift_depth = rng.uniform(0.0005, 0.002)
    drift = 1.0 + drift_depth * np.sin(2 * np.pi * drift_freq * t)

    f0_signal = np.full_like(t, base_freq, dtype=np.float64) * vibrato * drift
    return f0_signal


def get_f0_curve(name: str) -> Callable:
    """返回指定名称的基频曲线函数"""
    curves = {
        "random_jump": f0_curve_random_jump,
        "rocky": f0_curve_rocky,
    }
    return curves.get(name, f0_curve_rocky)


# =============================================================================
# 谐波系列定义 (Harmonic Series)
# =============================================================================

# 谐波序列决定音色的"质感"和"性格"
# - integer: 标准整数谐波，听感自然/机械
# - inharmonic: 非整数谐波，产生金属/水晶般的泛音
# - sub: 含次谐波(0.5倍频)，增加厚重低频感
# - ultra: 极端稀疏序列，用于尖锐/刺耳的外星感

HARMONIC_SERIES = {
    "integer": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],      # 标准谐波
    "inharmonic": [1, 2.1, 3.3, 4.7, 6.2, 8.1, 10.3, 12.8, 15.6, 18.7],  # 金属感
    "sub": [0.5, 1, 2, 3, 4, 5, 6, 7, 8],           # 含次谐波
    "ultra": [1, 2, 3, 5, 8, 13, 21, 34],           # 稀疏序列
}


# =============================================================================
# 数据类定义 (Data Classes)
# =============================================================================

@dataclass
class VoiceConfig:
    """语音合成配置"""
    sample_rate: int = 22050                         # 采样率 (Hz)
    seed_override: int | None = None                 # 种子覆盖值，用于调试/复现测试


AlienVoiceConfig = VoiceConfig


@dataclass
class AcousticParams:
    """单个声音片段的声学参数"""
    duration: float   # 持续时间 (秒)
    f0: float         # 基频 (Hz) - 声音的音高基准
    f1: float         # 第一共振峰 (Hz) - 主要音色特征
    f2: float         # 第二共振峰 (Hz) - 次要音色特征
    f3: float         # 第三共振峰 (Hz) - 高频音色特征
    f1_bw: float      # 第一共振峰带宽 (Hz)
    f2_bw: float      # 第二共振峰带宽 (Hz)
    f3_bw: float      # 第三共振峰带宽 (Hz)
    harmonics: list[float]  # 谐波倍数列表
    noise_level: float     # 噪声电平 (0-1)
    f0_curve_name: str = "random_jump"  # 基频曲线名称


# =============================================================================
# 核心合成器 (Core Synthesizer)
# =============================================================================

class VoiceSynthesizer:
    """
    程序化外星语音合成器，使用确定性随机数生成。

    相同文本输入始终产生完全相同的音频输出，
    因为随机数生成器通过文本哈希进行种子初始化。
    """

    def __init__(self, config: AlienVoiceConfig | None = None):
        self.config = config or AlienVoiceConfig()
        self.sr = self.config.sample_rate
        self._rng: np.random.Generator | None = None           # 确定性随机生成器
        self._text_for_current_run: str = ""                  # 当前处理的文本

    # -------------------------------------------------------------------------
    # 确定性种子生成 (Deterministic Seed Generation)
    # -------------------------------------------------------------------------

    def _text_to_seed(self, text: str) -> int:
        """将文本转换为确定性种子 - 核心设计：相同文本→相同种子→相同随机序列"""
        if self.config.seed_override is not None:
            return self.config.seed_override
        # SHA-256哈希的前4字节作为种子，保证结果在32位整数范围内
        h = hashlib.sha256(text.encode("utf-8")).digest()
        return int.from_bytes(h[:4], byteorder="big")

    def _init_rngs(self, text: str):
        """从文本哈希初始化随机数生成器"""
        self._text_for_current_run = text
        seed = self._text_to_seed(text)
        self._rng = np.random.default_rng(seed)

    def _ensure_rng(self, text: str):
        """确保RNG已针对给定文本初始化（惰性初始化）"""
        if self._rng is None or self._text_for_current_run != text:
            self._init_rngs(text)

    # -------------------------------------------------------------------------
    # 声学参数生成 (Acoustic Parameter Generation)
    # -------------------------------------------------------------------------

    def _get_voice_params(self) -> dict:
        """获取声学参数集"""
        return VOICE_PRESET

    def _generate_acoustic_params(self) -> AcousticParams:
        """为单个声音片段生成声学参数"""
        self._ensure_rng(self._text_for_current_run)

        p = self._get_voice_params()
        rng = self._rng

        # 谐波序列：支持整数谐波和非整数谐波两种模式
        # 整数谐波：机械/电子感
        # 非整数谐波（钟/乐器）：更温暖、有共鸣感
        use_inharmonic = rng.random() < 0.7  # 70%概率使用钟/乐器泛音
        if use_inharmonic and "inharmonic_ratios" in p:
            harmonics = p["inharmonic_ratios"]
        else:
            harmonics = [1, 2, 3, 4, 5]

        return AcousticParams(
            duration=rng.uniform(p["duration_range"][0], p["duration_range"][1]),
            f0=rng.uniform(p["f0_hz_range"][0], p["f0_hz_range"][1]),
            f1=rng.uniform(p["f1_range"][0], p["f1_range"][1]),
            f2=rng.uniform(p["f2_range"][0], p["f2_range"][1]),
            f3=rng.uniform(p["f3_range"][0], p["f3_range"][1]),
            # 带宽按中心频率的比例随机生成（使用新的带宽配置）
            f1_bw=rng.uniform(p["f1_bw_range"][0], p["f1_bw_range"][1]),
            f2_bw=rng.uniform(p["f2_bw_range"][0], p["f2_bw_range"][1]),
            f3_bw=rng.uniform(p["f3_bw_range"][0], p["f3_bw_range"][1]),
            harmonics=harmonics,
            noise_level=p["noise_level"],
            f0_curve_name=p["f0_curve"],
        )

    # -------------------------------------------------------------------------
    # 音频合成 (Audio Synthesis)
    # -------------------------------------------------------------------------

    def _synthesize_chunk(self, params: AcousticParams) -> np.ndarray:
        """
        从声学参数合成单个音频片段

        Rocky 声音特点（基于小说描述）：
        - 肢体敲击身体产生声波，像大钟被敲响
        - 低沉嗡鸣，钟/鼓质感，共鸣丰富
        - 高频泛音快速衰减，不刺耳
        """
        # 获取参数配置
        p = self._get_voice_params()

        n_samples = int(params.duration * self.sr)
        t = np.arange(n_samples) / self.sr

        f0 = params.f0
        if f0 < 10:
            f0 = 50.0

        # 获取基频曲线函数并生成基频信号
        curve_fn = get_f0_curve(params.f0_curve_name)
        f0_signal = curve_fn(t, f0, self._rng)

        # 限制基频范围防止混叠
        nyquist = self.sr / 2.0
        max_h = max(params.harmonics) if params.harmonics else 1
        f0_signal = np.clip(f0_signal, 0, nyquist / max_h * 0.9)

        signal = np.zeros(n_samples)

        # 谐波叠加：基频主导，高次泛音快速衰减
        # 使用 preset 中的泛音振幅配置
        harmonic_amps = p["harmonic_amplitudes"]
        for i, h in enumerate(params.harmonics[:len(harmonic_amps)]):
            freq = f0_signal * h
            # 基频振幅稍高，其余按比例衰减（钟/乐器特征）
            amplitude = 0.25 * harmonic_amps[i]
            phase = 2 * np.pi * np.cumsum(freq) / self.sr
            signal += amplitude * np.sin(phase)

        # 添加共振峰：塑造温暖低沉的音色
        def apply_formant_peak(sig: np.ndarray, f: float, bw: float, gain_db: float = 6.0) -> np.ndarray:
            if f <= 0 or bw <= 0:
                return sig
            nyquist = self.sr / 2
            if f >= nyquist:
                return sig
            Q = max(f / bw, 0.5)
            A = 10 ** (gain_db / 40.0)
            w0 = 2 * np.pi * f / self.sr
            alpha = np.sin(w0) / (2 * Q)
            b0 = 1 + alpha * A
            b1 = -2 * np.cos(w0)
            b2 = 1 - alpha * A
            a0 = 1 + alpha / A
            a1 = -2 * np.cos(w0)
            a2 = 1 - alpha / A
            b = np.array([b0, b1, b2]) / a0
            a = np.array([1, a1 / a0, a2 / a0])
            return lfilter(b, a, sig)

        # Rocky 共振峰：温暖低沉的嗡鸣核心
        signal = apply_formant_peak(signal, params.f1, params.f1_bw, gain_db=10.0)
        signal = apply_formant_peak(signal, params.f2, params.f2_bw, gain_db=7.0)
        signal = apply_formant_peak(signal, params.f3, params.f3_bw, gain_db=4.0)

        # 注入噪声成分：敲击声几乎无噪声
        if params.noise_level > 0:
            noise = self._rng.standard_normal(n_samples) * params.noise_level
            signal += noise

        # 应用敲击包络：短冲击 + 长共鸣尾音
        # 模拟钟/鼓被敲响的物理过程
        attack_ms = p.get("attack_ms", 3)
        hold_ms = p.get("hold_ms", 20)
        decay_ms = p.get("decay_ms", 300)

        envelope = np.ones(n_samples)
        attack_samples = int(attack_ms * self.sr / 1000)
        hold_samples = int(hold_ms * self.sr / 1000)
        decay_samples = int(decay_ms * self.sr / 1000)

        if attack_samples + hold_samples + decay_samples < n_samples:
            # Attack：指数上升（快速冲击感）
            if attack_samples > 0:
                envelope[:attack_samples] = np.linspace(0, 0.3, attack_samples) ** 0.5
            # Hold：保持峰值
            hold_end = attack_samples + hold_samples
            if hold_samples > 0 and hold_end < n_samples:
                envelope[attack_samples:hold_end] = 0.3 + 0.7 * np.linspace(0, 1, hold_samples)
            # Decay：指数衰减（钟的余韵）
            decay_start = hold_end
            if decay_start < n_samples:
                decay_len = n_samples - decay_start
                decay = np.exp(-3 * np.linspace(0, 1, decay_len))
                envelope[decay_start:] = envelope[decay_start] * decay
        else:
            # 短片段：使用三角形包络
            envelope = np.linspace(0, 1, n_samples) ** 0.5 * np.linspace(1, 0, n_samples) ** 0.5

        signal *= envelope

        return signal

    def _apply_global_modulation(self, signal: np.ndarray) -> np.ndarray:
        """
        应用全局调制效果增强外星感

        Rocky 的调制：极轻微的振幅波动，模拟钟/乐器的自然共振余韵
        注意：Rocky 是敲击声，全局颤音应该非常克制，避免"人在颤抖"的感觉
        """
        if len(signal) == 0:
            return signal
        n = len(signal)
        t = np.arange(n) / self.sr

        # Rocky：极轻微的自然共振，模拟钟被敲响后的余韵
        # 调制深度从 0.1 降到 0.02，更像真实乐器的自然振动
        mod_freq = 0.3 + self._rng.random() * 1.2  # 0.3-1.5 Hz
        tremolo = 1.0 + 0.02 * np.sin(2 * np.pi * mod_freq * t)
        signal = signal * tremolo

        return signal

    # -------------------------------------------------------------------------
    # 公共接口 (Public API)
    # -------------------------------------------------------------------------

    def _parse_text_phrases(self, text: str) -> list[dict]:
        """
        将文本按标点符号解析为短语/句子片段

        标点决定节奏：
        - 句子结束 → 长停顿（0.25-0.50s）
        - 从句/大停顿 → 中等停顿（0.15-0.25s）
        - 短语/分句 → 短停顿（0.08-0.15s）
        - 词间呼吸 → 极短停顿（0.03-0.06s）
        - 强调/间隔 → 可变停顿（0.20-0.40s）
        - 英文标点（.!?）→ 无停顿（作为句子边界标识，不占时间）
        """
        import re

        # 标点类型及其停顿时长配置
        # 按匹配优先级排序（长停顿先匹配，避免短停顿吞掉长停顿的边界）
        punctuation_config = [
            # --- 句子结束 ---
            # 中文句号、感叹号、问号：句子完整结束，长停顿
            (r'[。！？？]',         'sentence_end', 0.25, 0.50),
            # 英文句号、感叹、问号：句子结束，长停顿
            (r'[.!?]',             'sentence_end', 0.25, 0.50),
            # 英文逗号：短停顿
            (r',',                 'phrase_end',   0.08, 0.15),

            # --- 从句/大停顿 ---
            # 中文冒号、引号前：稍长的句中停顿
            (r'[：:]',            'clause_end',  0.15, 0.25),
            # 中文分号：句内并列分句，比冒号短
            (r'[;；]',             'clause_end',  0.12, 0.20),

            # --- 短语/分句 ---
            # 中文逗号、顿号：短停顿
            (r'[，、]',           'phrase_end',   0.08, 0.15),

            # --- 词间/呼吸 ---
            # 省略号、破折号、波浪号：特殊间隔，可长可短
            (r'[…~～]',           'pause',        0.20, 0.40),
            # 连字符：极短停顿（词内连接）或不占时间
            (r'[-–—·•]',          'word_break',   0.00, 0.06),
        ]

        segments: list[dict] = []
        pattern = '|'.join(f'(?P<p{i}>{p})' for i, (p, *_) in enumerate(punctuation_config))
        pattern = re.compile(pattern)

        last_end = 0
        prosody_rng = np.random.default_rng(self._rng.integers(0, 2**31))

        for m in pattern.finditer(text):
            # 纯文本片段（不含标点）
            char_count = m.start() - last_end
            if char_count > 0:
                phrase_text = text[last_end:m.start()]
                segments.append({
                    'text': phrase_text,
                    'char_count': char_count,
                    'pause_type': None,
                    'pause_dur': 0.0,
                })

            # 识别标点类型
            pause_type = None
            pause_dur = 0.0
            for i, (p, label, min_dur, max_dur) in enumerate(punctuation_config):
                if m.group(f'p{i}') is not None:
                    pause_type = label
                    pause_dur = prosody_rng.uniform(min_dur, max_dur)
                    break

            segments.append({
                'text': m.group(),
                'char_count': 0,
                'pause_type': pause_type,
                'pause_dur': pause_dur,
            })
            last_end = m.end()

        # 尾部剩余文本
        if last_end < len(text):
            char_count = len(text) - last_end
            segments.append({
                'text': text[last_end:],
                'char_count': char_count,
                'pause_type': None,
                'pause_dur': 0.0,
            })

        return segments

    def generate(self, text: str) -> np.ndarray:
        """
        从文本生成外星语音音频

        核心入口函数：文本 → 种子 → 短语解析 → 音节生成 → 拼接调制 → 归一化输出

        文本结构驱动韵律：
        - 根据标点符号划分短语，每个短语音节数量 = 字符数 × 语速系数
        - 标点决定停顿：句号 > 分号 > 逗号 > 无标点
        """
        self._init_rngs(text)

        p = self._get_voice_params()
        speech_rate = p["speech_rate"]

        # 按标点解析文本为短语和停顿
        segments = self._parse_text_phrases(text)

        # 计算总音节数（基于实际字符数和语速）
        total_chars = sum(seg['char_count'] for seg in segments)
        # 至少保证 2 个音节，避免过短
        num_chunks = max(2, int(total_chars * speech_rate))
        num_chunks = min(num_chunks, 60)  # 上限防止过长

        # 将音节分配到各个文本短语中（按字符比例）
        syllable_assignments: list[tuple[str, int]] = []
        for seg in segments:
            if seg['char_count'] == 0:
                continue
            seg_syllables = max(1, int(num_chunks * seg['char_count'] / total_chars))
            syllable_assignments.append((seg['text'], seg_syllables))

        # 建立索引映射：每个文本片段 -> 之后紧跟的标点停顿信息
        text_to_pause: dict[str, tuple[str, float]] = {}
        for i, seg in enumerate(segments):
            if seg['char_count'] > 0:
                continue  # 跳过文本条目
            # 当前条目是标点，找到它前面的那个文本条目
            if i > 0 and segments[i - 1]['char_count'] > 0:
                prev_text = segments[i - 1]['text']
                text_to_pause[prev_text] = (seg['pause_type'], seg['pause_dur'])

        chunks: list[np.ndarray] = []

        syllable_idx = 0
        for seg_idx, (seg_text, seg_syllables) in enumerate(syllable_assignments):
            for _j in range(seg_syllables):
                self._ensure_rng(f"{text}:{syllable_idx}")
                params = self._generate_acoustic_params()
                chunk = self._synthesize_chunk(params)
                chunks.append(chunk)
                syllable_idx += 1

        # 每段最后一个 chunk 的全局下标（停顿只允许出现在「段末 → 标点」之间，不能每个音节后都插）
        segment_last_chunk_idx: list[int] = []
        g = 0
        for _seg_text, n_syl in syllable_assignments:
            g += n_syl
            segment_last_chunk_idx.append(g - 1)

        interleaved: list[np.ndarray] = []
        for i, chunk in enumerate(chunks):
            interleaved.append(chunk)
            for seg_idx, last_i in enumerate(segment_last_chunk_idx):
                if i != last_i:
                    continue
                if seg_idx >= len(syllable_assignments) - 1:
                    break
                ended_text = syllable_assignments[seg_idx][0]
                if ended_text in text_to_pause:
                    _, pause_dur = text_to_pause[ended_text]
                    if pause_dur > 0:
                        interleaved.append(np.zeros(int(pause_dur * self.sr)))
                break

        audio = np.concatenate(interleaved)

        # 应用全局颤音调制
        audio = self._apply_global_modulation(audio)

        # 归一化振幅至±0.98范围，防止削波
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.98

        return audio

    def synthesize_file(self, text: str, output_path: str) -> str:
        """生成并保存外星语音音频文件"""
        audio = self.generate(text)
        sf.write(output_path, audio, self.sr)
        return output_path


# =============================================================================
# 便捷函数 (Convenience Functions)
# =============================================================================

def synthesize(text: str, output_path: str | None = None,
               sample_rate: int = 22050) -> np.ndarray | str:
    """
    单次调用完成语音合成（便捷封装）

    Args:
        text: 输入文本（作为确定性种子）
        output_path: 若提供则保存至该路径并返回路径字符串，否则返回numpy数组
        sample_rate: 音频采样率

    Returns:
        若 output_path 为 None 返回音频数组，否则返回输出路径字符串
    """
    config = VoiceConfig(sample_rate=sample_rate)
    synth = VoiceSynthesizer(config)
    audio = synth.generate(text)
    if output_path:
        sf.write(output_path, audio, sample_rate)
        return output_path
    return audio


# =============================================================================
# 独立执行入口 (Standalone Execution Entry Point)
# =============================================================================

def main():
    import argparse
    import os
    import sys
    import tempfile

    parser = argparse.ArgumentParser(
        prog="rocky-synth",
        description="Synthesize Rocky (Eridian) alien voice from text.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""Examples:
  rocky-synth "你好 Rocky"
  rocky-synth "Hello" --sample-rate 44100 -o output.wav
        """,
    )
    parser.add_argument("text", nargs="*", help="Text to synthesize into voice")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output .wav file path (default: prints path to stdout)")
    parser.add_argument("--sample-rate", type=int, default=22050,
                        help="Audio sample rate in Hz (default: 22050)")

    args = parser.parse_args()

    if not args.text:
        parser.error("the following arguments are required: text")

    text = " ".join(args.text)

    config = VoiceConfig(sample_rate=args.sample_rate)
    synth = VoiceSynthesizer(config)

    if args.output:
        output_path = args.output
    else:
        safe_name = "".join(c if c.isalnum() else "_" for c in text[:20])
        output_path = os.path.join(tempfile.gettempdir(), f"rocky_{safe_name}.wav")

    try:
        result = synth.synthesize_file(text, output_path)
        if args.output:
            print(f"Saved: {result}")
        else:
            print(result)
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
