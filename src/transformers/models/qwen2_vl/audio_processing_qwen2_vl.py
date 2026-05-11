# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Audio processor class for Qwen2-VL."""

import math
from typing import Optional, Union

import numpy as np

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType, logging


logger = logging.get_logger(__name__)

# Whisper encoder constants – must match the pre-trained weights.
_WHISPER_HOP_LENGTH = 160       # samples per mel frame at 16 kHz → 100 frames / s
_WHISPER_N_FFT = 400            # FFT window  (25 ms at 16 kHz)
_WHISPER_N_MELS = 128           # mel filter banks (whisper-large-v3-turbo uses 128)
_WHISPER_SAMPLING_RATE = 16_000
_WHISPER_MAX_FRAMES = 3_000     # hard cap: 30 s × 100 frames / s


def _num_audio_tokens(
    n_samples_at_16k: int,
    hop_length: int = _WHISPER_HOP_LENGTH,
    n_fft: int = _WHISPER_N_FFT,
    max_frames: int = _WHISPER_MAX_FRAMES,
) -> int:
    """
    Number of ``<|audio_pad|>`` tokens the Whisper encoder produces for an
    audio clip of *n_samples_at_16k* samples (already at 16 kHz).

    Whisper pipeline:
      raw samples
        → right-pad by n_fft // 2
        → STFT with hop_length                     → n_mel_frames
        → conv1 (kernel 3, stride 1, same length)
        → conv2 (kernel 3, stride 2)               → n_mel_frames // 2
    """
    n_mel_frames = (n_samples_at_16k + n_fft // 2) // hop_length + 1
    n_mel_frames = min(n_mel_frames, max_frames)
    return n_mel_frames // 2


class Qwen2VLAudioProcessor(SequenceFeatureExtractor):
    """
    Audio processor for Qwen2-VL.

    Converts raw waveforms to Whisper-compatible log-mel spectrograms and
    computes ``audio_lengths``: the number of ``<|audio_pad|>`` tokens each
    audio will occupy in the LLM token sequence.

    This mirrors ``Qwen2VLImageProcessor``: the main ``Qwen2VLProcessor``
    calls this class, receives ``input_features`` and ``audio_lengths``, then
    expands each single ``<|audio_pad|>`` placeholder in the tokenised text to
    ``audio_lengths[i]`` copies before feeding into the model.

    Args:
        sampling_rate (`int`, *optional*, defaults to 16000):
            Target sampling rate. Audio supplied at a different rate is
            resampled automatically (requires ``librosa``).
        n_fft (`int`, *optional*, defaults to 400):
            FFT window size (must match the Whisper weights used as encoder).
        hop_length (`int`, *optional*, defaults to 160):
            STFT hop / stride (must match the Whisper weights).
        n_mels (`int`, *optional*, defaults to 128):
            Number of mel filter banks (must match the Whisper weights).
        padding_value (`float`, *optional*, defaults to 0.0):
            Value used to right-pad ``input_features`` to the batch maximum
            length along the time axis.
    """

    model_input_names = ["input_features", "audio_lengths"]

    def __init__(
        self,
        sampling_rate: int = _WHISPER_SAMPLING_RATE,
        n_fft: int = _WHISPER_N_FFT,
        hop_length: int = _WHISPER_HOP_LENGTH,
        n_mels: int = _WHISPER_N_MELS,
        padding_value: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(
            feature_size=n_mels,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            **kwargs,
        )
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self._whisper_fe = None  # lazy-initialised in _compute_log_mel

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_number_of_audio_tokens(
        self, n_samples: int, sampling_rate: int = _WHISPER_SAMPLING_RATE
    ) -> int:
        """
        Return the number of ``<|audio_pad|>`` tokens for *n_samples* samples
        at *sampling_rate*.  Audio is conceptually resampled to 16 kHz first.

        Useful for pre-computing sequence lengths without running the full
        feature extraction pipeline.
        """
        if sampling_rate != self.sampling_rate:
            n_samples = math.ceil(n_samples * self.sampling_rate / sampling_rate)
        return _num_audio_tokens(n_samples, self.hop_length, self.n_fft)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_whisper_fe(self):
        """Lazy-init a WhisperFeatureExtractor with matching hyper-parameters."""
        if self._whisper_fe is None:
            from transformers import WhisperFeatureExtractor

            self._whisper_fe = WhisperFeatureExtractor(
                feature_size=self.n_mels,
                sampling_rate=self.sampling_rate,
                hop_length=self.hop_length,
                chunk_length=30,
                n_fft=self.n_fft,
                padding_value=self.padding_value,
            )
        return self._whisper_fe

    def _resample(self, array: np.ndarray, orig_sr: int) -> np.ndarray:
        if orig_sr == self.sampling_rate:
            return array
        try:
            import librosa
        except ImportError as exc:
            raise ImportError(
                "librosa is required for audio resampling. "
                "Install with: pip install librosa"
            ) from exc
        return librosa.resample(
            array.astype(np.float32), orig_sr=orig_sr, target_sr=self.sampling_rate
        )

    def _compute_log_mel(self, array: np.ndarray) -> np.ndarray:
        """
        Compute a Whisper-compatible log-mel spectrogram for a single waveform.

        ``padding=False`` lets the output length reflect the actual audio
        duration rather than always being padded to 3000 frames (30 s).

        Returns ``np.ndarray`` of shape ``[n_mels, T_actual]``.
        """
        fe = self._get_whisper_fe()
        out = fe(array, sampling_rate=self.sampling_rate, padding=False, return_tensors="np")
        return out["input_features"][0]  # [n_mels, T_actual]

    # ------------------------------------------------------------------
    # Public API  (mirrors Qwen2VLImageProcessor.preprocess)
    # ------------------------------------------------------------------

    def preprocess(
        self,
        audios,
        sampling_rate: Optional[Union[int, list[int]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> BatchFeature:
        """
        Process one or more audio inputs into model-ready tensors.

        Args:
            audios:
                A single audio or list of audios. Each entry may be a
                ``dict`` content element, a ``(np.ndarray, int)`` tuple from
                ``process_audio_info``, or a plain ``np.ndarray`` waveform.
            sampling_rate (`int` or `list[int]`, *optional*):
                Override the sampling rate when *audios* are plain arrays.
                A single int is broadcast to all entries.
            return_tensors (`str` or `TensorType`, *optional*):
                ``"pt"`` for PyTorch tensors, ``"np"`` for NumPy arrays.

        Returns:
            ``BatchFeature`` containing:

            - **input_features** – ``[B, n_mels, T_max]`` log-mel spectrograms,
              zero-padded along the time axis to the longest audio in the batch.
            - **audio_lengths** – ``[B]`` int64, the number of
              ``<|audio_pad|>`` tokens for each audio (based on un-padded
              duration; used by the processor to expand placeholder tokens and
              by the model to unpack the concatenated audio embeddings).
        """
        from qwen_vl_utils.vision_process import fetch_audio

        if not isinstance(audios, list):
            audios = [audios]

        if isinstance(sampling_rate, int) or sampling_rate is None:
            sampling_rates = [sampling_rate] * len(audios)
        else:
            if len(sampling_rate) != len(audios):
                raise ValueError(
                    f"sampling_rate list length ({len(sampling_rate)}) must match "
                    f"the number of audios ({len(audios)})."
                )
            sampling_rates = sampling_rate

        log_mels: list[np.ndarray] = []
        audio_lengths: list[int] = []

        for audio, sr in zip(audios, sampling_rates):
            sr_default = sr if sr is not None else self.sampling_rate
            array, orig_sr = fetch_audio(audio, sampling_rate=sr_default)
            array = self._resample(array, orig_sr)
            log_mel = self._compute_log_mel(array)  # [n_mels, T_i]
            # Derive token count from actual mel frames so it matches encoder output exactly.
            audio_lengths.append(log_mel.shape[-1] // 2)
            log_mels.append(log_mel)

        # Pad all log-mels to T_max along the time axis.
        t_max = max(lm.shape[-1] for lm in log_mels)
        input_features = np.stack(
            [
                np.pad(
                    lm,
                    ((0, 0), (0, t_max - lm.shape[-1])),
                    mode="constant",
                    constant_values=self.padding_value,
                )
                for lm in log_mels
            ],
            axis=0,
        )  # [B, n_mels, T_max]

        data = {
            "input_features": input_features,
            "audio_lengths": np.array(audio_lengths, dtype=np.int64),
        }
        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["Qwen2VLAudioProcessor"]
