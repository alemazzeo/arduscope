from __future__ import annotations

import numpy as np
import multiprocessing
import scipy.signal
from functools import partial
from typing import Callable

try:
    import simpleaudio as sa
except ModuleNotFoundError:
    raise ModuleNotFoundError("Please install simpleaudio to use this module.\n"
                              "pip install simpleaudio\n\n"
                              "For Linux see simpleaudio dependecies at:\n"
                              "https://simpleaudio.readthedocs.io"
                              "/en/latest/installation.html#linux-dependencies")


MAX_FREQ = 2000

SignalFunction = Callable[[np.ndarray], np.ndarray]


def sine(x: np.ndarray, freq: float, phase: float) -> np.ndarray:
    return np.sin(x * 2 * np.pi * freq + phase)


def triangular(x: np.ndarray, freq: float, phase: float) -> np.ndarray:
    return scipy.signal.sawtooth(x * 2 * np.pi * freq + phase, width=0.5)


def square(x: np.ndarray, freq: float, phase: float) -> np.ndarray:
    return scipy.signal.square(x * 2 * np.pi * freq + phase, duty=0.5)


def make_signal(f: SignalFunction, duration: float, fs: int = 44100):
    step = 1/fs
    n = duration // step
    t = np.arange(n) * step
    return f(t)


class WaveGeneratorChannel:
    def __init__(self, fs: int = 44100, duration: int = 60):
        self._duration = duration
        self._fs = fs
        self._freq = 440
        self._waveform = "sine"
        self._phase = 0
        self._amplitude = 1.0
        self._enabled_output = False
        self._enabled_controls = True
        self._waveforms = {
            "sine": sine,
            "triangular": triangular,
            "square": square
        }

    @property
    def enabled(self) -> bool:
        return self._enabled_output

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled_output = value

    @property
    def frequency(self) -> int:
        return self._freq

    @frequency.setter
    def frequency(self, value: int):
        if self._enabled_controls is False:
            raise RuntimeError("Controls of WaveGenerator are blocked during loop")
        if 1 <= value <= MAX_FREQ:
            self._freq = int(value)
        else:
            raise ValueError(f"MIN: 1, MAX: {MAX_FREQ}")

    @property
    def phase(self) -> float:
        return self._phase

    @phase.setter
    def phase(self, value: float):
        if self._enabled_controls is False:
            raise RuntimeError("Controls of WaveGenerator are blocked during loop")

        self._phase = (value - np.pi) % (2 * np.pi) - np.pi

    @property
    def amplitude(self) -> float:
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value: float):
        if self._enabled_controls is False:
            raise RuntimeError("Controls of WaveGenerator are blocked during loop")
        if 0.0 <= value <= 1.0:
            self._amplitude = float(value)
        else:
            raise ValueError(f"MIN: 0.0, MAX: 1.0")

    @property
    def waveform(self) -> str:
        return self._waveform

    @waveform.setter
    def waveform(self, value: str):
        if self._enabled_controls is False:
            raise RuntimeError("Controls of WaveGenerator are blocked during loop")
        if value.lower() not in self._waveforms.keys():
            raise ValueError(f"{value.lower()} is not a available waveform.\n"
                             f"Posible values are: {sorted(self._waveforms.keys())}")
        self._waveform = value.lower()

    def get_signal(self) -> np.ndarray:
        fun = partial(
            self._waveforms[self._waveform],
            freq=self._freq,
            phase=self._phase
        )
        signal = make_signal(fun, duration=self._duration, fs=self._fs)
        signal = signal * self._amplitude * (2**15 - 1)
        return signal


class WaveGenerator:
    def __init__(self, fs: int = 44100, duration: int = 60):
        self._duration = duration
        self._fs = fs
        self._channel1 = WaveGeneratorChannel(fs=fs, duration=duration)
        self._channel2 = WaveGeneratorChannel(fs=fs, duration=duration)
        self._wave_loop = None

    @property
    def channel1(self) -> WaveGeneratorChannel:
        return self._channel1

    @property
    def channel2(self) -> WaveGeneratorChannel:
        return self._channel2

    def get_channels_signal(self) -> np.ndarray:
        signal = np.array([
            channel.get_signal()
            for channel in [self._channel1, self._channel2]
            if channel.enabled is True
        ])
        return signal

    def play_loop(self) -> WaveLoop:
        self._wave_loop = WaveLoop(
            signal=self.get_channels_signal(),
            normalize=False,
            fs=self._fs
        )
        return self._wave_loop

    def play_once(self):
        self._wave_loop = WaveLoop(
            signal=self.get_channels_signal(),
            normalize=False,
            fs=self._fs
        )
        self._wave_loop.play_once()


class WaveLoop:
    def __init__(self, signal: np.ndarray, normalize: bool = False, fs: int = 44100):
        self._fs = fs
        self._play = multiprocessing.Event()
        self._force_stop = multiprocessing.Event()
        self._repeat = multiprocessing.Event()
        self._play.clear()
        self._wave_object = self._generate(signal, normalize)
        self._play_object = None
        self._process = multiprocessing.Process(target=self._loop, daemon=True)

    def is_playing(self) -> bool:
        return self._play_object.is_playing()

    def _generate(self, signal: np.ndarray, normalize: bool = True) -> sa.WaveObject:

        signal = np.squeeze(signal)
        if signal.ndim == 1:
            channels = [signal]
        elif signal.ndim == 2:
            if signal.shape[0] == 2:
                channels = [signal[0, :], signal[1, :]]
            else:
                raise ValueError(f"Unexpected shape of signal: "
                                 f"signal.shape={signal.shape}.\n"
                                 f"Posible values are: \n"
                                 f"  (n)    -> single channel\n"
                                 f"  (1, n) -> single channel\n"
                                 f"  (2, n) -> double channel\n")
        else:
            raise ValueError(f"Unexpected dimension of signal: "
                             f"signal.ndim={signal.ndim}.\n"
                             f"Posible values are: 1 or 2")

        if normalize is True:
            # Ensure that highest value is in 16-bit range
            channels = [
                ch * (2**15 - 1) / np.max(np.abs(ch))
                for ch in channels
            ]
        else:
            if np.any(signal < -(2**15 - 1)) or np.any(signal > (2**15 - 1)):
                raise ValueError(f"Signal type is a 16-bit signed integer. "
                                 f"Values under -32767 or upper 32767 are not allowed")

        audio = np.ascontiguousarray(np.asarray(channels).T)
        audio = audio.astype(np.int16)

        wave_object = sa.WaveObject(
            audio_data=audio,
            num_channels=len(channels),
            bytes_per_sample=2,
            sample_rate=self._fs
        )

        return wave_object

    def _loop(self):
        while self._play.is_set():
            self._play_object = self._wave_object.play()
            while self._play_object.is_playing() and not self._force_stop.is_set():
                pass

    def play_once(self):
        try:
            self._play_object = self._wave_object.play()
            self._play_object.wait_done()
        finally:
            self._play_object.stop()

    def __enter__(self) -> WaveLoop:
        self._play.set()
        self._force_stop.clear()
        self._process.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._play.clear()
        self._force_stop.set()
        self._process.join()
