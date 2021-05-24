# -*- coding: utf-8 -*-

from __future__ import annotations

import calendar
import inspect
import json
import os
import pathlib
import threading
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from functools import wraps
from tqdm import tqdm
from typing import List, Callable, Dict

import matplotlib.pyplot as plt
import numpy as np
from serial import Serial

BUFFER = 480
MAX_FREQ = 4000
MAX_PULSE_WIDTH = 32767
BAUDRATE = 115200

ARDUINO_PARAMS = [
    "limit",
    "frequency",
    "reference",
    "trigger",
    "trigger_channel",
    "trigger_offset",
    "trigger_tol",
    "channels",
    "adc_prescaler",
    "pulse_width",
]


@dataclass
class ArduscopeMeasure:
    acquire_time: float
    frequency: int
    pulse_width: float
    trigger_value: float
    amplitude: float
    n_channels: int
    trigger_channel: str
    trigger_offset: float

    x: np.ndarray = field(init=False)
    channels: List[np.ndarray] = list
    version: str = "0.1.2"

    def __post_init__(self):
        self.acquire_time = float(self.acquire_time)
        self.frequency = int(self.frequency)
        self.pulse_width = float(self.pulse_width)
        self.trigger_value = float(self.trigger_value)
        self.amplitude = float(self.amplitude)
        self.n_channels = int(self.n_channels)
        self.trigger_channel = str(self.trigger_channel)
        self.trigger_offset = float(self.trigger_offset)

        self.x = np.arange(BUFFER // self.n_channels) / self.frequency

    def save(self, file: [str, os.PathLike], overwrite: bool = False):
        """ Saves a screen into a file (csv, npz or json)

        Parameters
        ----------
        file : str or os.PathLike
            A filename with desired format in extension (.csv, .npz or .json)
        overwrite : bool
            Indicates if the file is overwrite on exists case
        """
        if isinstance(file, (str, os.PathLike)):
            filename = pathlib.Path(file).absolute()
        else:
            raise TypeError

        if overwrite is False and filename.exists():
            raise FileExistsError

        as_dict: dict = asdict(self)

        if filename.suffix == ".json":
            with open(filename, mode="w") as f:
                as_dict.pop("x")
                as_dict["channels"] = [
                    channel.tolist()
                    for channel in self.channels
                ]
                json.dump(as_dict, f)
        elif filename.suffix == ".npz":
            as_dict.pop("x")
            np.savez(filename, **as_dict)
        elif filename.suffix == ".csv":
            as_dict["acquire_time"] = time.strftime(
                "%d/%m/%Y %H:%M:%S",
                time.gmtime(self.acquire_time)
            )
            header = "\n".join([
                f"# {key} = {value}"
                for key, value in as_dict.items()
                if key not in ["x", "channels"]
            ])
            with open(filename, mode="w") as f:
                f.write(header)
                f.write("\n")
                for i in range(self.channels[0].shape[0]):
                    f.write(f"\n### Screen {i} of {self.channels[0].shape[0]}\n")
                    screen = [
                        channel[i, :]
                        for channel in self.channels
                    ]
                    data = np.append([self.x], screen, axis=0).T
                    np.savetxt(f, data, fmt="%.9e")
                    f.write("### End of screen\n")
        else:
            raise ValueError

    @classmethod
    def load(cls, file: [str, os.PathLike]) -> ArduscopeMeasure:
        """ Loads a screen from a file (csv, npz or json)

        Parameters
        ----------
        file : str or os.PathLike
            A filename with valid extension (.csv, .npz or .json)

        Returns
        -------
        ArduscopeScreen instance with loaded data
        """
        if isinstance(file, (str, os.PathLike)):
            filename = pathlib.Path(file).absolute()
        else:
            raise TypeError

        if not filename.exists():
            raise FileNotFoundError

        if filename.suffix == ".json":
            with open(filename, mode="r") as f:
                data = json.load(f)
                data["channels"] = [
                    np.array(ch)
                    for ch in data["channels"]
                ]
                return cls(**data)
        elif filename.suffix == ".npz":
            data = np.load(str(filename))
            return cls(**data)
        elif filename.suffix == ".csv":
            as_dict = {}
            with open(filename, mode="r") as f:
                line = f.readline().strip()
                while line.startswith("#"):
                    split = line.replace("#", "").split("=")
                    if len(split) == 2:
                        key, value = split[0].strip(), split[1].strip()
                    as_dict[key] = value
                    line = f.readline().strip()

            as_dict["acquire_time"] = calendar.timegm(
                time.strptime(as_dict["acquire_time"], "%d/%m/%Y %H:%M:%S")
            )
            data = np.loadtxt(str(filename))
            n = data.shape[0] // (BUFFER // int(as_dict["n_channels"]))
            as_dict["channels"] = [
                np.asarray(np.array_split(data[:, i+1], n))
                for i in range(int(as_dict["n_channels"]))
            ]
            return cls(**as_dict)
        else:
            raise ValueError


class Arduscope:
    _open_ports: Dict[str, Arduscope] = {}

    def __init__(self, port: str, deque_max_size: int = 100):
        """
        Parameters
        ----------
        port : str
            Connection port of Arduino, like "COM1" or "/dev/ttyUS0"
        deque_max_size : int
            Max size of screen buffer (a double-ended queue)
        """
        if not isinstance(port, str):
            raise TypeError

        self._port = port
        self._baudrate = BAUDRATE

        self._serial = self._open_serial()

        self._capture_parameters = None

        self._freq = None
        self._pulse_width = None
        self._amplitude = None
        self._ref = None
        self._n_channels = None
        self._trigger_value = None
        self._trigger_channel = None
        self._trigger_channel_code = None
        self._trigger_tol = None
        self._trigger_offset = None
        self._adc_prescaler = 4
        self._ref_values = {"5.0": 0, "1.1": 1}

        self._measure_params = None
        self._data_buffer = deque(maxlen=deque_max_size)

        self._daemon = None
        self._running = threading.Event()
        self._screen_ready = threading.Event()

        self._uptime = time.time()

        self.frequency = 200
        self.pulse_width = 0.1
        self.trigger_value = 2.5
        self.amplitude = 5.0
        self.n_channels = 2
        self.trigger_channel = "A0"
        self.trigger_offset = 0.05
        self._trigger_tol = 5

        self._live_mode_on = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _open_serial(self) -> Serial:
        """
        Opens a serial port between Arduino and Python

        Returns
        -------
        Serial (from PySerial library)
        """

        if self._port in Arduscope._open_ports.keys():
            print(f"Closing other Arduscope instances in port {self._port}...")
            other_arduscope = Arduscope._open_ports[self._port]
            try:
                other_arduscope.stop_acquire()
                other_arduscope._serial.close()
            except AttributeError:
                pass

        Arduscope._open_ports.update({
            self._port: self
        })

        serial = Serial(port=self._port, baudrate=self._baudrate, timeout=1)
        msg = ""
        start_time = time.time()
        while msg != "BOOTED\r\n":
            try:
                msg = serial.readline().decode('utf-8')
            except UnicodeDecodeError:
                pass
            if time.time() - start_time > 5:
                raise TimeoutError("Arduino is not responding")
        return serial

    @property
    def uptime(self) -> float:
        """ Uptime of Arduscope object creation"""
        return time.time() - self._uptime

    @property
    def x(self) -> np.ndarray:
        """ Time-array for x axes representation """
        return np.arange(BUFFER // self.n_channels) / self.frequency

    @property
    def channels(self) -> List[np.ndarray]:
        return [
            np.asarray([channels[i] for channels in self._data_buffer])
            for i in range(self._n_channels)
        ]

    @property
    def frequency(self) -> int:
        """ Frequency of sampling (in Hz) """
        return self._freq

    @frequency.setter
    def frequency(self, value: int):
        if 1 <= value <= MAX_FREQ:
            self._freq = int(value)
        else:
            raise ValueError(f"MIN: 1, MAX: {MAX_FREQ}")
        self._on_property_change()

    @property
    def pulse_width(self) -> float:
        """ Output pulse width in PIN7 (in seconds)  """
        return self._pulse_width * 0.001

    @pulse_width.setter
    def pulse_width(self, value: float):
        if 0.002 <= value <= MAX_PULSE_WIDTH / 1000.0:
            self._pulse_width = int(value * 1000)
        else:
            raise ValueError(f"MIN: 0.002, MAX: {MAX_PULSE_WIDTH / 1000.0}")
        self._on_property_change()

    @property
    def amplitude(self) -> float:
        """ Max amplitude measured (in Volts) """
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value: float):
        if 0 < value <= 1.1:
            self._ref = "1.1"
            self._amplitude = value
        elif 1.1 <= value <= 5.0:
            self._ref = "5.0"
            self._amplitude = value
        else:
            raise ValueError("0.0 < value <= 5.0")
        self._on_property_change()

    @property
    def trigger_value(self) -> float:
        """ Trigger value (in Volts) """
        return self._trigger_value

    @trigger_value.setter
    def trigger_value(self, value: float):
        if 0 < value < 5.0:
            self._trigger_value = value
        else:
            raise ValueError("MIN: 0, MAX: 5.0")
        self._on_property_change()

    @property
    def trigger_channel(self) -> str:
        """ Trigger channel

        Posible values:
          - "A0" to "A6" -> Analog inputs
          - "D7OUT_HIGH" -> When PIN7 changes to HIGH state
          - "D7OUT_LOW"  -> When PIN7 changes to LOW state
          - "REPEAT"     -> Immediately after transmitting the last measurement
        """
        return self._trigger_channel

    @trigger_channel.setter
    def trigger_channel(self, value: str):
        if isinstance(value, str):
            if value.upper().startswith("A") and len(value) == 2:
                self._trigger_channel_code = int(value[1])
            elif value.upper() == "D7OUT_HIGH":
                self._trigger_channel_code = -1
            elif value.upper() == "D7OUT_LOW":
                self._trigger_channel_code = -2
            elif value.upper() == "REPEAT":
                self._trigger_channel_code = -3
            else:
                raise ValueError("Posible values: "
                                 '"A0", "A1", "A2", "A3", "A4", "A5", "A6", '
                                 '"D7OUT_HIGH", "D7OUT_LOW", "REPEAT"')
        else:
            raise TypeError("Posible values: "
                            '"A0", "A1", "A2", "A3", "A4", "A5", "A6", '
                            '"D7OUT_HIGH", "D7OUT_LOW", "REPEAT"')
        self._trigger_channel = value
        self._on_property_change()

    @property
    def trigger_offset(self) -> float:
        """ Trigger offset in screen fraction (-1.0 to 1.0) """
        return self._trigger_offset

    @trigger_offset.setter
    def trigger_offset(self, value: float):
        if isinstance(value, (int, float)):
            if -1.0 <= value <= 1.0:
                self._trigger_offset = value
            else:
                raise ValueError("MIN: -1.0, MAX: 1.0")
        else:
            raise TypeError("Expected <float>, MIN: -1.0, MAX: 1.0")
        self._on_property_change()

    @property
    def n_channels(self) -> int:
        """ Number of channels (1 to 6 available)"""
        return self._n_channels

    @n_channels.setter
    def n_channels(self, value: int):
        if 1 <= value <= 6:
            self._n_channels = int(value)
        else:
            raise ValueError("MIN: 1, MAX: 6")
        self._on_property_change()

    @property
    def factor(self) -> float:
        """ Conversion factor given by Arduino Reference and bit depth (10 bits)"""
        if self._ref == "5.0":
            return 1024 / 5.0
        elif self._ref == "1.1":
            return 1024 / 1.1

    @property
    def measure(self) -> ArduscopeMeasure:
        """ An ArduscopeMeasure object with measurement params and channel data"""
        return ArduscopeMeasure(channels=self.channels, **self._measure_params)

    def start_acquire(self):
        """ Starts acquire in background (clearing previous state) """
        parameters = {
            "limit": 0,
            "frequency": self._freq,
            "reference": self._ref_values[self._ref],
            "trigger": self._trigger_value * self.factor,
            "trigger_channel": self._trigger_channel_code,
            "trigger_offset": int((BUFFER // self._n_channels) * self._trigger_offset),
            "trigger_tol": self._trigger_tol,
            "channels": self._n_channels,
            "adc_prescaler": self._adc_prescaler,
            "pulse_width": self._pulse_width // 2
        }

        if self._ref == "1.1":
            if self._trigger_value > 1.1 and self._trigger_channel_code >= 0:
                raise ValueError(f"Trigger value {self._trigger_value}V "
                                 f"greater than maximum amplitude {self._ref}V.")

        self._measure_params = {
            "acquire_time": time.time(),
            "frequency": self.frequency,
            "pulse_width": self.pulse_width,
            "trigger_value": self.trigger_value,
            "amplitude": self.amplitude,
            "n_channels": self.n_channels,
            "trigger_channel": self.trigger_channel,
            "trigger_offset": self.trigger_offset
        }

        typed_array = np.asarray(
            [parameters[x] for x in ARDUINO_PARAMS],
            dtype=np.int16
        )

        for param in typed_array:
            self._serial.write(int(param).to_bytes(2, byteorder="little", signed=True))

        if self._daemon is not None:
            if self._daemon.is_alive():
                self._running.clear()
                self._daemon.join()

        self._running.set()
        self._screen_ready.clear()
        self._data_buffer.clear()
        self._daemon = threading.Thread(target=self._acquire_daemon, daemon=True)
        self._daemon.start()
        self._screen_ready.wait()

    def clear_buffer(self):
        self._data_buffer.clear()

    def wait_signal(self):
        """ Stops execution until screen buffer has at least one measurement"""
        self._screen_ready.wait()

    def wait_until(self, n_screens: int, timeout: float = None):
        """ Stops execution until screen buffer has at least <n_screen>

        Parameters
        ----------
        n_screens : int
            Number of screens required
        timeout : float
            Timeout in seconds (raises a TimeoutError exception)

        """
        if isinstance(n_screens, int):
            if n_screens > self._data_buffer.maxlen:
                raise ValueError(f"0 < n_screens < {self._data_buffer.maxlen}")
        else:
            raise TypeError(f"0 < n_screens < {self._data_buffer.maxlen}")

        if timeout is not None:
            if not isinstance(timeout, (int, float)):
                raise TypeError("Timeout type: float")

        start = time.time()
        current_screens = len(self._data_buffer)
        if current_screens < n_screens:
            with tqdm(
                total=n_screens,
                miniters=1,
                initial=current_screens,
                ncols=80,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}"
            ) as pb:
                pb.set_description("Waiting for requested screens")
                while current_screens < n_screens:
                    if timeout is not None:
                        if time.time() - start > timeout:
                            raise TimeoutError()
                    pb.update(current_screens - pb.n)
                    current_screens = len(self._data_buffer)
                pb.update(n_screens - pb.n)

    def stop_acquire(self):
        """ Stops acquire without clearing the buffer """
        if self._running.isSet():
            self._running.clear()
        if self._daemon is not None:
            if self._daemon.is_alive():
                self._running.clear()
                self._daemon.join()

    def close(self):
        self.stop_acquire()
        self._serial.close()

    def _on_property_change(self):
        """ Handles the properties changes resetting acquisition"""
        self._data_buffer.clear()
        if self._running.isSet():
            self.stop_acquire()
            self.start_acquire()

    def _acquire_daemon(self):
        """ Background daemon that performs the buffer read """
        while self._running.isSet():
            if self._serial.inWaiting() >= BUFFER * 2 + 2:
                channels = self._read_buffer()
                self._data_buffer.append(channels)
                self._screen_ready.set()

    def _read_buffer(self) -> List[np.ndarray]:
        """ Private function for buffer reading and conversion """

        if self._serial.inWaiting() < BUFFER * 2 + 2:
            raise BufferError("Empty buffer")

        raw_start = self._serial.read(2)
        start = int.from_bytes(raw_start, byteorder="little", signed=True)

        raw_data = self._serial.read(BUFFER * 2)
        data = np.frombuffer(raw_data, dtype=np.uint16)
        data = data.reshape((BUFFER // self._n_channels, self._n_channels))
        channels = [
            np.roll(data[:, i], shift=-(start + 1) // self._n_channels) / self.factor
            for i in range(self._n_channels)
        ]

        return channels

    def simple_plot(self):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        curves = [
            ax.plot([], [], lw=2.0, label=f'Channel A{i}')[0]
            for i in range(self.n_channels)
        ]

        for i, channel in enumerate(self._data_buffer[-1]):
            curves[i].set_data(self.x, channel)

        ax.grid()
        ax.set_xlim(0, max(self.x))
        ax.set_ylim(0, self.amplitude)
        ax.set_xlabel("Time (s)", fontsize=14)
        ax.set_ylabel("Voltage (V)", fontsize=14)
        ax.legend(loc=1, fontsize=14)

    def live_plot(self):
        """ Deploy a Matplotlib window with the live state of Arduscope """
        if not self._running.isSet():
            raise RuntimeError('First call "start_acquire()"')

        backend = plt.get_backend()

        if 'inline' in backend:
            print(
                f"\nCurrent backend of Matplotlib is {plt.get_backend()}"
                f"\nLive mode not available for this backend"
            )
            self.simple_plot()
            return

        def on_close(event):
            self._live_mode_on = False

        interactive_state = plt.isinteractive()

        plt.ion()

        self._live_mode_on = True
        fig: plt.Figure
        ax: plt.Axes
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.canvas.mpl_connect('close_event', on_close)
        curves = [
            ax.plot([], [], lw=2.0, label=f'Channel A{i}')[0]
            for i in range(self.n_channels)
        ]

        ax.grid()
        ax.set_xlim(0, max(self.x))
        ax.set_ylim(0, self.amplitude)
        ax.set_xlabel("Time (s)", fontsize=14)
        ax.set_ylabel("Voltage (V)", fontsize=14)
        ax.legend(loc=1, fontsize=14)

        current_screens = len(self._data_buffer)

        with tqdm(
            total=self._data_buffer.maxlen,
            initial=current_screens,
            ncols=80,
            bar_format = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}"
        ) as pb:
            pb.set_description("Live mode on. Screen buffer status")
            while self._live_mode_on is True:
                plt.pause(0.001)
                if self._screen_ready.isSet():
                    for i, channel in enumerate(self._data_buffer[-1]):
                        curves[i].set_data(self.x, channel)
                    self._screen_ready.clear()
                pb.update(current_screens - pb.n)
                current_screens = len(self._data_buffer)

        if interactive_state is False:
            plt.ioff()
