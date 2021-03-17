# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import time
import calendar
import threading
import json
from functools import wraps
from dataclasses import dataclass, field, asdict
from collections import deque
from typing import List, Callable

import numpy as np
import matplotlib.pyplot as plt
import pathlib
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
class ArduscopeScreen:
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
                as_dict["x"] = self.x.tolist()
                as_dict["channels"] = [
                    channel.tolist()
                    for channel in self.channels
                ]
                json.dump(as_dict, f)
        elif filename.suffix == ".npz":
            np.savez(filename, **as_dict)
        elif filename.suffix == ".csv":
            as_dict["acquire_time"] = time.strftime(
                "%d/%m/%Y %H:%M:%S",
                time.gmtime(self.acquire_time)
            )
            header = "\n".join([
                f"{key} = {value}"
                for key, value in as_dict.items()
                if key not in ["x", "channels"]
            ])
            data = np.append([self.x], self.channels, axis=0).T
            np.savetxt(filename, data, fmt="%.9e", header=header)
        else:
            raise ValueError

    @classmethod
    def load(cls, file: [str, os.PathLike]) -> ArduscopeScreen:
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
                data["x"] = np.array(data["x"])
                data["channels"] = [
                    np.array(ch)
                    for ch in data["channels"]
                ]
                return cls(**data)
        elif filename.suffix == ".npz":
            data = np.load(filename)
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
            data = np.loadtxt(filename)
            as_dict["channels"] = data[:, 1:]
            return cls(**as_dict)
        else:
            raise ValueError


class Arduscope:
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

        self._screen_buffer = deque(maxlen=deque_max_size)

        self._daemon = None
        self._running = threading.Event()
        self._screen_ready = threading.Event()

        self._uptime = time.time()

        self._on_new_screen_function = None

        msg = ""
        while msg != "BOOTED\r\n":
            try:
                msg = self._serial.readline().decode('utf-8')
            except UnicodeDecodeError:
                pass
            if self.uptime > 5:
                raise TimeoutError("Arduino is not responding")

        self.frequency = 200
        self.pulse_width = 0.1
        self.trigger_value = 2.5
        self.amplitude = 5.0
        self.n_channels = 2
        self.trigger_channel = "A0"
        self.trigger_offset = 0.05
        self._trigger_tol = 5

    def _open_serial(self) -> Serial:
        """
        Opens a serial port between Arduino and Python

        Returns
        -------
        Serial (from PySerial library)
        """
        return Serial(port=self._port, baudrate=self._baudrate, timeout=1)

    @property
    def uptime(self) -> float:
        """ Uptime of Arduscope object creation"""
        return time.time() - self._uptime

    @property
    def x(self) -> np.ndarray:
        """ Time-array for x axes representation """
        return np.arange(BUFFER // self.n_channels) / self.frequency

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
        if 0.001 <= value <= MAX_PULSE_WIDTH / 1000.0:
            self._pulse_width = int(value * 1000)
        else:
            raise ValueError(f"MIN: 0.001, MAX: {MAX_PULSE_WIDTH / 1000.0}")
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
    def screens(self) -> deque:
        """ Screen buffer (a deque object)"""
        return self._screen_buffer

    @property
    def last_screen(self) -> ArduscopeScreen:
        """ Last screen in the buffer """
        if self._screen_ready.isSet():
            return self._screen_buffer[-1]
        else:
            raise RuntimeError("Screen not ready")

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
            "pulse_width": self._pulse_width
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
        self._screen_buffer.clear()
        self._daemon = threading.Thread(target=self._acquire_daemon, daemon=True)
        self._daemon.start()
        self._screen_ready.wait()

    def on_new_screen(self, function: Callable[[ArduscopeScreen], None]):
        """ Event handler for new screen available

        Parameters
        ----------
        function : Callable
            A function that receives a ArduscopeScreen and performs some action.
            Used by live_plot to update the graph.
        """
        @wraps(function)
        def wrapper(screen):
            function(screen)

        self._on_new_screen_function = wrapper

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
            if n_screens > self._screen_buffer.maxlen:
                raise ValueError(f"0 < n_screens < {self._screen_buffer.maxlen}")
        else:
            raise TypeError(f"0 < n_screens < {self._screen_buffer.maxlen}")

        if timeout is not None:
            if not isinstance(timeout, (int, float)):
                raise TypeError("Timeout type: float")

        start = time.time()
        while len(self._screen_buffer) < n_screens:
            if timeout is not None:
                if time.time() - start > timeout:
                    raise TimeoutError()

    def stop_acquire(self):
        """ Stops acquire without clearing the buffer """
        self._on_new_screen_function = None
        if self._running.isSet():
            self._running.clear()
        if self._daemon is not None:
            if self._daemon.is_alive():
                self._running.clear()
                self._daemon.join()

    def _on_property_change(self):
        """ Handles the properties changes resetting acquisition"""
        self._screen_buffer.clear()
        if self._running.isSet():
            self.stop_acquire()
            self.start_acquire()

    def _acquire_daemon(self):
        """ Background daemon that performs the buffer read """
        while self._running.isSet():
            if self._serial.inWaiting() >= BUFFER * 2 + 2:
                channels = self._read_buffer()
                screen = ArduscopeScreen(
                    acquire_time=time.time(),
                    frequency=self.frequency,
                    pulse_width=self.pulse_width,
                    trigger_value=self.trigger_value,
                    amplitude=self.amplitude,
                    n_channels=self.n_channels,
                    trigger_channel=self.trigger_channel,
                    trigger_offset=self.trigger_offset,
                    channels=channels
                )
                self._screen_buffer.append(screen)
                self._screen_ready.set()
                if self._on_new_screen_function is not None:
                    self._on_new_screen_function(self.last_screen)

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

    def live_plot(self):
        """ Deploy a Matplotlib window with the live state of Arduscope """
        if not self._running.isSet():
            raise RuntimeError('First call "start_acquire()"')

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
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

        def plot(screen: ArduscopeScreen):
            for i, channel in enumerate(screen.channels):
                curves[i].set_data(screen.x, channel)
            plt.draw()

        self.on_new_screen(plot)
        plt.show()
