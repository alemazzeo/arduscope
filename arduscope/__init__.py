from arduscope.arduscope import Arduscope, ArduscopeMeasure
from arduscope.fit_tools import sine, fit_signal, UnfittableDataError
try:
    from arduscope.wave_generator import WaveGenerator, WaveLoop
except ModuleNotFoundError:
    print("  NOTE: WaveGenerator requieres the extra package: simpleaudio\n"
          "      Use `pip install simpleaudio`."
          "      For Linux see simpleaudio dependecies at:\n"
          "      https://simpleaudio.readthedocs.io"
          "/en/latest/installation.html#linux-dependencies")
