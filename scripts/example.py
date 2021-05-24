import matplotlib.pyplot as plt
from arduscope import Arduscope
# from arduscope import ArduscopeMeasure

with Arduscope(port='/dev/ttyUSB0') as arduino:
    arduino.frequency = 1000
    arduino.pulse_width = 0.2
    arduino.trigger_value = 1.0
    arduino.amplitude = 5.0
    arduino.n_channels = 2
    arduino.trigger_channel = "D7OUT_HIGH"
    arduino.trigger_offset = 0.0

    arduino.start_acquire()
    arduino.live_plot()
    arduino.wait_until(n_screens=10, timeout=None)

measure = arduino.measure

ax: plt.Axes
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(measure.x, measure.channels[0].mean(axis=0), label='a0')
ax.plot(measure.x, measure.channels[1].mean(axis=0), label='a1')
ax.set_title(f"Trigger: {measure.trigger_value}V")

plt.show()

# measure.save("data.csv")
# measure.save("data.npz")
# measure.save("data.json")

# measure = ArduscopeMeasure.load("data.csv")

