from arduscope import Arduscope, ArduscopeScreen

arduino = Arduscope(port='/dev/ttyUSB0')

arduino.frequency = 2000
arduino.pulse_width = 0.05
arduino.trigger_value = 2.5
arduino.amplitude = 5.0
arduino.n_channels = 2
arduino.trigger_channel = "A0"
arduino.trigger_offset = 0.0

arduino.start_acquire()
arduino.live_plot()

screen = arduino.last_screen
x = screen.x
a0 = screen.channels[0]
a1 = screen.channels[1]

screen.save(file="prueba.csv", overwrite=True)

ArduscopeScreen.load(file="prueba.csv")
