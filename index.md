# Arduscopio

Arduscopio es una interfaz Arduino-Python desarrollada para la adquisición de datos analógicos. El comportamiento de la interfaz imita algunas características propias de un osciloscopio, tales como la funcionalidad de Trigger y el cambio de escala temporal.

Arduscopio surge como una iniciativa para facilitar la adquisición de datos en la materia Laboratorio 3 del Departamento de Física de la Facultad de Ciencias Exactas y Naturales de la UBA.

## Instalación

### Python

El paquete puede ser instalado utilizando [PIP](https://pypi.org/project/arduscope):

```
pip install arduscope
```

El paquete instala las siguientes dependencias en caso de no encontrarse presentes: 
  - Numpy
  - Matplotlib
  - PySerial

### Arduino

El código de Arduino puede ser descargado desde el repositorio del proyecto.

[Click derecho en este link para guardar -> Guardar como](https://raw.githubusercontent.com/alemazzeo/arduscope/main/arduscope/arduscope.ino)

Debe ser cargado en una Arduino UNO (excluyente).
**Ninguna fracción del código fue pensada para ser compatible con otra placa.**
Cualquier funcionamiento del código fuera de Arduino UNO es mera coincidencia.

## Ejemplo de uso

Los elementos centrales del paquete son el objeto `Arduscope` (la interfaz entre Python y el Arduino)
y el objeto `ArduscopeScreen` (un contenedor para los resultados adquiridos)

En primer lugar debemos importar estas clases del paquete que instalamos:

```python
from arduscope import Arduscope, ArduscopeScreen
```

Luego podemos crear una instancia del Arduscopio, indicando el puerto de conexión.
Arduino IDE por ejemplo nos muestra el nombre del puerto en la esquina inferior derecha.
En Windows los puertos suelen llamarse `"COM1"`, `"COM2"`, etc.
En Linux el formato suele ser `"/dev/ttyUSB0"` o parecido.

```python
arduino = Arduscope(port='/dev/ttyUSB0')
```

El objeto que instanciamos bajo el nombre arduino (podría ser cualquier otro) tiene
una serie de propiedades y métodos.

La lógica general es que las propiedades se configuren primero y luego se active la adquisición.
Si el usuario olvida detener la adquisición y cambia una propiedad la adquisición se detiene y reanuda automáticamente.

```python
arduino.frequency = 2000        # Frecuencia de adquisición (en Hz)
arduino.pulse_width = 0.05      # Ancho del pulso digital (en Segundos)
arduino.trigger_value = 2.5     # Valor del trigger (en Volts)
arduino.amplitude = 5.0         # Amplitud de la señal (en Volts)
arduino.n_channels = 2          # Cantidad de canales (1 a 6)
arduino.trigger_channel = "A0"  # Canal/Modo de trigger (ver apartado)
arduino.trigger_offset = 0.0    # Offset del trigger (en fracción de pantalla)
```

Para comenzar la adquisición se utiliza el método `start_acquire()`.

La misma puede ser detenida mediante `stop_acquire()`.

Estos métodos no detienen la ejecución principal del programa.

El método `live_plot()` abre un gráfico de Matplotlib y actualiza su contenido
para cada nueva pantalla. Requiere que el modo ventana de Matplotlib se encuentre activo.
Hasta que el gráfico no es cerrado la ejecución principal se detiene, generando un momento
para que el usuario vea los datos en tiempo real y decida si quiere continuar con su script o detenerlo.

```python
arduino.start_acquire()
arduino.live_plot()
```

### Ejemplo completo

```python
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
```
