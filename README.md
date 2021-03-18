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

[Click derecho en este link -> Guardar como](https://raw.githubusercontent.com/alemazzeo/arduscope/main/arduscope/arduscope.ino)

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

Arduino IDE nos muestra el nombre del puerto en la esquina inferior derecha.

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

A continuación podemos trabajar con los datos adquiridos.

En primer lugar conviene asegurar que fueron adquiridas la cantidad de pantallas que necesitamos.
Para eso podemos usar la función `wait_until(n_screens, timeout)`.

Esta función detiene la ejecución principal hasta que el buffer de pantallas tenga `n_screens` almacenadas.
También podemos pasarle un parámetro opcional de `timeout` para que se produzca un error por tiempo límite (en segundos).

Si la cantidad de pantallas ya fue alcanzada al llamar esta función el resultado será inmediato.
También podriamos detener la adquisición para que el buffer deje de sobreescribirse.

```python
arduino.wait_until(n_screens=50, timeout=None)
```

Podemos acceder al buffer mediante la propiedad `screens`.
Este buffer es un objeto de tipo `deque` pero a efectos de lectura podemos tratarlo como una lista común de Python.

Cada elemento en el buffer es un objeto de tipo `ArduscopeScreen`, nuestro contenedor de resultados.
Si queremos trabajar con la última pantalla podemos acceder a ella utilizando un índice negativo:

```python
screen = arduino.screens[-1]
```

Este objeto contiene todas las propiedades del Arduscopio al momento de la adquisición.
Podemos consultar la frecuencia o el valor del trigger (por ejemplo).

También tiene un vector `x` generado a partir de la frecuencia que corresponde al eje temporal.
Los canales quedan almacenados en una lista dentro de la propiedad `channels`.

Podriamos graficar del siguiente modo:

```python
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(screen.x, screen.channels[0], label='a0')
ax.plot(screen.x, screen.channels[1], label='a1')
ax.set_title(f"Trigger: {screen.trigger_value}")
plt.show()
```

Finalmente podríamos querer almacenar una medicin en un archivo.
El objeto screen provee un metodo `save` para facilitar esta tarea.
El formato se decide en base a la extensión del archivo creado:

```python
screen.save("data.csv")   # Formato CSV (separado por comas)
screen.save("data.npz")   # Formato NPZ (array comprimido de Numpy)
screen.save("data.json")  # Formato JSON (objeto de JavaScript)
```

Para recuperar una pantalla guardada vamos a crear un nuevo objeto `ArduscopeScreen` del siguiente modo:

```python
screen = ArduscopeScreen.load("data.csv")
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

arduino.wait_until(n_screens=50, timeout=None)

screen = arduino.screens[-1]

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(screen.x, screen.channels[0], label='a0')
ax.plot(screen.x, screen.channels[1], label='a1')
ax.set_title(f"Trigger: {screen.trigger_value}")
plt.show()

screen.save("data.csv")   # Formato CSV (separado por comas)
screen.save("data.npz")   # Formato NPZ (array comprimido de Numpy)
screen.save("data.json")  # Formato JSON (objeto de JavaScript)

# screen = ArduscopeScreen.load("data.csv")

```