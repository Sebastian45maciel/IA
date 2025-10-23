# problemas de los canibales y monjes    

En el acertijo de los misioneros y los caníbales, tres misioneros y tres caníbales tienen que cruzar un río con una barca que solo puede llevar como máximo dos personas,
lo cual es un constreñimiento para ambos bandos, porque si hay misioneros presentes en el barco, los caníbales se comerían a los misioneros. La barca no puede cruzar por el río sin personas a bordo. 
### 
| #  | Movimiento del Barco |  Orilla 1 (Restante) |  Orilla 2 (Destino) | Explicación                                              |
| -- | -------------------- | ---------------------- | --------------------- | ----------------------------------------------------------- |
| 1  | → 2 Caníbales        | 3M, 1C                 | 0M, 2C                | Ambas orillas seguras (3≥1 y 0M en la otra).                |
| 2  | ← 1 Caníbal          | 3M, 2C                 | 0M, 1C                | Un caníbal regresa.                                         |
| 3  | → 2 Caníbales        | 3M, 0C                 | 0M, 3C                | Dos caníbales cruzan. Todos los caníbales están en destino. |
| 4  | ← 1 Caníbal          | 3M, 1C                 | 0M, 2C                | Regresa un caníbal para ayudar a los misioneros.            |
| 5  | → 2 Misioneros       | 1M, 1C                 | 2M, 2C                |  Movimiento clave: ambas orillas seguras.                   |
| 6  | ← 1 M y 1 C          | 2M, 2C                 | 1M, 1C                | Regresan juntos manteniendo proporción segura.              |
| 7  | → 2 Misioneros       | 0M, 2C                 | 3M, 1C                | Dos misioneros cruzan, orilla 1 sin misioneros.             |
| 8  | ← 1 Caníbal          | 0M, 3C                 | 3M, 0C                | Regresa un caníbal por los últimos.                         |
| 9  | → 2 Caníbales        | 0M, 1C                 | 3M, 2C                | Dos caníbales cruzan. Orilla 2 segura.                      |
| 10 | ← 1 Caníbal          | 0M, 2C                 | 3M, 1C                | Un caníbal regresa.                                         |
| 11 | → 2 Caníbales        | 0M, 0C                 | 3M, 3C                |    ¡Éxito! Todos cruzaron.                                  |


# problema del puente y la antorcha 
Cuatro individuos llegan a un río de noche. Hay un puente estrecho, pero este solo soporta a dos personas a la vez. Los individuos tienen una antorcha y,
debido a que es de noche, deben utilizarla cuando cruzan el puente; por lo tanto, si cruzan dos personas, uno debe volver atrás llevando la antorcha para que puedan cruzar los demás.
El individuo A puede cruzar el puente en un minuto, el individuo B en dos minutos, el individuo C en cinco minutos, y el individuo D en ocho minutos. Cuando dos individuos cruzan el puente juntos, 
tardan lo que tarda el más lento de ellos.
## 
|  # | Movimiento (→ cruzan / ← regresa) | Estado (Izquierda — Derecha) | Tiempo del movimiento (min) | Tiempo acumulado (min) |  Explicación                                     |
| -: | :-------------------------------: | :--------------------------: | --------------------------: | ---------------------: | :------------------------------------------------- |
|  1 |              A y B →              |        (C, D) — (A, B)       |                           2 |                      2 | Cruzan los dos más rápidos; coste = 2.             |
|  2 |                A ←                |        (A, C, D) — (B)       |                           1 |                      3 | A regresa con la antorcha (el más rápido).         |
|  3 |              C y D →              |        (A) — (B, C, D)       |                           8 |                     11 | Los dos lentos cruzan juntos; coste = 8.           |
|  4 |                B ←                |        (A, B) — (C, D)       |                           2 |                     13 | B regresa con la antorcha (el segundo más rápido). |
|  5 |              A y B →              |     ( — ) — (A, B, C, D)     |                           2 |                     15 | A y B cruzan por última vez; coste = 2.            |

# 4 reynas en un tablero de 4 por 4

# cuadro magico
todas las diagonales me deben de dar 15
|  8 |  1  |  6  |
|  3 |  5  |  7  |
|  4 |  9  |  2  |


# come solo https://youtu.be/uqYLJJDrEcM?si=3oqceTPDLrRLmPJY
saltar y las canicas tiene que estar en medio

instalar python 
sklearn
keras
opencv

# problemas de dataset de la flor y de monito que salta y tu lo tienes que entrenar
# imagen 10 de septiembre 
plantear un posible analisis de datos en la imagen, patrones que debo buscar para ese juego, velocidad variable, pueden ser n bolita, por distancia publiana, es colision

#juego
variante de juego, ya esta en github. pero la variantes es 

# fotos y tarea
dia 22 de whatsapp y fotos de 24 
matriz de convulacion
para escalar una matriz 
para sacar la derivada de una imagen se le aplica la convolucion

# filtros de convolucion 25 sep
![Imagen de WhatsApp 2025-09-25 a las 10 23 44_33949bd6](https://github.com/user-attachments/assets/db85c7e7-7011-462e-9b04-c7200b2022f3)

foto de hoy, suavisa la imagen aqui no hay numeros negativos, va en el pivote o sea en el centro, se va cambiando con la original, todos los calculos van a ser el del centro, y para el nuevo calculo tambien es todo con lo original no con el nuevo

