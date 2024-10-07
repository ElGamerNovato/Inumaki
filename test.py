import cv2
import speech_recognition as sr
import pyttsx3
import random
import threading
from datetime import datetime

# Inicialización del reconocimiento de voz y motor de texto a voz
r = sr.Recognizer()
engine = pyttsx3.init()

# Lista de apodos posibles
apodos = ["Héroe", "Explorador", "Ninja", "Mago", "Guerrero", "Viajero", "Sabio", "Cazador"]

# Almacenar las transcripciones con el tiempo
transcripciones = []
sesion_activa = True  # Controlar la ejecución del ciclo

# Asignar un apodo aleatorio
def asignar_apodo():
    return random.choice(apodos)

# Función para procesar el audio en tiempo real
def escuchar_audio_en_tiempo_real():
    global sesion_activa
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)  # Ajustar al ruido de fondo
        print("Escuchando en tiempo real...")

        while sesion_activa:
            try:
                # Captura en fragmentos pequeños de 1 a 2 segundos
                audio = r.listen(source, timeout=1, phrase_time_limit=2)
                
                # Obtener el tiempo actual para marcar la transcripción
                hora_actual = datetime.now().strftime('%H:%M:%S')
                
                # Transcribir el audio
                texto = r.recognize_google(audio, language='es-ES')
                print(f"[{hora_actual}] Transcripción en tiempo real: {texto}")
                
                # Guardar la transcripción con el tiempo en la lista
                transcripciones.append(f"[{hora_actual}] {texto}")
                
                # Comprobar si la sesión debe terminar
                if "fin de la sesión" in texto.lower():
                    print("Fin de la sesión detectado. Cerrando...")
                    sesion_activa = False  # Detener el ciclo
                    break  # Sale del bucle si escucha la frase de cierre

            except sr.UnknownValueError:
                pass  # Continuar escuchando si no se entiende el audio
            except sr.WaitTimeoutError:
                pass  # Permitir silencio en el audio sin detener el programa
            except sr.RequestError as e:
                print(f"Error en el servicio de reconocimiento de voz: {e}")
                sesion_activa = False  # Detener en caso de error grave
                break

# Guardar las transcripciones en un archivo de texto estilo subtítulos
def guardar_transcripciones():
    try:
        with open("transcripcion_sesion.txt", "w", encoding="utf-8") as f:
            for linea in transcripciones:
                f.write(linea + "\n")
        print("Transcripción guardada en 'transcripcion_sesion.txt'.")
    except Exception as e:
        print(f"Error al guardar el archivo: {e}")

# Inicialización de la cámara
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Variable para controlar si se ha detectado una persona
persona_detectada = False
apodo_asignado = None

# Función para manejar la cámara y detección de personas
def procesar_camara():
    global persona_detectada, apodo_asignado, sesion_activa
    
    while sesion_activa:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rostros = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(rostros) > 0 and not persona_detectada:
            apodo_asignado = asignar_apodo()
            print(f"Persona detectada. Apodo asignado: {apodo_asignado}")
            engine.say(f"Hola {apodo_asignado}")
            engine.runAndWait()
            persona_detectada = True

        for (x, y, w, h) in rostros:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('Cámara', frame)

        # Salir si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            sesion_activa = False  # Detener sesión manualmente
            break

# Ejecutar la transcripción en tiempo real en un hilo separado
audio_thread = threading.Thread(target=escuchar_audio_en_tiempo_real)
audio_thread.start()

# Ejecutar el procesamiento de la cámara
procesar_camara()

# Esperar a que el hilo del audio termine
audio_thread.join()

# Guardar la transcripción al finalizar
guardar_transcripciones()

# Cerrar todo cuando finalice
cap.release()
cv2.destroyAllWindows()

