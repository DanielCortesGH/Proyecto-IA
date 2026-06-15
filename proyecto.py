import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# ==========================================
# 1. PREPARACIÓN DE LOS DATOS (Input / Output)
# ==========================================
descripciones = [
    "tiene cuatro patas, ladra y es el mejor amigo del hombre",
    "es pequeño, maúlla y le gusta cazar ratones",
    "tiene una trompa muy larga, es gris y muy grande",
    "tiene cuello largo, manchas y come hojas altas",
    "es verde, salta mucho y come moscas",
    "nada en el mar, tiene aletas y es muy inteligente",
    "vuela, tiene plumas y repite lo que dices",
    "es el rey de la selva y tiene una gran melena"
]

animales = [
    "perro",
    "gato",
    "elefante",
    "jirafa",
    "rana",
    "delfin",
    "loro",
    "leon"
]

# Convertimos las palabras (animales) a números para que la red los entienda
animales_unicos = list(set(animales))
diccionario_animales = {animal: i for i, animal in enumerate(animales_unicos)}
etiquetas = np.array([diccionario_animales[animal] for animal in animales])

# ==========================================
# 2. PROCESAMIENTO DE LENGUAJE NATURAL (NLP)
# ==========================================
vocab_size = 100
embedding_dim = 16
max_length = 15
trunc_type = 'post'
oov_tok = "<OOV>" 

# Inicializamos el tokenizador y lo ajustamos a nuestras descripciones
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(descripciones)

# Convertimos el texto a secuencias numéricas y aplicamos padding (relleno)
secuencias = tokenizer.texts_to_sequences(descripciones)
secuencias_pad = pad_sequences(secuencias, maxlen=max_length, truncating=trunc_type)

# ==========================================
# 3. CREACIÓN Y ENTRENAMIENTO DEL MODELO
# ==========================================
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(len(animales_unicos), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Entrenando la red neuronal... (esto tomará solo unos segundos)")
# Entrenamos el modelo con los datos
model.fit(secuencias_pad, etiquetas, epochs=300, verbose=0)
print("¡Modelo entrenado exitosamente!\n")

# ==========================================
# 4. LÓGICA DE PREDICCIÓN Y BUCLE DE CHAT
# ==========================================
def predecir_animal(texto_usuario):
    # Procesar la entrada del usuario exactamente igual que en el entrenamiento
    secuencia_usuario = tokenizer.texts_to_sequences([texto_usuario])
    secuencia_pad_usuario = pad_sequences(secuencia_usuario, maxlen=max_length, truncating=trunc_type)
    
    # Hacer la predicción
    prediccion = model.predict(secuencia_pad_usuario, verbose=0)
    
    # Obtener el índice con mayor probabilidad y su valor
    indice_ganador = np.argmax(prediccion)
    probabilidad = np.max(prediccion)
    
    # Traducir el número de vuelta al nombre del animal
    for animal, indice in diccionario_animales.items():
        if indice == indice_ganador:
            return animal, probabilidad

# Interfaz en la consola
print("="*50)
print("¡Hola! Soy el Chatbot de Animales impulsado por Deep Learning.")
print("Dime características de un animal e intentaré adivinar cuál es.")
print("Escribe 'salir' para terminar el chat.")
print("="*50 + "\n")

while True:
    entrada = input("Tú: ")
    
    if entrada.lower() == 'salir':
        print("Chatbot: ¡Hasta luego! Fue divertido.")
        break
        
    # Llamamos a nuestra función de predicción
    animal_predicho, confianza = predecir_animal(entrada)
    
    # Umbral de confianza para responder
    if confianza > 0.4:
        print(f"Chatbot: Mmmm... ¡Creo que estás hablando de un {animal_predicho.upper()}! (Confianza: {confianza*100:.1f}%)")
    else:
        print("Chatbot: Vaya, no estoy muy seguro con esas palabras... ¿Podrías darme más detalles?")