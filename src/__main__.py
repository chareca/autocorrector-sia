from generador_muestras import GeneradorMuestras
from autocorrector import Autocorrector
import random

if __name__ == '__main__':
    # Configuración
    ratio_error = 0.5
    ruta_directorio_libros = "autocorrector-sia/libros"
    cantidad_frases_test = 20
    test_ratio = 0.2
    seed = 42

    # Carga de datos
    generador = GeneradorMuestras(
        ruta_directorio_libros=ruta_directorio_libros,
        ratio_error=ratio_error
    )
    X_train = generador.get_corpus()

    # =========================
    # DATASET MANUAL (FIJO)
    # =========================
    X_test = [
        ("el gato duerme en la csa", "el gato duerme en la casa"),
        ("la nina come pan con quso", "la niña come pan con queso"),
        ("el perro corre por el parqe", "el perro corre por el parque"),
        ("me guta leer libors coto", "me gusta leer libros cortos"),
        ("vamos a salri esta tarde", "vamos a salir esta tarde"),
        ("el sol brilia en el cielo", "el sol brilla en el cielo"),
        ("mi amigo viv en madrdi", "mi amigo vive en madrid"),
        ("el coche es muy raido", "el coche es muy rapido"),
        ("ella canta muy bein", "ella canta muy bien"),
        ("tengo una casa grnade", "tengo una casa grande"),
        ("el niño jueg con la pelota", "el niño juega con la pelota"),
        ("el libro es en la msa", "el libro esta en la mesa"),
        ("el tubo un buen dia", "el tuvo un buen dia"),
        ("voy a haber a mi amigo", "voy a ver a mi amigo"),
        ("el ino esta en la mesa", "el vino esta en la mesa"),
        ("el banco esta serrado", "el banco esta cerrado"),
        ("el camino es mu largo", "el camino es muy largo"),
        ("la yave abre la puerta", "la llave abre la puerta"),
        ("el barco llega al pueto", "el barco llega al puerto"),
        ("el save la respuesta", "el sabe la respuesta"),
    ]

    # Entrenamiento del autocorrector
    autocorrector = Autocorrector()
    autocorrector.fit(X_train)

    # Evaluación de resultados
    for modo in ["solo_distancias", "solo_contexto", "combinado"]:
        print(f"\n=== Evaluación: {modo} ===")
        generador.testear_modelo(
            autocorrector,
            cantidad=cantidad_frases_test,
            modo=modo,
            pares=X_test  # requiere parámetro opcional en testear_modelo
        )