from generador_muestras import GeneradorMuestras
from autocorrector import Autocorrector

if __name__ == '__main__':
    # Configuración
    ratio_error = 0.5
    ruta_directorio_libros = "autocorrector-sia/libros"
    cantidad_frases_test = 100

    # Carga de datos
    generador = GeneradorMuestras(
        ruta_directorio_libros=ruta_directorio_libros,
        ratio_error=ratio_error
    )
    X_train = generador.get_corpus()

    # Entrenamiento del autocorrector
    autocorrector = Autocorrector()
    autocorrector.fit(X_train)

    # Evaluación de resultados
    #generador.testear_modelo(autocorrector, cantidad=cantidad_frases_test)