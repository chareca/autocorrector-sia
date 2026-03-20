from generador_muestras import GeneradorMuestras
from autocorrector import Autocorrector
import random

if __name__ == '__main__':
    # Configuración
    ratio_error = 0.5
    ruta_directorio_libros = "./libros"
    cantidad_frases_test = 20
    test_ratio = 0.2
    seed = 42

    # Carga de datos
    generador = GeneradorMuestras(
        ruta_directorio_libros=ruta_directorio_libros,
        ratio_error=ratio_error
    )
    X_all = generador.get_corpus()

    random.seed(seed)
    X_all = X_all[:]
    random.shuffle(X_all)
    corte = int(len(X_all) * (1 - test_ratio))
    X_train = X_all[:corte]
    X_test = X_all[corte:]

    # Entrenamiento del autocorrector
    autocorrector = Autocorrector()
    autocorrector.fit(X_train)

    generador.corpus = X_test
    pares_eval = generador.generar_frases_con_errores(cantidad_frases_test)

    # Evaluación de resultados
    for modo in ["solo_distancias", "solo_contexto", "combinado"]:
        print(f"\n=== Evaluación: {modo} ===")
        generador.testear_modelo(
            autocorrector,
            cantidad=cantidad_frases_test,
            modo=modo,
            pares=pares_eval  # requiere parámetro opcional en testear_modelo
        )