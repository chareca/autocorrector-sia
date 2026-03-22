from typing import List, Tuple

from cargador_corpus import LectorCorpus
from autocorrector import Autocorrector

# Configuración
ruta_directorio_libros = "./libros"

def cargar_train_test():
    generador = LectorCorpus(
        ruta_directorio_libros=ruta_directorio_libros
    )
    X_train = generador.get_corpus()
    X_test = [
        ("me gusta ser el primero", "me gusa ser el primeo"), # Error de borrado
        ("Sabía que le gustaría", "Sbía que le gustaría"), # Error de borrado
        ("A mi hermano le gustan los helados", "A mi herrmano le gustan los helados"), # Error de inserción
        ("Los jugadores juegan mucho", "Los jugadrores juegan mucho"), # Error de inserción
        ("Odio hacer las frases de comprobación", "Odio hacer las frares de comprobación"), # Error de reemplazo (Poca distancia)
        ("Sin embargo, esta frase me gusta más", "Sin embargo, esta frase qe gusta más"), # Error de reemplazo (Mucha distancia)
        ("Ya queda menos para terminar", "Ya qeuda menos para terminar"), # Error de intercambio
        ("Espero que saquemos buena nota", "Espero que saquemos buean nota"), # Error de intercambio
        ("Algún día me iré de aquí", "Algún dua me iré de aquí"), # Las tildes
        ("Ojalá ganara el partido", "Ojala ganara el partido"), # Las tildes
        ("Tuvo un buen día", "Tubo un buen día"), # Error de contexto
        ("Voy a ver a mi amigo", "Voy a haber a mi amigo"), # Error de contexto
        ("Mi banco esta cerrado", "Mi banco esta serrado"), # Error de contexto
        ("Estoy echando el agua", "Estoy hechando el agua"), # Error de contexto
        ("Hay mucha gente aquí", "Ay mucha gente aquí"), # Error de contexto
        ("Vaya sorpresa me llevé", "Valla sorpresa me llevé"), # Error de contexto
        ("Vino tarde ayer", "Bino tarde ayer"), # Error de contexto (vino / bino)
        ("Has hecho bien el trabajo", "As hecho bien el trabajo") # Error de contexto (has / as)
    ]
    return X_train, X_test

def evaluar_autocorrector(nombre_autocorrector: str, autocorrector: Autocorrector, X_test: List[Tuple[str, str]]):
    print("=================================================")
    print(f"     Evaluación ({nombre_autocorrector})")
    print("=================================================\n")

    normalizar = lambda s: s.lower().translate(str.maketrans("áéíóúü", "aeiouu"))
    frases_ok = 0

    for frase_original, frase_con_errores in X_test:
        frase_corregida = autocorrector.corregir(frase_con_errores)
        original_norm = normalizar(frase_original)
        corregida_norm = normalizar(frase_corregida)
        acierto = corregida_norm == original_norm
        frases_ok += int(acierto)
        
        print("Frase original: ", frase_original.lower())
        print("Frase con errores: ", frase_con_errores.lower())
        print("Frase corregida: ", frase_corregida)
        print("Resultado: ", "OK" if acierto else "FAIL")
        print("\n")

    print(f"Accuracy por frase: {frases_ok / len(X_test):.2%}")
    print()

if __name__ == '__main__':
    X_train, X_test = cargar_train_test()

    autocorrector_distancias = Autocorrector(modo="distancias")
    autocorrector_distancias.fit(X_train)
    evaluar_autocorrector("AutocorrectorDistancias", autocorrector_distancias, X_test)

    autocorrector_contexto = Autocorrector(modo="contexto", numero_ngramas=2, min_apperance=11)
    autocorrector_contexto.fit(X_train)
    evaluar_autocorrector("AutocorrectorContexto", autocorrector_contexto, X_test)

    autocorrector_ambos = Autocorrector(modo="ambos", numero_ngramas=2, min_apperance=11)
    autocorrector_ambos.fit(X_train)
    evaluar_autocorrector("AutocorrectorDistanciasContexto", autocorrector_ambos, X_test)