import re
import numpy as np

from typing import List

from sistema_distancias import SistemaDistancias
from sistema_contexto import SistemaContexto

class Autocorrector:
    def __init__(self):
        self._sistema_distancias = SistemaDistancias()
        self._sistema_contexto = SistemaContexto(min_apperance=10, size_ngram=2)
        self._vocabulario = None

    def fit(self, X_train: List[str]) -> None:
        """ Se entrena con una lista de frases """
        self._sistema_distancias.fit(X_train)
        self._sistema_contexto.fit(X_train)
        self._calc_vocabulary(X_train)

    def _calc_vocabulary(self, X_train: List[str]):
        self._vocabulario = set()
        for frase in X_train:
            for pal in frase.split():
                self._vocabulario.add(pal)

    def corregir(self, frases: List[str], modo: str = "combinado") -> List[str]:
        """ Corrige todas las frases que recibe.
            modo: 'solo_distancias' | 'solo_contexto' | 'combinado'
        """
        if modo not in {"solo_distancias", "solo_contexto", "combinado"}:
            raise ValueError("modo inválido. Use: 'solo_distancias', 'solo_contexto' o 'combinado'")

        frases_corregidas = []
        for frase in frases:
            frase_lower = frase.lower()
            frase_limpia = re.sub(r'[^a-záéíóúñ\s]', '', frase_lower)
            frase_split = frase_limpia.split()

            frase_corregida = []
            for i, palabra in enumerate(frase_split):
                if palabra in self._vocabulario:
                    frase_corregida.append(palabra)
                    continue

                palabras_candidatas, distancias = self._sistema_distancias.predict(palabra, 10)
                probs_palabras = self._sistema_contexto.predict(frase_split, i, palabras_candidatas)
                
                puntuaciones_palabras_candidatas = []
                min_distancia, max_distancia = np.min(distancias), np.max(distancias)
                min_prob, max_prob = np.min(probs_palabras), np.max(probs_palabras)

                for palabra, distancia, prob in zip(palabras_candidatas, distancias, probs_palabras):
                    puntuacion_distancia = 0
                    if min_distancia != max_distancia:
                        puntuacion_distancia = 1 - (distancia - min_distancia) / (max_distancia - min_distancia)
                    
                    puntuacion_probabilidad = 0
                    if min_prob != max_prob:
                        puntuacion_probabilidad = (prob - min_prob) / (max_prob - min_prob)
                    
                    if modo == "solo_distancias":
                        puntuacion_total = puntuacion_distancia
                    elif modo == "solo_contexto":
                        puntuacion_total = puntuacion_probabilidad
                    else:  # combinado
                        puntuacion_total = (puntuacion_distancia + puntuacion_probabilidad) / 2

                    puntuaciones_palabras_candidatas.append(puntuacion_total)

                puntuaciones_palabras_candidatas = np.array(puntuaciones_palabras_candidatas)
                pos = np.argmax(puntuaciones_palabras_candidatas)
                palabra_corregida = palabras_candidatas[pos]
                frase_corregida.append(palabra_corregida)
                
            frase_corregida = " ".join(frase_corregida)
            frases_corregidas.append(frase_corregida)

        return frases_corregidas