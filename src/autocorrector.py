import re
import numpy as np

from typing import List, Literal

from sistema_distancias import SistemaDistancias
from sistema_contexto import SistemaContexto

class Autocorrector:
    def __init__(self, modo: str=Literal["distancias", "contexto", "ambos"], numero_ngramas: int=None):
        if not(modo in ["distancias", "contexto", "ambos"]):
            raise ValueError("Requisito: El modo solo puede ser \"distancias\", \"contexto\" o \"ambos\"")
        if (modo == "contexto" or modo == "ambos") and numero_ngramas == None:
            raise ValueError("Requisito: Se debe especificar el número de ngramas")

        self._modo = modo
        self._numero_ngramas = numero_ngramas

        self._sistema_distancias = None
        self._sistema_contexto = None
        self._vocabulario = None

    def fit(self, X_train: List[str]) -> None:
        """ Se entrena con una lista de frases """
        if self._modo == "distancias":
            self._sistema_distancias = SistemaDistancias()
            self._sistema_distancias.fit(X_train)
        elif self._modo == "contexto":
            self._sistema_contexto = SistemaContexto(min_apperance=10, size_ngram=self._numero_ngramas)
            self._sistema_contexto.fit(X_train)
        elif self._modo == "ambos":
            self._sistema_distancias = SistemaDistancias()
            self._sistema_contexto = SistemaContexto(min_apperance=10, size_ngram=self._numero_ngramas)
            self._sistema_distancias.fit(X_train)
            self._sistema_contexto.fit(X_train)
        
        self._calc_vocabulary(X_train)

    def _calc_vocabulary(self, X_train: List[str]):
        self._vocabulario = set()
        for frase in X_train:
            for pal in frase.split():
                self._vocabulario.add(pal)

    def corregir(self, frase: str) -> str:
        """ Corrige la frase que recibe """
        frase_lower = frase.lower()
        if self._modo == "distancias":
            frase_corregida = []
            frase_split = frase_lower.split()
            for palabra in frase_split:
                palabras_candidatas, distancias = self._sistema_distancias.predict(palabra=palabra, max_correciones=10, intercambiar=True)
                frase_corregida.append(palabras_candidatas[0]) # Esta es la palabra más cercana y frecuente (esta ordenado de dicha forma)
        elif self._modo == "contexto":
            frase_corregida = []
            frase_split = frase_lower.split()
            for pos_pal, pal in enumerate(frase_split):
                palabras_candidatas, probabilidades = self._sistema_contexto.predict(frase_split, pos_pal, num_sugerencias=10)
                frase_corregida.append(palabras_candidatas[0]) # Esta es la palabra más probable por el contexto (esta ordenado de dicha forma)
        elif self._modo == "ambos":
            frase_corregida = []
            frase_split = frase_lower.split()
            for pos_pal, palabra in enumerate(frase_split):
                palabras_candidatas, probabilidades = self._sistema_contexto.predict(frase_split, pos_pal, num_sugerencias=30)
                
                distancias = []
                for palabra_candidata in palabras_candidatas:
                    distancia = self._sistema_distancias._distancia_levenshtein_ponderada(palabra, palabra_candidata)
                    distancias.append(distancia)

                palabras_filtradas = []
                probabilidades_filtradas = []
                distancias_filtradas = []
                for palabra_candidata, prob, dist in zip(palabras_candidatas, probabilidades, distancias):
                    if dist <= 2:
                        palabras_filtradas.append(palabra_candidata)
                        probabilidades_filtradas.append(prob)
                        distancias_filtradas.append(dist)
                palabras_candidatas = palabras_filtradas
                probabilidades = probabilidades_filtradas
                distancias = distancias_filtradas

                if len(palabras_candidatas) == 0:
                    frase_corregida.append(palabra)
                else:
                    probs_array = np.array(probabilidades)
                    mejor_pos = np.argmax(probs_array)
                    palabra_corregida = palabras_candidatas[mejor_pos]
                    frase_corregida.append(palabra_corregida)

        frase_corregida = " ".join(frase_corregida)
        return frase_corregida