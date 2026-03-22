import re
import numpy as np

from typing import List, Literal

from sistema_distancias import SistemaDistancias
from sistema_contexto import SistemaContexto

class Autocorrector:
    def __init__(self, modo: str=Literal["distancias", "contexto", "ambos"], numero_ngramas: int=None, min_apperance: int=None):
        if not(modo in ["distancias", "contexto", "ambos"]):
            raise ValueError("Requisito: El modo solo puede ser \"distancias\", \"contexto\" o \"ambos\"")
        if (modo == "contexto" or modo == "ambos") and (numero_ngramas == None or min_apperance == None):
            raise ValueError("Requisito: Se debe especificar el número de ngramas")

        self._modo = modo
        self._numero_ngramas = numero_ngramas
        self._min_apperance = min_apperance

        self._sistema_distancias = None
        self._sistema_contexto = None
        self._vocabulario = None

    def fit(self, X_train: List[str]) -> None:
        """ Se entrena con una lista de frases """
        if self._modo == "distancias":
            self._sistema_distancias = SistemaDistancias()
            self._sistema_distancias.fit(X_train)
        elif self._modo == "contexto":
            self._sistema_distancias = SistemaDistancias(a=1.0, min_freq=1, intercambiar=True, usar_cercania_teclado=False)
            self._sistema_contexto = SistemaContexto(min_apperance=self._min_apperance, size_ngram=self._numero_ngramas)
            self._sistema_distancias.fit(X_train)
            self._sistema_contexto.fit(X_train)
        elif self._modo == "ambos":
            self._sistema_distancias = SistemaDistancias()
            self._sistema_contexto = SistemaContexto(min_apperance=self._min_apperance, size_ngram=self._numero_ngramas)
            self._sistema_distancias.fit(X_train)
            self._sistema_contexto.fit(X_train)
        
        self._calc_vocabulary(X_train)

    def _calc_vocabulary(self, X_train: List[str]):
        self._vocabulario = {}
        for frase in X_train:
            for pal in frase.split():
                if not(pal in self._vocabulario):
                    self._vocabulario[pal] = 1
                else:
                    self._vocabulario[pal] += 1

    def corregir(self, frase: str) -> str:
        """ Corrige la frase que recibe """
        frase_lower = " ".join(re.sub(r"[^a-záéíóúñ\s]", " ", frase.lower()).split())
        if self._modo == "distancias":
            frase_corregida = []
            frase_split = frase_lower.split()
            for palabra in frase_split:
                if palabra in self._vocabulario:
                    frase_corregida.append(palabra)
                    continue

                palabras_candidatas, distancias = self._sistema_distancias.predict(palabra=palabra, max_correciones=10, intercambiar=True)
                frase_corregida.append(palabras_candidatas[0]) # Esta es la palabra más cercana y frecuente (esta ordenado de dicha forma)
        elif self._modo == "contexto":
            frase_corregida = []
            frase_split = frase_lower.split()
            normalizar = lambda s: s.translate(str.maketrans("áéíóúü", "aeiouu"))
            for pos_pal, pal in enumerate(frase_split):
                if pal in self._vocabulario:
                    frase_corregida.append(pal)
                    continue

                palabras_distancia, _ = self._sistema_distancias.predict(palabra=pal, max_correciones=10, intercambiar=True, a=1.0, min_freq=1, solo_original_si_conocida=False)
                palabras_contexto, probabilidades = self._sistema_contexto.predict(frase_split, pos_pal, num_sugerencias=300_000) # 300_000 para que nos sugiera todas las que pueda

                palabras_elegidas = []
                for palabra_distancia in palabras_distancia:
                    for i, palabra_contexto in enumerate(palabras_contexto):
                        if palabra_distancia == normalizar(palabra_contexto):
                            palabras_elegidas.append((palabra_contexto, probabilidades[i]))
                            break
                palabras_elegidas = sorted(palabras_elegidas, key=lambda x: -x[1])

                if len(palabras_elegidas) == 0:
                    frase_corregida.append(pal)
                else:
                    frase_corregida.append(palabras_elegidas[0][0])
        elif self._modo == "ambos":
            frase_corregida = []
            frase_split = frase_lower.split()
            normalizar = lambda s: s.translate(str.maketrans("áéíóúü", "aeiouu"))
            for pos_pal, pal in enumerate(frase_split):
                if pal in self._vocabulario:
                    frase_corregida.append(pal)
                    continue

                palabras_distancia, distancias = self._sistema_distancias.predict(palabra=pal, max_correciones=10, intercambiar=True)
                palabras_contexto, probabilidades = self._sistema_contexto.predict(frase_split, pos_pal, num_sugerencias=300_000) # 300_000 para que nos sugiera todas las que pueda

                distancias = np.array(distancias)
                probabilidades = np.array(probabilidades)
                rango_distancias = np.max(distancias) - np.min(distancias)
                rango_probabilidades = np.max(probabilidades) - np.min(probabilidades)
                distancias = np.zeros_like(distancias) if rango_distancias == 0 else (distancias - np.min(distancias)) / rango_distancias
                probabilidades = np.zeros_like(probabilidades) if rango_probabilidades == 0 else (probabilidades - np.min(probabilidades)) / rango_probabilidades

                palabras_elegidas = []
                for i, palabra_distancia in enumerate(palabras_distancia):
                    for j, palabra_contexto in enumerate(palabras_contexto):
                        if palabra_distancia == normalizar(palabra_contexto):
                            palabras_elegidas.append((palabra_contexto, distancias[i], probabilidades[j]))
                            break
                
                if len(palabras_elegidas) == 0:
                    frase_corregida.append(pal)
                else:
                    puntuaciones = []
                    for i in range(len(palabras_elegidas)):
                        puntuacion_distancia = 1 - palabras_elegidas[i][1]
                        puntuacion_contexto = palabras_elegidas[i][2]
                        puntuaciones.append((palabras_elegidas[i][0], (puntuacion_distancia + puntuacion_contexto) / 2))
                    puntuaciones = sorted(puntuaciones, key=lambda x: -x[1])
                    frase_corregida.append(puntuaciones[0][0])

        frase_corregida = " ".join(frase_corregida)
        return frase_corregida
