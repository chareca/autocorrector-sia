import re
from typing import List, Tuple

import numpy as np


class SistemaDistancias:
    def __init__(self,a: float = 2.0,min_freq: int = 3,top_k: int = 10,intercambiar: bool = True,costo_intercambio: float = 0.5,):
        if a <= 0:
            raise ValueError("a debe ser > 0")
        if min_freq < 1:
            raise ValueError("min_freq debe ser >= 1")
        if top_k < 1:
            raise ValueError("top_k debe ser >= 1")
        if costo_intercambio < 0:
            raise ValueError("costo_intercambio debe ser >= 0")

        self._a = a
        self._min_freq = min_freq
        self._top_k = top_k
        self._intercambiar = intercambiar
        self._costo_intercambio = costo_intercambio

        self._alphabet = "abcdefghijklmnñopqrstuvwxyz"
        self._words = {}
        self._teclado_pos = self._mapa_teclado_es()

    def fit(self, X_train: List[str]):
        #Crea las frecuencias de las palabras
        self._words = {}
        for frase in X_train:
            linea = self._normalizar_frase(frase)
            for word in linea.split():
                self._words[word] = self._words.get(word, 0) + 1

    def predict(self,palabra: str,max_correciones: int | None = None,intercambiar: bool | None = None,a: float | None = None,min_freq: int | None = None,) -> Tuple[List[str], List[float]]:

        palabra = self._normalizar_palabra(palabra)
        if not palabra:
            return [], []

        if max_correciones is None:
            max_correciones = self._top_k
        if max_correciones <= 0:
            return [], []

        if intercambiar is None:
            intercambiar = self._intercambiar
        if a is None:
            a = self._a
        if min_freq is None:
            min_freq = self._min_freq

        ranking = self._sugerencias(palabra,top_k=max_correciones,intercambiar=intercambiar,a=a,min_freq=min_freq,)

        palabras = [word for word, _, _ in ranking]
        distancias = [distancia for _, distancia, _ in ranking]

        return palabras, distancias

    def is_known(self, palabra: str, min_freq: int | None = None) -> bool:
        palabra = self._normalizar_palabra(palabra)
        if min_freq is None:
            min_freq = self._min_freq
        return self._words.get(palabra, 0) >= min_freq

    def _normalizar_tildes(self, texto: str) -> str:
        return texto.translate(str.maketrans("áéíóúü", "aeiouu"))

    def _normalizar_frase(self, frase: str) -> str:
        frase = self._normalizar_tildes(frase.lower())
        return re.sub(r"[^\w\s]", "", frase)

    def _normalizar_palabra(self, palabra: str) -> str:
        palabra = self._normalizar_tildes(palabra.lower())
        return re.sub(r"[^a-zñ]", "", palabra)

    def _insert(self, word: str) -> set[str]:
        words_created = set()
        for i in range(len(word) + 1):
            for char in self._alphabet:
                new_word = word[:i] + char + word[i:]
                words_created.add(new_word)
        return words_created

    def _delete(self, word: str) -> set[str]:
        words_created = set()
        for i in range(len(word)):
            new_word = word[:i] + word[i + 1 :]
            words_created.add(new_word)
        return words_created

    def _replace(self, word: str) -> set[str]:
        words_created = set()
        for i in range(len(word)):
            for char in self._alphabet:
                new_word = word[:i] + char + word[i + 1 :]
                words_created.add(new_word)
        return words_created

    def _exchange(self, word: str) -> set[str]:
        words_created = set()
        for i in range(len(word) - 1):
            new_word = word[:i] + word[i + 1] + word[i] + word[i + 2 :]
            words_created.add(new_word)
        return words_created

    def _una_edicion(self, word: str, intercambiar: bool = False) -> set[str]:
        set1 = self._insert(word)
        set2 = self._delete(word)
        set3 = self._replace(word)
        if not intercambiar:
            return set1.union(set2, set3)
        set4 = self._exchange(word)
        return set1.union(set2, set3, set4)

    def _dos_ediciones(self, word: str, intercambiar: bool = False) -> set[str]:
        palabras_ed1 = self._una_edicion(word, intercambiar)
        palabras_ed2 = set()
        for palabra in palabras_ed1:
            palabras_ed2 = palabras_ed2.union(self._una_edicion(palabra, intercambiar))
        return palabras_ed2

    def _palabras_conocidas(self, candidatas: set[str], min_freq: int = 3) -> set[str]:
        return {pal for pal in candidatas if self._words.get(pal, 0) >= min_freq}

    def _candidatos(self, word: str, intercambiar: bool = True, min_freq: int = 3) -> set[str]:
        if self._words.get(word, 0) >= min_freq:
            return {word}
        c1 = self._palabras_conocidas(self._una_edicion(word, intercambiar),min_freq=min_freq,)
        if c1:
            return c1
        c2 = self._palabras_conocidas(self._dos_ediciones(word, intercambiar),min_freq=min_freq,)
        if c2:
            return c2

        return {word}

    def _mapa_teclado_es(self) -> dict[str, tuple[int, int]]:
        filas = ["qwertyuiop", "asdfghjklñ", "zxcvbnm"]
        posiciones = {}
        for i, fila in enumerate(filas):
            for j, letra in enumerate(fila):
                posiciones[letra] = (i, j)
        return posiciones

    def _distancia_euclidea_2d(self, p1: np.ndarray, p2: np.ndarray) -> float:
        pos1 = np.array(p1)
        pos2 = np.array(p2)
        return np.sqrt(np.sum((pos1 - pos2) ** 2))

    def _costo_reemplazo(self, letra_origen: str, letra_destino: str, a: float = 2.0) -> float:
        if letra_origen == letra_destino:
            return 0.0

        pos_origen = self._teclado_pos.get(letra_origen)
        pos_destino = self._teclado_pos.get(letra_destino)
        if pos_origen is None or pos_destino is None:
            return 1.0

        distancia = self._distancia_euclidea_2d(pos_origen, pos_destino)
        return distancia / a

    def _distancia_levenshtein_ponderada(self,origen: str,destino: str,a: float = 2.0,costo_intercambio: float = 0.5) -> float:
        filas, columnas = len(origen) + 1, len(destino) + 1

        matriz = []
        for i in range(filas):
            fila = []
            for j in range(columnas):
                fila.append(0.0)
            matriz.append(fila)

        for i in range(filas):
            matriz[i][0] = float(i)
        for j in range(columnas):
            matriz[0][j] = float(j)

        for i in range(1, filas):
            for j in range(1, columnas):
                costo_insertar = matriz[i][j - 1] + 1.0
                costo_borrar = matriz[i - 1][j] + 1.0
                costo_reemplazar_actual = (matriz[i - 1][j - 1]+ self._costo_reemplazo(origen[i - 1], destino[j - 1], a=a))

                matriz[i][j] = min(costo_insertar, costo_borrar, costo_reemplazar_actual)

                if (i > 1 and j > 1 and origen[i - 1] == destino[j - 2] and origen[i - 2] == destino[j - 1]):
                    matriz[i][j] = min(matriz[i][j],matriz[i - 2][j - 2] + costo_intercambio)

        return matriz[filas - 1][columnas - 1]

    def _sugerencias(self,word: str,top_k: int = 10,intercambiar: bool = True,a: float = 2.0,min_freq: int = 3,):

        word = self._normalizar_palabra(word)
        cands = self._candidatos(word, intercambiar, min_freq=min_freq)

        ranking = []
        for cand in cands:
            dist = self._distancia_levenshtein_ponderada(
                word,
                cand,
                a=a,
                costo_intercambio=self._costo_intercambio,
            )
            freq = self._words.get(cand, 0)
            ranking.append((cand, dist, freq))

        ranking.sort(key=lambda x: (x[1], -x[2], x[0]))
        return ranking[:top_k]
