import re
import numpy as np

from typing import List, Tuple

class SistemaDistancias:
    def __init__(self):
        self._words = {}
        self._distances = {}
        self._alphabet = "abcdefghijklmnñopqrstuvwxyzáéíóúü"
        self._replacement_scale = 2

    def fit(self, X_train: List[str]):
        self._create_words_freqs(X_train)
        self._create_distances()

    def _create_words_freqs(self, X_train: List[str]):
        self._words = {}
        for frase in X_train:
            linea = self._normalize_phrase(frase)

            for word in linea.split():
                self._words[word] = self._words.get(word,0) + 1
    
    def _create_distances(self):
        self._distances = {}
        matrix = self._position_matrix()

        keys = list(matrix.keys())
        values = list(matrix.values())
        
        for i in range(len(keys)):
            for j in range(len(keys)):
                string = keys[i] + keys[j]
                self._distances[string] = self._euclidean(values[i], values[j]) / self._replacement_scale

    def _position_matrix(self):
        filas = ["qwertyuiop", "asdfghjklñ", "zxcvbnm"]

        dic = {}
        for i, fila in enumerate(filas):
            for j, letter in enumerate(fila):
                dic[letter] = [i,j]
        dic["á"] = dic["a"]
        dic["é"] = dic["e"]
        dic["í"] = dic["i"]
        dic["ó"] = dic["o"]
        dic["ú"] = dic["u"]
        dic["ü"] = dic["u"]
        
        return dic

    def _euclidean(self, pos1: np.ndarray, pos2: np.ndarray):
        pos1 = np.array(pos1)
        pos2 = np.array(pos2)
        return np.sqrt(np.sum((pos1-pos2)**2))

    def _normalize_phrase(self, phrase: str) -> str:
        phrase = phrase.lower()
        return re.sub(r'[^a-záéíóúñ\s]', ' ', phrase)

    def _normalize_word(self, word: str) -> str:
        word = word.lower()
        return re.sub(r'[^a-záéíóúñ]', '', word)

    def predict(self, palabra: str, max_correciones: int, intercambiar=True) -> Tuple[List[str], List[float]]:
        """
            Devuelve una lista de las palabras más cercanas de tal forma que la 
            palabra de menor distancia esté en la primera posición y las de mayor
            distancia en la última posicion. Tambien devuelve las distancias de
            cada palabra.
          """
        palabra = self._normalize_word(palabra)
        if max_correciones <= 0:
            return [], []
        if palabra in self._words:
            return [palabra], [0.0]

        for words_posibles in [self._una_edicion(palabra, intercambiar), self._dos_ediciones(palabra, intercambiar)]:
            ranking = []
            for word in words_posibles:
                if word in self._words:
                    distance = self._calcular_coste_edicion_palabra(palabra, word)
                    frequency = self._words[word]
                    ranking.append((word, distance, frequency))

            if len(ranking) > 0:
                #prueba 1 :
                #ranking.sort(key=lambda x: (x[1], -x[2], x[0])) ->76%
                #prueba 2
                #ranking.sort(key=lambda x: (round(x[1], 3), x[0])) ->75.9%
                #prueba3
                # ranking.sort(key=lambda x: (x[1], x[0])) ->75.9%
                ranking.sort(key=lambda x: (x[1], -x[2], x[0]))
                top_candidates = ranking[:max_correciones]
                words = [word for word, _, _ in top_candidates]
                costes = [distance for _, distance, _ in top_candidates]
                return words, costes
        return [palabra], [0.0]

    def _una_edicion(self, word: str, intercambiar: bool = False) -> set:
        return self._insert(word).union(self._delete(word), self._replace(word), self._exchange(word) if intercambiar else None)

    def _dos_ediciones(self, word: str, intercambiar: bool = False) -> set:
        palabras_ed1 = self._una_edicion(word, intercambiar)
        palabras_ed2 = set()

        for palabra in palabras_ed1:
            palabras_ed2.update(self._una_edicion(palabra, intercambiar))

        return palabras_ed2
    
    def _insert(self, word: str) -> set:
        words_created = set()
        for i in range(len(word) + 1):
            for char in self._alphabet:
                new_word = word[:i] + char + word[i:]
                words_created.add(new_word)
        
        return words_created

    def _delete(self, word:str) -> set:
        words_created = set()
        for i in range(len(word)):
            new_word = word[:i] + word[i+1:]
            words_created.add(new_word)

        return words_created
        
    def _replace(self, word:str) ->set:
        words_created = set()
        for i in range(len(word)):
            for char in self._alphabet:
                new_word = word[:i] + char + word[i+1:]
                words_created.add(new_word)

        return words_created

    def _exchange(self, word:str)->set:
        words_created = set()
        for i in range(len(word) -1):
            new_word = word[:i] + word[i+1] + word[i] + word[i+2:]
            words_created.add(new_word)

        return words_created

    def _calcular_coste_edicion_palabra(
        self,
        palabra_original: str,
        palabra_corregida: str,
    ) -> float:
    
        coste_insertar = 1
        coste_borrar = 1
        coste_transposicion = 0.5
        # coste_reemplazar = (Depende, abajo se pone)

        matriz = np.zeros((
            len(palabra_original) + 1,
            len(palabra_corregida) + 1
        ))
        idx_filas = np.arange(0, matriz.shape[0])
        idx_cols = np.arange(0, matriz.shape[1])
        
        matriz[:, 0] = idx_filas
        matriz[0, :] = idx_cols

        for i in range(1, matriz.shape[0]):
            for j in range(1, matriz.shape[1]):
                if palabra_original[i - 1] == palabra_corregida[j - 1]:
                    coste_reemplazar = 0.0
                else:
                    coste_reemplazar = max(1.0,self._distances.get(palabra_original[i - 1] + palabra_corregida[j - 1],1.0,))


                mejor_coste = min(
                    matriz[i - 1, j] + coste_borrar,
                    matriz[i, j - 1] + coste_insertar,
                    matriz[i - 1, j - 1] + coste_reemplazar,
                )

                if (
                    i > 1 and j > 1
                    and palabra_original[i - 1] == palabra_corregida[j - 2]
                    and palabra_original[i - 2] == palabra_corregida[j - 1]
                ):
                    mejor_coste = min(
                        mejor_coste,
                        matriz[i - 2, j - 2] + coste_transposicion,
                    )

                matriz[i, j] = mejor_coste

        return float(matriz[-1, -1])
