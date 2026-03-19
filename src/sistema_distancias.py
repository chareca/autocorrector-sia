import re
import numpy as np

from typing import List, Tuple

class SistemaDistancias:
    def __init__(self):
        self._words = {}
        self._distances = {}

    def fit(self, X_train: List[str]):
        self._create_words_freqs(X_train)
        self._create_distances()

    def _create_words_freqs(self, X_train: List[str]):
        self._words = {}
        for frase in X_train:
            linea = frase.lower()
            linea = re.sub(r'[^\w\s]','', linea)

            for word in linea.split():
                self._words[word] = self._words.get(word,0) + 1
    
    def _create_distances(self):
        tildes = {
            "á": "a",
            "é": "e",
            "í": "i",
            "ó": "o",
            "ú": "u"
        }

        self._distances = {}
        matrix = self._position_matrix()
        keys = list(matrix.keys())
        values = list(matrix.values())
        for i in range(len(keys)):
            for j in range(len(keys)):
                string = keys[i] + keys[j]
                self._distances[string] = self._euclidean(values[i], values[j])

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
        
        return dic

    def _euclidean(self, pos1: np.ndarray, pos2: np.ndarray):
        pos1 = np.array(pos1)
        pos2 = np.array(pos2)
        return np.sqrt(np.sum((pos1-pos2)**2))

    def _order_ascii(self, letter1, letter2=None):
        if letter2 is None and len(letter1) == 2:
            letter2 = letter1[1]
        elif letter2 is None:
            raise ValueError(["[ERROR]: Introduce solo un string de 2 letras o dos letras por separado."])
        return "".join(sorted(letter1+letter2))

    def predict(self, palabra: str, max_correciones: int, intercambiar=False) -> Tuple[List[str], List[float]]:
        """
            Devuelve una lista de las palabras más cercanas de tal forma que la 
            palabra de menor distancia esté en la primera posición y las de mayor
            distancia en la última posicion. Tambien devuelve las distancias de
            cada palabra.
          """
        for words_posibles in [self._una_edicion(palabra, intercambiar), self._dos_ediciones(palabra, intercambiar)]:
            validas = {}
            for word in words_posibles:
                if word in self._words:
                    validas[word] = self._words[word]
            validas_sort = sorted(validas.items(), key=lambda x: x[1], reverse=True)
        
            if len(validas_sort) > 0:
                words = []
                costes = []
                minimo = min(len(validas_sort), max_correciones)
                for tupla in validas_sort[:minimo]:
                    costes.append(self._calcular_coste_edicion_palabra(palabra, tupla[0]))
                    words.append(tupla[0])
                return words, costes
        return [palabra], [0]

    def _una_edicion(self, word:str, intercambiar: bool=False) ->set:
        set1 = self._insert(word)
        set2 = self._delete(word)
        set3 = self._replace(word)
        if not intercambiar:
            words_one_edition = set1.union(set2, set3)
        else:
            set4 = self._exchange(word)
            words_one_edition = set1.union(set2, set3, set4)

        return words_one_edition

    def _dos_ediciones(self, word:str,intercambiar:bool =False) ->set:
        palabras_ed1 = self._una_edicion(word, intercambiar)
        palabras_ed2 = set()
        for palabra in palabras_ed1:
            palabras_ed2 = palabras_ed2.union(self._una_edicion(palabra, intercambiar))
        return palabras_ed2
    
    def _insert(self, word: str) -> set:
        alphabet = 'abcdefghijklmnñopqrstuvwxyz'
        words_created = set()
        for i in range(len(word) + 1):
            for char in alphabet:
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
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        words_created = set()
        for i in range(len(word)): #hola
            for char in alphabet:
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
    ) -> tuple[int, np.ndarray]:
    
        coste_insertar = 1
        coste_borrar = 1
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
                    matriz[i, j] = matriz[i - 1, j - 1]
                else:
                    coste_reemplazar = self._distances[palabra_original[i - 1] + palabra_corregida[j - 1]]
                    costes_manipulaciones = [coste_borrar, coste_insertar, coste_reemplazar]
                    costes_anteriores = [matriz[i - 1, j], matriz[i, j - 1], matriz[i - 1, j - 1]]
                    pos = np.argmin(costes_anteriores)
                    matriz[i, j] = costes_anteriores[pos] + costes_manipulaciones[pos]
        return matriz[-1, -1]