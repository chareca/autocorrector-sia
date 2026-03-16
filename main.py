from typing import List, Tuple

class Preprocesador:
    def __init__(self):
        pass

    def preprocesar(self, rutas_ficheros: List[str]) -> List[str]:
        """
            Lee las frases que hay en cada uno de los ficheros, las preprocesa,
            y devuelve una lista con todas las frases de todos los ficheros.
        """
        pass

class SistemaDistancias:
    def __init__(self):
        pass

    def fit(self, X_train: List[str]):
        pass

    def predict(self, palabra: str, max_distancia: float) -> Tuple[List[str], List[float]]:
        """
            Devuelve una lista de las palabras más cercanas de tal forma que la 
            palabra de menor distancia esté en la primera posición y las de mayor
            distancia en la última posicion. Tambien devuelve las distancias de
            cada palabra.
          """
        pass

class SistemaContexto:
    def __init__(self, min_apperance: int=10, size_ngram: int=2):
        if min_apperance <= 1:
            raise ValueError("min_apperance > 1")
        if size_ngram <= 1:
            raise ValueError("size_ngram > 1")

        self._min_apperance = min_apperance
        self._size_ngram = size_ngram

        self._vocabulary = None
        self._ngram_matrix = None
        self._rows_ngram_matrix = None
        self._columns_ngram_matrix = None

    def fit(self, X_train: List[str]):
        X_train_split = self._split_phrases(X_train)
        X_train_split = self._add_unknow_words(X_train_split)
        self._vocabulary = self._calc_vocabulary(X_train_split)
        matrix, rows, columns = self._calc_ngram_matrix(X_train_split)

        self._ngram_matrix = matrix
        self._rows_ngram_matrix = rows
        self._columns_ngram_matrix = columns

    def _split_phrases(self, X_train: List[str]) -> List[List[str]]:
        X_train_split = []
        for i, phrase in enumerate(X_train):
            phrase_split = ['<s>' for j in range(self._size_ngram - 1)] + phrase.split() + ['</s>']
            X_train_split.append(phrase_split)
        return X_train_split
    
    def _add_unknow_words(self, X_train_split: List[List[str]]) -> List[List[str]]:
        freqs_words = {}
        for phrase_split in X_train_split:
            for word in phrase_split:
                if word in freqs_words:
                    freqs_words[word] += 1
                else:
                    freqs_words[word] = 1
        
        unknow_label = "<unk>"
        for i, phrase_split in enumerate(X_train_split):
            for j, word in enumerate(phrase_split):
                if freqs_words[word] < self._min_appearance:
                    X_train_split[i][j] = unknow_label
        return X_train_split

    def _calc_ngram_matrix(self, X_train_split: list[list[str]]):
        freqs_ngrams_n, freqs_ngrams_n_1 = self._ngram_freqs(X_train_split)
        
        columns = {'</s>': 0}
        for word in self._vocabulary:
            columns[word] = len(columns)
        
        ngrams = freqs_ngrams_n.keys()
        rows = {['<s>' for i in range(self._size_ngram - 1)]: 0}
        for ngram in ngrams:
            ngram_split = ngram.split()
            ngram_n_1 = ' '.join(ngram_split[:-1])

            if not(ngram_n_1 in rows):
                rows[ngram_n_1] = len(rows)

        matrix = [[1 / len(self._vocabulary) for j in range(len(columns))] for i in range(len(rows))]
        for ngram in ngrams:
            ngram_split = ngram.split()
            ngram_n_1 = ' '.join(ngram_split[:-1])
            end_word = ngram_split[-1]

            prob = (freqs_ngrams_n[ngram] + 1) / (freqs_ngrams_n_1[ngram_n_1] + len(self._vocabulary))
            matrix[rows[ngram_n_1]][columns[end_word]] = prob

        return matrix, rows, columns

    def _ngram_freqs(self, X_train_split: list[list[str]]) -> dict[str, int]:
        freqs_n = {}
        freqs_n_1 = {}
        n = self._size_ngram
        for phrase in X_train_split:
            for i in range(0, len(phrase) - n):
                ngram_n = ' '.join(phrase[i:i + n])
                if ngram_n in freqs_n:
                    freqs_n[ngram_n] += 1
                else:
                    freqs_n[ngram_n] = 1

                ngram_n_1 = ' '.join(phrase[i:i + (n - 1)])
                if ngram_n_1 in freqs_n_1:
                    freqs_n_1[ngram_n_1] += 1
                else:
                    freqs_n_1[ngram_n_1] = 1
        return freqs_n, freqs_n_1
    
    def _calc_vocabulary(self, X_train_split: List[List[str]]) -> set:
        vocabulary = set()
        for phrase_split in X_train_split:
            for word in phrase_split:
                vocabulary.add(word)
        return vocabulary

    def predict(self, frase: List[str], pos_pal: int, palabras_posibles: str) -> str:
        """ 
            Se le da la frase, la posición de la palabra a corregir, y las
            posibles palabras candidatas a ser la corrección de dicha palabra.
            Devuelve la probabilidad de que aparezca cada palabra posible.
        """
        unknow_row = ' '.join(['<unk>' for j in range(self._size_ngram - 1)])
        copy_phrase = ['<s>' for j in range(self._size_ngram - 1)] + frase + ['</s>']

        probs = []
        for candidate_word in palabras_posibles:
            new_pos_pal = pos_pal + (self._size_ngram - 1)
            ngram_n_1 = copy_phrase[new_pos_pal - (self._size_ngram - 1):new_pos_pal]
            for i, word in enumerate(ngram_n_1):
                if not(word in self._vocabulary):
                    ngram_n_1[i] = "<unk>"

            prob = 0
            if not(ngram_n_1 in self._rows_ngram_matrix) and candidate_word in self._columns_ngram_matrix:
                prob = self._ngram_matrix[self._rows_ngram_matrix[unknow_row]][self._columns_ngram_matrix[candidate_word]]
            elif ngram_n_1 in self._rows_ngram_matrix and candidate_word in self._columns_ngram_matrix:
                prob = self._ngram_matrix[self._rows_ngram_matrix[ngram_n_1]][self._columns_ngram_matrix[candidate_word]]
            probs.append(prob)

        return probs

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

    def corregir(self, frases: List[str]) -> List[str]:
        """ Corrige todas las frases que recibe """
        frases_corregidas = []
        for frase in frases:
            frase_corregida = []
            for i, palabra in enumerate(frase.split()):
                if palabra in self._vocabulario:
                    frase_corregida.append(palabra)
                    continue

                palabras_candidatas, distancias = self._sistema_distancias.predict(palabra, 2)
                probs_palabras = self._sistema_contexto.predict(frase, i, palabras_candidatas)
                
                palabras_candidatas_ordenadas = []
                for palabra, distancia, prob in zip(palabras_candidatas, distancias, probs_palabras):
                    palabras_candidatas_ordenadas.append((palabra, distancia, prob))
                palabras_candidatas_ordenadas = sorted(palabras_candidatas_ordenadas, key=lambda x: (x[1], -x[2]))
                palabra_corregida = palabras_candidatas_ordenadas[0]
                frase_corregida.append(palabra_corregida)
            frases_corregidas.append(frase_corregida)
        return frases_corregidas

if __name__ == '__main__':
    prep = Preprocesador()
    X_train = prep.preprocesar()
    X_val = ["frase_1...", "frase_2..."] # Ns si necesitaremos
    X_test = ["frase_1...", "frase_2..."]

    autocorrector = Autocorrector()
    autocorrector.fit(X_train)
    frases_corregidas = autocorrector.corregir(X_test)