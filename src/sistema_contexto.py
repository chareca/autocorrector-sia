from typing import List, Tuple

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
                if freqs_words[word] < self._min_apperance:
                    X_train_split[i][j] = unknow_label
        return X_train_split
    
    def _calc_vocabulary(self, X_train_split: List[List[str]]) -> set:
        vocabulary = set()
        for phrase_split in X_train_split:
            for word in phrase_split:
                vocabulary.add(word)
        return vocabulary

    def _calc_ngram_matrix(self, X_train_split: list[list[str]]):
        freqs_ngrams_n, freqs_ngrams_n_1 = self._ngram_freqs(X_train_split)
        
        columns = {'</s>': 0}
        for word in self._vocabulary:
            if word != "</s>":
                columns[word] = len(columns)
        
        ngrams = freqs_ngrams_n.keys()
        start_unk = " ".join(['<s>' for i in range(self._size_ngram - 1)])
        rows = {start_unk: 0}
        for ngram in ngrams:
            ngram_split = ngram.split()
            ngram_n_1 = ' '.join(ngram_split[:-1])

            if not(ngram_n_1 in rows):
                if ngram_n_1 != start_unk:
                    rows[ngram_n_1] = len(rows)

        matrix = [[0 for j in range(len(columns))] for i in range(len(rows))]
        for ngram in ngrams:
            ngram_split = ngram.split()
            ngram_n_1 = ' '.join(ngram_split[:-1])
            end_word = ngram_split[-1]

            prob = (freqs_ngrams_n[ngram] + 1) / (freqs_ngrams_n_1[ngram_n_1] + len(self._vocabulary))
            matrix[rows[ngram_n_1]][columns[end_word]] = prob

        return matrix, rows, columns

    def _ngram_freqs(self, X_train_split: list[list[str]]) -> Tuple[dict[str, int], dict[str, int]]:
        freqs_n = {}
        freqs_n_1 = {}
        n = self._size_ngram
        for phrase in X_train_split:
            for i in range(0, len(phrase) - n + 1):
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

    def predict(self, frase: List[str], pos_pal: int, num_sugerencias: int) -> Tuple[List[str], List[float]]:
        """ 
            Se le da la frase y la posición de la palabra a corregir,
            Devuelve las palabras más probables a poner en dicha posición junto con sus probabilidades
            y todas ordenadas de menor a mayor.
        """

        unknow_row = ' '.join(['<unk>' for j in range(self._size_ngram - 1)])
        copy_phrase = ['<s>' for j in range(self._size_ngram - 1)] + frase + ['</s>']

        # Obtenemos el ngrama n - 1
        new_pos_pal = pos_pal + (self._size_ngram - 1)
        ngram_n_1 = copy_phrase[new_pos_pal - (self._size_ngram - 1):new_pos_pal]
        for i, word in enumerate(ngram_n_1):
            if not(word in self._vocabulary):
                ngram_n_1[i] = "<unk>"
        ngram_n_1 = " ".join(ngram_n_1)

        if not(ngram_n_1 in self._rows_ngram_matrix) and unknow_row in self._rows_ngram_matrix:
            ngram_n_1 = unknow_row

        # Obtenemos las palabras más probables
        words_sugeridas = []
        sugerencias_prohibidas = set(["<s>", "</s>", "<unk>"])
        row_ngram_matrix = self._ngram_matrix[self._rows_ngram_matrix[ngram_n_1]]
        for end_word in self._columns_ngram_matrix:
            if not(end_word in sugerencias_prohibidas):
                prob = row_ngram_matrix[self._columns_ngram_matrix[end_word]]
                words_sugeridas.append((end_word, prob))
        words_sugeridas = sorted(words_sugeridas, key=lambda x: -x[1])
        words_sugeridas = words_sugeridas[:num_sugerencias]

        words = []
        probs = []
        for word, prob in words_sugeridas:
            words.append(word)
            probs.append(prob)
        return words, probs