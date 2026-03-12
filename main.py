from typing import List

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

    def predict(self, palabra: str, max_distancia: float) -> List[str]:
        """
            Devuelve una lista de las palabras más cercanas de tal forma que la 
            palabra de menor distancia esté en la primera posición y las de mayor
            distancia en la última posicion
          """
        pass

class SistemaContexto:
    def __init__(self):
        pass

    def fit(self, X_train: List[str]):
        pass

    def predict(self, frase: List[str], pos_pal: int, palabras_posibles: str) -> str:
        """ 
            Se le da la frase, la posición de la palabra a corregir, y las
            posibles palabras candidatas a ser la corrección de dicha palabra.
            Devuelve la mejor palabra de las candidatas en función del contexto.
        """
        pass

class Autocorrector:
    def __init__(self):
        self._sistema_distancias = SistemaDistancias()
        self._sistema_contexto = SistemaContexto()

    def fit(self, X_train: List[str]) -> None:
        """ Se entrena con una lista de frases """
        pass

    def corregir(self, frases: List[str]) -> List[str]:
        """ Corrige todas las frases que recibe """
        pass

if __name__ == '__main__':
    prep = Preprocesador()
    X_train = prep.preprocesar()
    X_val = ["frase_1...", "frase_2..."] # Ns si necesitaremos
    X_test = ["frase_1...", "frase_2..."]

    autocorrector = Autocorrector()
    autocorrector.fit(X_train)
    frases_corregidas = autocorrector.corregir(X_test)