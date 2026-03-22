import re
import os

class LectorCorpus:
    def __init__(self, ruta_directorio_libros):
        self.ruta_directorio_libros = ruta_directorio_libros
        self.corpus = []
        self._cargar_corpus()

    def _cargar_corpus(self):
        nombres_libros = os.listdir(self.ruta_directorio_libros)
        for nombre_libro in nombres_libros:
            if nombre_libro[-4:] != ".txt":
                continue

            if self.ruta_directorio_libros[-1] != "/":
                ruta_libro = self.ruta_directorio_libros + "/" + nombre_libro
            else:
                ruta_libro = self.ruta_directorio_libros + nombre_libro

            with open(ruta_libro, 'r', encoding='utf-8') as archivo:
                texto = archivo.read().lower()
                frases_texto = re.split(r'[.!?]', texto)

                for frase in frases_texto:
                    #saca todo lo q no seanletras
                    limpia = re.sub(r'[^a-záéíóúñ\s]', ' ', frase)
                    limpia = " ".join(limpia.split())
                    conteo_palabras = len(limpia.split())
                    if 5 <= conteo_palabras <= 20:
                        self.corpus.append(limpia)

    def get_corpus(self):
        return self.corpus.copy()
