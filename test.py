#2 datasets, uno de train y otro de test, que el de train contenga frases
#aleatorias del corpus (carpeta libros) y el de test que tenga las mismas frases mal escritas, para que el modelo aprenda a corregirla
import re

import pandas as pd
import random


class GeneradorDataset:
    def __init__(self, ruta_corpus, ratio_error=0.1):
        self.ruta_corpus = ruta_corpus
        self.ratio_error = ratio_error
        self.corpus = []
        # Mapa simplificado de cercanía de teclado
        self.teclas_adyacentes = {
            'a': 'qwsz', 's': 'awedxz', 'd': 'erfcxs', 'f': 'rtgvcd',
            'g': 'tyhbvf', 'h': 'yujnbg', 'j': 'uikmnh', 'k': 'iolmj',
            'l': 'opk', 'z': 'asx', 'x': 'sdzc', 'c': 'dfxv', 'v': 'cfgb',
            'b': 'vghn', 'n': 'bhjm', 'm': 'njk'
        }

    def cargar_corpus(self):
        with open(self.ruta_corpus, 'r', encoding='utf-8') as archivo:
            texto = archivo.read()
            frases_texto = re.split(r'[.!?]', texto)

            for frase in frases_texto[:1000]:
                #saca todo lo q no seanletras
                limpia = re.sub(r'[^a-zA-ZáéíóúüñÁÉÍÓÚÜÑ\s]', '', frase)
                limpia = " ".join(limpia.split())
                conteo_palabras = len(limpia.split())
                if 5 <= conteo_palabras <= 20:
                    self.corpus.append(limpia)

    def generar_frase_con_error(self, frase):
        palabras = frase.split()
        resultado = []
        for palabra in palabras:
            if random.random() < self.ratio_error and len(palabra) > 1:
                lista_char = list(palabra.lower())
                i = random.randint(0, len(lista_char) - 1)
                char = lista_char[i]
                tipo_error = random.choice(['sustitucion', 'eliminacion', 'insercion','intercambio'])
                if tipo_error == 'intercambio' and i < len(lista_char) - 1:
                    lista_char[i], lista_char[i+1] = lista_char[i+1], lista_char[i]
                elif tipo_error == 'sustitucion' and char in self.teclas_adyacentes:
                    lista_char[i] = random.choice(self.teclas_adyacentes[char])
                elif tipo_error == 'eliminacion':
                    lista_char.pop(i)
                elif tipo_error == 'insercion':
                    lista_char.insert(i, random.choice('aeiou'))

                resultado.append(''.join(lista_char))
            else: 
                resultado.append(palabra)
        return ' '.join(resultado)
    # falta generar el csv o donde se guarden las frases


#probar
gen = GeneradorDataset('libros/LasFurias.txt', 0.5)
gen.cargar_corpus()

print(f"Frases cargadas: {len(gen.corpus)}")
print("Ejemplo de frase cargada:")
print(gen.corpus[1])
print("\nEjemplo de frase con error:")
print(gen.generar_frase_con_error(gen.corpus[1]))
