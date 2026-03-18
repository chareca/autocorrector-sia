#2 datasets, uno de train y otro de test, que el de train contenga frases
#aleatorias del corpus (carpeta libros) y el de test que tenga las mismas frases mal escritas, para que el modelo aprenda a corregirla
import re

import pandas as pd
import random

class GeneradorMuestras:
    def __init__(self, ruta_corpus, ratio_error=0.1):
        self.ruta_corpus = ruta_corpus
        self.ratio_error = ratio_error
        self.corpus = []
        # Mapa simplificado de cercanía de teclado
        self.teclas_adyacentes = {
            'q': 'wqa', 'w': 'qweas', 'e': 'wersd', 'r': 'rftdg', 't': 'tgyhu',
            'y': 'yhguj', 'u': 'uijhk', 'i': 'iikjl', 'o': 'oolkp', 'p': 'ppo',
            'a': 'qwsz', 's': 'awedxz', 'd': 'erfcxs', 'f': 'rtgvcd',
            'g': 'tyhbvf', 'h': 'yujnbg', 'j': 'uikmnh', 'k': 'iolmj',
            'l': 'opk', 'z': 'asx', 'x': 'sdzc', 'c': 'dfxv', 'v': 'cfgb',
            'b': 'vghn', 'n': 'bhjm', 'm': 'njk', 'á': 'qwsz', 'é': 'awedxz', 'í': 'erfcxs', 'ó': 'rtgvcd',
            'ú': 'tyhbvf', 
        }
        

    def cargar_corpus(self):
        with open(self.ruta_corpus, 'r', encoding='utf-8') as archivo:
            texto = archivo.read()
            frases_texto = re.split(r'[.!?]', texto)

            for frase in frases_texto[:2000]:
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
    
    def generar_evaluacion(self, cantidad = 100):
        #genera pares (frase_con_error, frase_correcta) para evaluar el modelo
        pares = []
        muestra = random.sample(self.corpus, min(cantidad, len(self.corpus)))
        for frase in muestra:
            frase_con_error = self.generar_frase_con_error(frase)
            pares.append((frase_con_error, frase))
        return pares

    def testear(self, modelo, cantidad = 100):
        pares = self.generar_evaluacion(cantidad)
        correctas = 0
        for frase_con_error, frase_correcta in pares:
            correccion = modelo.corregir(frase_con_error)
            if correccion == frase_correcta:
                correctas += 1
        return correctas / len(pares) if pares else 0



