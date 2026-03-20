#2 datasets, uno de train y otro de test, que el de train contenga frases
#aleatorias del corpus (carpeta libros) y el de test que tenga las mismas frases mal escritas, para que el modelo aprenda a corregirla
import re
import os

import pandas as pd
import random
import time

class GeneradorMuestras:
    def __init__(self, ruta_directorio_libros, ratio_error=0.1):
        self.ruta_directorio_libros = ruta_directorio_libros
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
                    limpia = re.sub(r'[^a-záéíóúñ\s]', '', frase)
                    limpia = " ".join(limpia.split())
                    conteo_palabras = len(limpia.split())
                    if 5 <= conteo_palabras <= 20:
                        self.corpus.append(limpia)

    def modificar_frase_con_errores(self, frase):
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
    
    def generar_frases_con_errores(self, cantidad = 100):
        #genera pares (frase_con_error, frase_correcta) para evaluar el modelo
        pares = []
        muestra = random.sample(self.corpus, min(cantidad, len(self.corpus)))
        for frase in muestra:
            frase_con_error = self.modificar_frase_con_errores(frase)
            pares.append((frase_con_error, frase))
        return pares

    def testear_modelo(self, modelo, cantidad = 100):
        start = time.perf_counter()
        sistema = modelo._sistema_distancias
        pares = self.generar_frases_con_errores(cantidad)
        print("frases con errores generadas en: ", time.perf_counter() - start, " segundos")
        #vamos a evaluar %precision palabras = (palabras_correctas / total_palabras)*100
        # y % mejora = (1 - (dist_final_total / dist_inicial_total))*100
        #para la mejora
        distancia_inicial= 0
        distancia_final = 0
        #para la precision
        palabras_correctas =0
        total_palabras = 0
        predicciones = modelo.corregir([par[0] for par in pares])
        print("frases corregidas tardo: ", time.perf_counter() - start, " segundos")
        frases_correctas = [par[1] for par in pares]
        frases_error = [par[0] for par in pares]
        for frase_pred, frase_corr, frase_err in zip(predicciones, frases_correctas, frases_error):
            palabras_pred = frase_pred.split()
            palabras_corr = frase_corr.split()
            palabras_err = frase_err.split()
            total_palabras += len(palabras_corr)
            for p_pred, p_corr, p_err in zip(palabras_pred, palabras_corr, palabras_err):
                if p_pred == p_corr:
                    palabras_correctas += 1
                distancia_inicial += sistema._calcular_coste_edicion_palabra(p_err, p_corr)
                distancia_final += sistema._calcular_coste_edicion_palabra(p_pred, p_corr)

        precision = (palabras_correctas / total_palabras)*100
        mejora = (1 - (distancia_final / distancia_inicial))*100

        print(f"Precision por palabra: {precision:.2f}%")
        print(f"Porcentaje de mejora: {mejora:.2f}%")


    def get_corpus(self):
        return self.corpus.copy()