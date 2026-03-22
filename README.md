# Trabajo Autocorrector
### Sistemas Inteligentes. Aplicaciones

#### Integrantes
 - Eloy Urriens
 - Juan Pulgarín
 - Tomas Alignani
 - Nicolás Chareca


#### Objetivos
- Distancia de teclado (No es lo mismo distancia M y N que distancia M y Q).
- Usar ngramas para integrar información contextual y no sugerir palabras correctas por diccionario pero fuera de contexto.
- ...


#### Explicacion
Autocorrector base + distancias:
    Para cada palabra de la frase:
        Si esta en el vocabulario => La dejamos igual
        Si no:
            Obtenemos varias palabras que están cerca, que son frecuentes y nos quedamos con la de menor distancia.

Autocorrector base + contexto:
    Para cada palabra de la frase:
        Si esta en el vocabulario => La dejamos igual
        Si no:
            Obtenemos varias palabras que están cerca, que son frecuentes y nos quedamos con la de mayor probabilidad
            de contexto.

Autocorrector base + distancias + contexto:
    Para cada palabra de la frase:
        Si esta en el vocabulario => La dejamos igual
        Si no:
            Obtenemos varias palabras que están cerca, que son frecuentes y escalamos al rango 0-1 sus distancias
            y probabilidades para agregar ambas medidas usando la media. Nos quedamos con la de mayor valor.

##### Archivos
* __main__.py : Carga los datos, entrena el autocorrector y prueba el autocorrector.
* generador_muestras.py : Carga los datos de los libros y genera frases erroneas.
* autocorrector.py : Tiene el sistema de distancias y el sistema de contexto junto con la lógica principal explicada anteriormente en la idea.
* sistema_distancias.py : Calcula las palabras más cercanas a otra y sus distancias.
* sistema_contexto.py : Calcula las probabilidades de aparición de las palabras candidatas a introducir en una posición de una frase.
