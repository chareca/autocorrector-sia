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
##### Idea
Para corregir una frase, vamos palabra por palabra y cuando vemos una palabra que no existe en el vocabulario, se la pasamos al SistemaDistancias
para que nos devuelva las palabras más parecidas junto con sus distancias, después, esas palabras candidatas se las pasamos al SistemaContexto 
junto con la frase para que devuelva por cada una de ellas una probabilidad de contexto, luego, normalizamos las distancias y las probabilidades
para dejarlo todo entre 0 y 1, y agregamos por cada palabra esas 2 medidas con la media. Finalmente, el autocorrector elige la palabra de mayor
puntuación y va repitiendo todo esto por cada palabra hasta corregir una frase completa.
Esta idea no contempla a las palabras que son correctas en el diccionario pero que estan fuera de contexto. Lo he pensado así porque si quisieramos
contemplar esto, tendríamos intentar corregir cada palabra de la frase, independientemente de que exista o no en el vocabulario, y sinceramente, al
tener más palabras bien escritas que palabras mal escritas en una frase, creo que las palabras que están bien escritas las cambiaríamos por palabras
que no tienen sentido ponerlas allí, así que la liaríamos más (Creo).

##### Archivos
__main__.py : Carga los datos, entrena el autocorrector y prueba el autocorrector.
generador_muestras.py : Carga los datos de los libros y genera frases erroneas.
autocorrector.py : Tiene el sistema de distancias y el sistema de contexto junto con la lógica principal explicada anteriormente en la idea.
sistema_distancias.py : Calcula las palabras más cercanas a otra y sus distancias.
sistema_contexto.py : Calcula las probabilidades de aparición de las palabras candidatas a introducir en una posición de una frase.
