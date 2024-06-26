# Computacion_Social_Practica_2
Repositorio para la segunda práctica de la asignatura de computación social.

Distribution-based Learnable Filters with Side Information for Sequential Recommendation: https://dl.acm.org/doi/pdf/10.1145/3604915.3608782
Enlace al github mencionado: https://github.com/zxiang30/DLFS-Rec

El congreso ACM Conference on Recommender Systems (RecSys) recoge cada año las principales aportaciones de los investigadores de todo el mundo al campo de los sistemas de recomendación. El congreso se organiza en sesiones que agrupan las publicaciones en temas de interés para la conferencia de cada año. En este trabajo deberéis investigar los últimos avances en el ámbito de los sistemas de recomendación mediante la lectura, compresión, resumen y replicación de alguna de las publicaciones presentadas en el RecSys de los años 2023 y 2022.

Para poder replicar los artículos será necesario seleccionar solo aquellos que proporcionen tanto el código fuente (en Python) del algoritmo implementado en el artículo, como también los datasets utilizados para el estudio. Por ello, se ha hecho una preselección de artículos que tienen estas características. Podéis acceder a las publicaciones a partir de los siguientes enlaces (deberéis hacerlo desde la ETSISI o usando la VPN de la UPM):



Para la realización del trabajo debéis seleccionar uno de los artículos anteriormente mencionados y estudiarlo en profundidad. Cuando lo comprendáis debéis acceder al código fuente y replicar los experimentos realizados en el artículo con los mismos datasets. En este estudio de réplica se deberán comprobar las métricas explicadas en los cuadernillos de la asignatura (Precisión, Recall, F1, NDCG, ...) y en caso de no ser usadas en el artículo, hacer uso de ellas. Además, se deberá comparar los resultados del algoritmo del artículo con los algoritmos básicos explicados en la asignatura (KNN, MF, BeMF, Neural Collaborative Filtering) sobre los datasets utilizados en el artículo.

A continuación, se deberá realizar un resumen del artículo en un póster científico, junto con los resultados obtenidos del estudio de réplica. Un póster científico o póster académico es una forma de presentar la información que resulta de una investigación académica. A diferencia de un cartel, cuyo objetivo primordial es publicitario, la finalidad del póster es resumir, de forma accesible, física y conceptualmente, los procesos y resultados de una investigación.

Los requisitos a cumplir son:

La práctica se realizará en parejas.
El póster deberá tener un formato DIN A1: 841×594 mm.
El póster deberá entregarse en formato digital, junto con el código realizado durante el estudio de réplica, antes del viernes 17 de mayo a través de Moodle.
El viernes 24 de mayo se presentarán físicamente los pósteres.
La calificación de la práctica será puesta por los profesores en función de la calidad del póster y del código de réplica entregado.



**Título:** Distribution-based Learnable Filters with Side Information for Sequential Recommendation

**Autores:** Haibo Liu, Zhixiang Deng, Liang Wang, Jinjia Peng, Shi Feng

**Evento:** Proceedings of the 17th ACM Conference on Recommender Systems

**Páginas:** 78-88

**Año:** 2023


Créditos al profesor de la asignatura de Computación Social y Personalización por sus ideas para el trabajo y los respectivos dueños de los documentos originales



Los datasets que hemos empleado son los datasets que contienen los ratings de los datasets de Amazon (https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews) [All_Beauty.csv, Home_and_Kitchen.csv, Sports_and_Outdoors.csv, Toys_and_Games.csv]

En KNN, Matriz Factorization, Neural Collaborative Filter y Bernoulli Matrix se encuentran los modelos y las métricas utilizadas en clase
En prueba_paper hemos combinado el modelo estudiado con el NCF para lograr sacar las métricas teniendo como estudio qué tan buenas fueron las recomendaciones que le hicimos al usuario.
Las bajas métricas son debido a que hemos supuesto que nuestros id's de usuario y de item coinciden (cosa que en realidad  no ocurre) al no tener acceso a los mismos datos y el mismo preprocesamiento.

![POSTER](https://github.com/pandita-IA/Computacion_Social_Practica_2/blob/main/Infograf%C3%ADa%20CSP2%20(594%20x%20841%20mm).png)
