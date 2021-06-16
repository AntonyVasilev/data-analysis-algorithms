# data-analysis-algorithms

### Задача: предсказать средний балл на экзамене по математике, который получают ученики репетиторов из датасета test.csv. Вам будут даны два датасета: train.csv (содержит признаки и целевую переменную) и test.csv (только признаки).
Метрика для оценки – Коэффициент детерминации.
При выполнении проекта допускается использовать только импорты следующих библиотек:  
import numpy as np  
import pandas as pd  
from sklearn.model_selection import train_test_split  
import matplotlib.pyplot as plt  
import seaborn as sns  

### Решение.
Собственноручно написал алгоритм модели градиентного бустинга, основанного на деревьях решений.  
Наилучший результат: 0.75797
