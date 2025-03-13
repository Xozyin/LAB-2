# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 21:30:32 2025

@author: AM4
"""

# Попробуем обучить один нейрон на задачу классификации двух классов

import pandas as pd  # библиотека pandas нужна для работы с данными
import matplotlib.pyplot as plt  # matplotlib для построения графиков
import numpy as np  # numpy для работы с векторами и матрицами

# Считываем данные
# df = pd.read_csv('https://archive.ics.uci.edu/ml/'
#     'machine-learning-databases/iris/iris.data', header=None)

df = pd.read_csv('data.csv')

# смотрим что в них
print(df.head())

# три столбца - это признаки, четвертый - целевая переменная (то, что мы хотим предсказывать)

# выделим целевую переменную в отдельную переменную
y = df.iloc[:, 4].values

# так как ответы у нас строки - нужно перейти к численным значениям
y = np.where(y == "Iris-setosa", 1, -1)

# возьмем три признака для обучения
X = df.iloc[:, [0, 1, 2]].values  # теперь используем три признака

# Признаки в X, ответы в y - посмотрим на 3D-графике как выглядит задача
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # создаем 3D-график

# Визуализируем данные
ax.scatter(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2], color='red', marker='o', label='Iris-setosa')
ax.scatter(X[y == -1, 0], X[y == -1, 1], X[y == -1, 2], color='blue', marker='x', label='Iris-versicolor')
ax.set_xlabel('Признак 1')
ax.set_ylabel('Признак 2')
ax.set_zlabel('Признак 3')
plt.legend()
plt.show()

# переходим к созданию нейрона
# функция нейрона:
# значение = w1*признак1 + w2*признак2 + w3*признак3 + w0
# ответ = 1, если значение > 0
# ответ = -1, если значение < 0

def neuron(w, x):
    if ((w[1] * x[0] + w[2] * x[1] + w[3] * x[2] + w[0]) >= 0):
        predict = 1
    else:
        predict = -1
    return predict

# проверим как это работает (веса зададим пока произвольно)
w = np.array([0, 0.1, 0.4, 0.2])  # теперь 4 веса: w0, w1, w2, w3
print(neuron(w, X[1]))  # вывод ответа нейрона для примера с номером 1

# теперь создадим процедуру обучения
# корректировка веса производится по выражению:
# w_new = w_old + eta * x * y

# зададим начальные значения весов
w = np.random.random(4)  # теперь 4 веса: w0, w1, w2, w3
eta = 0.01  # скорость обучения
w_iter = []  # пустой список, в него будем добавлять веса, чтобы потом построить график
for xi, target, j in zip(X, y, range(X.shape[0])):
    predict = neuron(w, xi)
    w[1:] += (eta * (target - predict)) * xi  # target - predict - это и есть ошибка
    w[0] += eta * (target - predict)
    # каждую 10ю итерацию будем сохранять набор весов в специальном списке
    if j % 10 == 0:
        w_iter.append(w.tolist())

# посчитаем ошибки
sum_err = 0
for xi, target in zip(X, y):
    predict = neuron(w, xi)
    sum_err += (target - predict) / 2

print("Всего ошибок: ", sum_err)

# попробуем визуализировать процесс обучения
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # создаем 3D-график

# Визуализируем данные
ax.scatter(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2], color='red', marker='o', label='Iris-setosa')
ax.scatter(X[y == -1, 0], X[y == -1, 1], X[y == -1, 2], color='blue', marker='x', label='Iris-versicolor')
ax.set_xlabel('Признак 1')
ax.set_ylabel('Признак 2')
ax.set_zlabel('Признак 3')
plt.legend()

# Визуализируем разделяющую гиперплоскость
for i, w in zip(range(len(w_iter)), w_iter):
    # Уравнение гиперплоскости: w1*x1 + w2*x2 + w3*x3 + w0 = 0
    # Для визуализации создаем сетку точек по x1 и x2, а x3 вычисляем
    x1_range = np.linspace(min(X[:, 0]), max(X[:, 0]), 10)
    x2_range = np.linspace(min(X[:, 1]), max(X[:, 1]), 10)
    x1, x2 = np.meshgrid(x1_range, x2_range)
    x3 = -(w[1] * x1 + w[2] * x2 + w[0]) / w[3]  # вычисляем x3

    # Рисуем гиперплоскость
    ax.plot_surface(x1, x2, x3, alpha=0.5, color='gray')
    plt.pause(1)  # пауза для анимации

ax.text(x1_range[-1] - 0.3, x2_range[-1], x3[-1, -1], 'END', size=14, color='red')
plt.show()