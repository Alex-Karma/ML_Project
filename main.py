# Импортируем необходимые библиотеки
import cv2
import numpy as np
import matplotlib.pyplot as plt

# импортируем файл с функциями для работы программы
import package


# получаем считанное фото и фото в бинарных данных (с реверсом цвета и без шумов)
img, img_bin = package.reading_photo("Photos/try4.png")

# определяем длинну ядра для распознания горизонталей и вертикалей, как сотую часть длинны картинки
kernel_len = np.array(img).shape[1]//100

# получаем вертикали и горизонтали
vertical_lines = package.vertical_lines_detection(kernel_len, img_bin)
horizontal_lines = package.horizontal_lines_detection(kernel_len, img_bin)

# получаем таблицу - объединениие верикалей и горизонталей
img_vh = package.combining_horizontals_and_verticals(horizontal_lines, vertical_lines)

# накладываем нашу отдельную табицу на фото
bitxor = cv2.bitwise_xor(img, img_vh)
bitnot = cv2.bitwise_not(bitxor)  # меняем цвета наоборот
# выводим это наложение
plotting = plt.imshow(bitnot, cmap='gray')
plt.show()

# находим контуры в таблице, чтобы дальше получить точные координтаты каждой ячейки таблицы
# Контуры - это список Python всех контуров на изображении.
# Каждый отдельный контур представляет собой числовой массив (x,y) координат граничных точек объекта.
contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# получаем отсортированные контуры сверху вниз и их рамки
contours, boundingBoxes = package.sort_contours(contours, method="top-to-bottom")

# формируем ячейки в формате: вершина (x,y), длинна и ширина
box = package.form_cells(contours, img)

# получем список всех ячеек по строкам (элементы в одинаковых идут подряд)
row = package.rows(box, boundingBoxes)

# находим максимальное число коллонок, для создания итоговой таблички
countcol = 0
for i in range(len(row)):
    if len(row[i]) > countcol:
        countcol = len(row[i])

# находим центры каждой колонки
center = [int(row[i][j][0]+row[i][j][2]/2) for j in range(len(row[i])) if row[0]]
center = np.array(center)
center.sort()

# получаем полностью отсортированный (теперь и слева направо) список ячеек
finalboxes = package.rows_sort(row, countcol, center)

# получаем список с распознанными значениями из ячеек
outer = package.recognition(finalboxes, bitnot)

# кладём данные в excel таблицу
package.data_to_excel(outer, row, countcol)
