# Импортируем необходимые библиотеки
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract

# указываем путь к tesseract модулю Tesseract-OCR для библиотеки pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# считываем картинку в оттенках серого
img_name = "Photos/try.png"
img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
# выводим картинку
cv2.imshow("Image_1", img)
cv2.waitKey(0)

# переводим фото в бинарные данные
# и избавляемся от всех шумов, свидением картинки к 2 цветам - чёрный(фон) и белый(надписи)
thresh, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
# выводим это фото
cv2.imshow("Image_2", img_bin)
cv2.waitKey(0)

# определяем длинну ядра для распознания горизонталей и вертикалей, как сотую часть длинны картинки
kernel_len = np.array(img).shape[1]//100
# определяем вертикальное ядро для распознавания вертикалей
ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
# определяем горизонтальное ядро для распознавания горизонталей
hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
# определяем ядро для объединения вертикалей и горизонталей
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

# используем вертикальное ядро для распознавания вертикалей
image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)
# рисуем вертикали
plotting = plt.imshow(image_1, cmap='gray')
plt.show()

# используем горизонтальное ядро для распознавания горизонталей
image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)
# рисуем горизонтали
plotting = plt.imshow(image_2, cmap='gray')
plt.show()

# объединяем горизонтали и вертикали
img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
# Eroding and thresholding картинку
img_vh = cv2.erode(~img_vh, kernel, iterations=2)
thresh, img_vh = cv2.threshold(img_vh, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# выводим объединённые гооризонтали и вертикали - таблицу
plotting = plt.imshow(img_vh, cmap='gray')
plt.show()

# накладываем нашу отдельную табицу на фото
bitxor = cv2.bitwise_xor(img, img_vh)
bitnot = cv2.bitwise_not(bitxor) # меняем цвета наоборот
# выводим это наложение
plotting = plt.imshow(bitnot, cmap='gray')
plt.show()

# находим контуры в таблице, чтобы дальше получить точные координтаты каждой ячейки таблицы
# Контуры - это список Python всех контуров на изображении.
# Каждый отдельный контур представляет собой числовой массив (x,y) координат граничных точек объекта.
contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# функция сортирует контуры слева направо и сверху вниз
def sort_contours(cnts, method="left-to-right"):
    # reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes(рамки контуров) and sort them from top to bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

# сортируетм контуры сверху вниз
contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")

# список высот рамок контуров
heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
# получаем среднее значение этих высот
mean = np.mean(heights)

# список всех ячеек - вершина (x,y), длинна и ширина
box = []
for c in contours:
    # получаем значения рамок контуров
    x, y, w, h = cv2.boundingRect(c)
    if w < 1000 and h < 500:
        # рисуем рамки контуров на фото
        # для ячеек с ограниченной шириной и длинной - должна быть меньше размера таблицы
        image = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        box.append([x, y, w, h])
# выводим картинку с прорисованными рамками контуров на фото
plotting = plt.imshow(image, cmap='gray')
plt.show()

# списки с распределением ячеек по строкам и столбцам

# До тех пор, пока поле не отличается больше, чем его собственное (высота + среднее значение / 2),
# поле находится в том же ряду. Как только разница в высоте становится больше текущей (высота + среднее значение /2),
# мы знаем, что начинается новая строка. Столбцы логически расположены слева направо.
row = []
column = []
j = 0
for i in range(len(box)):
    if i == 0:
        column.append(box[i])
        previous = box[i]
    else:
        if box[i][1] <= previous[1] + mean/2:
            column.append(box[i])
            previous = box[i]
            if i == len(box)-1:
                row.append(column)
        else:
            row.append(column)
            column = []
            previous = box[i]
            column.append(box[i])
print(column)
print(row)

# находим максимальную число коллонок, для создания итоговой таблички
countcol = 0
for i in range(len(row)):
    countcol = len(row[i])
    if countcol > countcol:
        countcol = countcol

# находим центры каждой колонки
center = [int(row[i][j][0]+row[i][j][2]/2) for j in range(len(row[i])) if row[0]]
center = np.array(center)
center.sort()

# сортируем значения в правильном порядке относительно расстояния до центра столбцов
finalboxes = []
for i in range(len(row)):
    lis = []
    for k in range(countcol):
        lis.append([])
    for j in range(len(row[i])):
        diff = abs(center-(row[i][j][0]+row[i][j][2]/4))
        minimum = min(diff)
        indexing = list(diff).index(minimum)
        lis[indexing].append(row[i][j])
    finalboxes.append(lis)

# распознаём значения на каждой ячейке и складываем в список
outer = []
for i in range(len(finalboxes)):
    for j in range(len(finalboxes[i])):
        inner = ''
        if len(finalboxes[i][j]) == 0:
            outer.append(' ')
        else:
            for k in range(len(finalboxes[i][j])):
                y, x, w, h = finalboxes[i][j][k][0], finalboxes[i][j][k][1], finalboxes[i][j][k][2], finalboxes[i][j][k][3]
                finalimg = bitnot[x:(x + h), y:(y + w)]
                # подготавливаем содержимое ячейки к распознанию содержимого
                # расширяя и размывая содержимое
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
                border = cv2.copyMakeBorder(finalimg, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255])
                resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                dilation = cv2.dilate(resizing, kernel, iterations=1)
                erosion = cv2.erode(dilation, kernel, iterations=1)
                # анализируем с помощью pytesseract
                out = pytesseract.image_to_string(erosion)
                if len(out) == 0:
                    out = pytesseract.image_to_string(erosion, config='--psm 3')
                inner = inner + " " + out
            outer.append(inner)

# создаём dataframe сгенирированных OCR списков
arr = np.array(outer)
dataframe = pd.DataFrame(arr.reshape(len(row), countcol))
print(dataframe)
data = dataframe.style.set_properties(align="left")
# создаём excel
writer = pd.ExcelWriter('output.xlsx')
# записываем в excel
data.to_excel(writer)
writer.save()
