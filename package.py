# Импортируем необходимые библиотеки
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import interpolation as inter
import pytesseract


# указываем путь к tesseract модулю Tesseract-OCR для библиотеки pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def rotate_to_correct(img):
    # функция поворачивает картинку в правильное - горизонтальное положение текста

    # функция определения правильности положения картинки
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score

    # переводим фото в бинарные данные
    # и избавляемся от всех шумов, свидением картинки к 2 цветам - чёрный(фон) и белый(надписи)
    thresh, img_bin = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # нахождение угла поворота
    scores = []
    angles = np.arange(0, 360, 1)
    for angle in angles:
        histogram, score = determine_score(img_bin, angle)
        scores.append(score)
    best_angle = angles[scores.index(max(scores))]

    # нахождение матрицы поворота
    (height, width) = img_bin.shape[:2]
    center = (width // 2, height // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)

    # нахождение повернутого изображения для изменения разрешения
    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    M[0, 2] += bound_w / 2 - center[0]
    M[1, 2] += bound_h / 2 - center[1]

    return cv2.warpAffine(img, M, (bound_w, bound_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def reading_photo(path):
    # функция считывает фото и выдаёт его само + фото в бинарных данных (с реверсом цвета и без шумов)

    # считываем картинку
    img_name = path
    origin = cv2.imread(img_name)
    cv2.imshow("Origin Image", origin)
    cv2.waitKey(0)

    # поворачиваем фото в правильное состояние
    origin = rotate_to_correct(origin)

    # увеличиваем расширение картинки минимум до 600 пиксилей в высоту для более корректной работы
    if np.array(origin).shape[0] < 600:
        k = 600 / np.array(origin).shape[0]
        origin = cv2.resize(origin, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)

    # сохраняем откорректированный оригинал фото
    cv2.imwrite("Detections/resized_origin.jpg", origin)

    # начинаем предобработку фото, переводя в серый цвет
    img = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)

    # подготавливаем содержимое к распознанию, размывая
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    img = cv2.erode(img, kernel, iterations=1)
    # выводим картинку
    cv2.imshow("Preprocessed Image", img)
    cv2.waitKey(0)

    # переводим фото в бинарные данные и избавляемся от всех шумов, свидением картинки к 2 цветам - чёрный(фон) и белый(надписи)
    img_bin = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    # выводим это фото
    cv2.imshow("Thresholded Image", img_bin)
    cv2.waitKey(0)

    return origin, img, img_bin


def vertical_lines_detection(kernel_len, img_bin):
    # функция распознаёт вертикальные линии, формирующие таблицу

    # определяем вертикальное ядро для распознавания вертикалей
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    # используем вертикальное ядро для распознавания вертикалей
    image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)
    cv2.imwrite("Detections/verticals.jpg", vertical_lines)
    # рисуем вертикали
    plt.imshow(image_1, cmap='gray')
    plt.show()
    return vertical_lines


def horizontal_lines_detection(kernel_len, img_bin):
    # функция распознаёт вертикальные линии, формирующие таблицу

    # определяем горизонтальное ядро для распознавания горизонталей
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    # используем горизонтальное ядро для распознавания горизонталей
    image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)
    cv2.imwrite("Detections/horizontals.jpg", horizontal_lines)
    # рисуем горизонтали
    plotting = plt.imshow(image_2, cmap='gray')
    plt.show()
    return horizontal_lines


def combining_horizontals_and_verticals(horizontal_lines, vertical_lines):
    # функция объединяет горизонтали и вертикали, для формирования цельной таблицы без содержимого

    # определяем ядро для объединения вертикалей и горизонталей
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # объединяем горизонтали и вертикали
    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    # Eroding and thresholding картинку
    img_vh = cv2.erode(~img_vh, kernel, iterations=2)
    thresh, img_vh = cv2.threshold(img_vh, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite("Detections/table.jpg", img_vh)
    # выводим объединённые гооризонтали и вертикали - таблицу
    plotting = plt.imshow(img_vh, cmap='gray')
    plt.show()
    return img_vh


def sort_contours(cnts, method="left-to-right"):
    # функция сортирует контуры слева направо и сверху вниз (в нашем случае сверху вниз)

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


def form_cells(contours, origin):
    # функция формирует из контуров список ячеек с рамками

    # получаем ширину и длину фото
    width = np.array(origin).shape[1]
    height = np.array(origin).shape[0]

    # список всех ячеек - вершина (x,y), длинна и ширина
    box = []
    for c in contours:
        # получаем значения рамок контуров
        x, y, w, h = cv2.boundingRect(c)
        if w < width * 0.5 and h < height * 0.5:
            # для ячеек с ограниченной шириной и длинной - должна быть меньше размера таблицы
            box.append([x, y, w, h])

    # проверяем лежит ли какая-то ячейка внутри другой и удаляем её, так как она точно лишняя
    for j in box:
        for i in box:
            if i != j and i[1] < j[1] and i[0] < j[0] and i[2] - (j[0] - i[0]) > j[2] and i[3] - (j[1] - i[1]) > j[3]:
                box.remove(j)
                break

    # Если погрешность в сравнении чисел меньше какого-то процента, то ОК
    def mistake(a, b, percent=0.05):
        if b != 0 and a / b <= 1 and 1 - a / b < percent:
            return 1
        if a != 0 and b / a <= 1 and 1 - b / a < percent:
            return 1
        return 0

    # проверяем лежат ли две ячейки в одном столбце и в одной строке - погрешность, из-за черты, которую человек мог провести в какой-то строке
    # удаляем нижнюю из этих двух, а верхнюю ячейку расширяем, добавляя высоту нижней
    for j in box:
        for i in box:
            if i != j and j[0] > i[0] and j[1] > 1.05 * i[1] and mistake(j[1] + j[3], i[1] + i[3]):
                for k in box:
                    if k != j and mistake(k[0], j[0]) and mistake(k[1] + k[3], j[1]):
                        tmp = k
                        tmp[3] += j[3] + 4
                        box = [tmp if x == k else x for x in box]
                        break
                box.remove(j)
                break

    # рисуем рамки контуров на фото
    for j in box:
        image = cv2.rectangle(origin, (j[0], j[1]), (j[0] + j[2], j[1] + j[3]), (0, 255, 0), width // 400)
    cv2.imwrite("Detections/sells.jpg", image)
    # выводим картинку с прорисованными рамками контуров на фото
    plt.imshow(image)
    plt.show()
    return box


def rows(box, boundingBoxes):
    # функция разбивает список рамок на ячейки по строкам

    # список высот рамок контуров
    heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
    # получаем среднее значение этих высот
    mean = np.mean(heights)

    # До тех пор, пока поле не отличается больше, чем его собственное (высота + среднее значение / 2),
    # поле находится в том же ряду. Как только разница в высоте становится больше текущей (высота + среднее значение /2)
    # мы знаем, что начинается новая строка. Столбцы логически расположены слева направо.
    row = [] # список нижних строк (каждая ячееки из разных столбцов)
    column = [] # список всех ячеек по строкам
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
    print(row)
    return row


def rows_sort(row, countcol, center):
    # функция сортирует ячейки в правильном порядке(слева направо) относительно расстояния до центра столбцов

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
    return finalboxes


def recognition(finalboxes, bitnot):
    # функци распознаёт значения на каждой ячейке и складывает в список

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
                    out = pytesseract.image_to_string(erosion, lang='rus')
                    if len(out) == 0:
                        # специальная конфигурация для считывания одиночных символов
                        out = pytesseract.image_to_string(erosion, lang='rus', config='--psm 10 --oem 1 -c tessedit_char_whitelist=123456789+-')
                    inner = inner + " " + out
                outer.append(inner)
    return outer


def data_to_excel(outer, row, countcol):
    # функция записывает список распознанных в ячейках таблицы значений в excel

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
