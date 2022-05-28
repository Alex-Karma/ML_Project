import pandas as pd
from PIL import Image
import os
import re
import csv
import cv2
from time import strftime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def visualize(image):
    plt.figure(figsize=(15, 15))
    title = 'Cropped Table'
    plt.title(title)
    plt.imshow(tf.keras.preprocessing.image.array_to_img(image))
    plt.axis('off')
    plt.show()


def render(mask):
    mask = tf.argmax(mask, axis=-1)
    mask = mask[..., tf.newaxis]
    return mask[0]


# extract function
def adjust(new_rows, maxi):
    """
    A function to set all with maxi number of columns
    for making csv compatible
    """
    rows = []
    for each_row in new_rows:
        if len(each_row) < maxi:
            for i in range(maxi - len(each_row)):
                each_row.append("-")
        rows.append(each_row)
    return rows


# transforms a text to csv
def text_to_excel(text):
    """
    This funtion transorms a text with newline and spaces to
    a csv that treats the spaces in the text as comma and newlines as carriage return
    """
    rows = text.split('\n')
    new_rows = []
    maxi = 0
    for each_row in rows:
        temp_row = each_row.split()
        if maxi < len(temp_row):
            maxi = len(temp_row)
        new_rows.append(temp_row)
    new_rows = adjust(new_rows, maxi)

    # запись данных в excel
    dataframe = pd.DataFrame(new_rows)
    print(dataframe)
    # data = dataframe.style.set_properties(align="left")
    writer = pd.ExcelWriter('output.xlsx')
    dataframe.to_excel(writer)
    writer.save()

    header = ['column_{}'.format(i) for i in range(maxi)]
    tstr = strftime("%Y%m%d-%H%M")
    temp_dir = os.path.join('output', 'temporary_files')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    temp_file = os.path.join(temp_dir, 'temp_{}.csv'.format(tstr))

    with open(temp_file, 'w') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(header)
        csvwriter.writerows(new_rows)

    return temp_file


def append_offset(name, offset):
    """
    This function is used for assigning a name with offset if a file with the same name exists
    It takes a filename and a offset and returns a valid equivalent name with offset number

    Example :
    # assume two variables
    name = 'python.py'
    offset = '2'
    append_offset(name, offset)

    # The above invocation will return string as
    # 'python_2.py'
    """
    fname, extension = name.split('.')
    fname = ''.join([fname, '_', offset, '.', extension])
    return fname


def extract_table(image_fname, interpreter, show_table=True):
    # interpreter.allocate_tensors()
    image_orig = Image.open(image_fname)
    original_dim = image_orig.size
    image = image_orig.resize((512, 512))
    np_imag = np.asarray(image) / 255.0

    np_image = np_imag
    if np_imag.shape[2] == 4:
        np_image = np_imag[:, :, :-1]
    np_image = np_image.astype(np.float32)
    np_image = np.expand_dims(np_image, axis=0)

    # ip_d = interpreter.get_input_details()[0]
    # op_d = interpreter.get_output_details()[0]

    # interpreter.set_tensor(ip_d['index'], np_image)
    # interpreter.invoke()

    # tab_mask = interpreter.get_tensor(op_d['index'])

    tab_mask, _ = model.predict(np_image)

    tab_mask = np.squeeze(render(tab_mask).numpy())

    tab_mask = Image.fromarray(np.uint8(tab_mask))
    tab_mask = tab_mask.resize(original_dim)

    tab_mask = np.array(tab_mask)

    image_orig = image_orig
    x, y, w, h = cv2.boundingRect(tab_mask)
    tab = image_orig.crop((x, y, x + w, y + h))

    text = pytesseract.image_to_string(tab, lang='rus')
    text = text.strip()
    text = re.sub("[\r\n]+", "\r\n", text)
    csv = text_to_excel(text)
    csv_fname = image_fname.split(os.path.sep)[-1].replace('png', 'csv')
    dest_dir = os.path.join('output_dir')
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    dest = os.path.join(dest_dir, csv_fname)
    # if file already exists in the temp directory it will save the csv
    # by appending some offset to the filename before extension
    try:
        os.rename(csv, dest)
    except:
        f_save = 'fail'
        i = 2
        while (f_save == 'fail'):
            name_off = str(i)
            try:
                dest = os.path.join(dest_dir, append_offset(csv_fname, name_off))
                os.rename(csv, dest)
                f_save = 'pass'
            except:
                i += 1

    if show_table:
        visualize(tab)

    return dest


model_path = 'final_model'
model = tf.keras.models.load_model(model_path)
img_filepath = r"test_photos/1.3.png"
extract_table(img_filepath, model)


'''
a = np.arange(27).reshape(3, 3, 3)
print(a)
print(a.shape)
b = a[:, :, :-1]
print(b.shape)
'''
