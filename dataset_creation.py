# Запустите этот код для получения датасета обучения нейросети
# Полученные данные будут записаны в папку data в директории программы

import os
import numpy as np
import pandas as pd
import cv2
import struct
from PIL import Image, ImageDraw
from xml.etree import ElementTree as eltree
from tqdm import tqdm

# Путь к папке с данными для обучения нейросети
data_dir = r"D:\3. ML Project\TF Programm\marmot_dataset_v1.0\data"


def hex2double(hexcode):
    """
    This function takes a 16 digit hexadecimal representation and
    convert it into its equivalent floating point form
    """
    return struct.unpack('!d', bytes.fromhex(hexcode))[0]


def return_bboxes_v1(xml, width, height):
    """
    Input: Annotated xml file of an image from the marmot v1.0 dataset
    Output: The list of bounding boxes of tables present in the image
    """
    xml_file = eltree.parse(xml)
    root = xml_file.getroot()
    left, bottom, right, top = (hex2double(i) for i in root.get('CropBox').split())
    pa_width = np.abs(left - right)
    pa_height = np.abs(top - bottom)
    bboxes = []
    for each_table_comp in xml_file.findall(".//Composite[@Label='Table']"):
        t_left, t_bottom, t_right, t_top = [hex2double(i) for i in each_table_comp.get('BBox').split()]
        t_left = int(width / pa_width * np.abs(t_left - left))
        t_right = int(width / pa_width * np.abs(t_right - left))
        t_bottom = int(height / pa_height * np.abs(t_bottom - bottom))
        t_top = int(height / pa_height * np.abs(t_top - bottom))
        bboxes.append([t_left, t_top, t_right, t_bottom])
    return bboxes


def return_bboxes_extended(xml):
    """
    Input: Annotated xml file of an image from the marmot extended dataset
    Output: Dimension of the image and the list of bounding boxes of table columns present in the image
    """
    xml_file = eltree.parse(xml)
    root = xml_file.getroot()
    bboxes = []
    width = int(root.findall('size')[0].findall('width')[0].text)
    height = int(root.findall('size')[0].findall('height')[0].text)
    for ele in root.findall('object'):
        if ele.find('name').text == 'column':
            bbox = ele.find('bndbox')
            left = int(bbox.find('xmin').text)
            right = int(bbox.find('xmax').text)
            top = int(bbox.find('ymin').text)
            bottom = int(bbox.find('ymax').text)
        bboxes.append([left, top, right, bottom])
    return width, height, bboxes


def crop_table(image_path, list_of_bboxes):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    i = 1
    for each_bbox in list_of_bboxes:
        left, top, right, bottom = (int(i) for i in each_bbox)
        table_image = image[bottom:top, left:right]
        cv2.imshow("Table", table_image)
        cv2.waitKey(0)
        i = i + 1
    return None


def create_mask(img_path, list_of_bboxes, dim, mode='table'):
    """
    Input: Image, its corresponding list of bounding boxes, dimension of the image
    and mode specifying the annotation type (table/column)
    Output: Create corresponding mask image (containing the boxes) and return the mask file path
    """
    data_dir = os.path.join('data', 'marmot_masked')
    image_dir = os.path.join(data_dir, 'images')
    mask_dir = os.path.join(data_dir, 'masks', mode)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
    fname = img_path.split(os.path.sep)[-1].replace('bmp', 'png')
    dest = os.path.join(image_dir, fname)
    if mode == 'table':
        img = Image.open(img_path)
        img.save(dest, 'png')
    image = Image.new("RGB", dim)
    mask = ImageDraw.Draw(image)
    for each_list in list_of_bboxes:
        mask.rectangle(each_list, fill=255)
    mask_fn = os.path.join(mask_dir, fname)
    image = np.array(image)
    image = Image.fromarray(image[:, :, 0])
    image.save(mask_fn)
    if mode == 'table':
        return dest, mask_fn
    return mask_fn


def compute_masks(images_dir):
    """
    Returns a pandas dataframe containing the imagepath and mask_path and dimension.
    It invokes the above two methods to create corresponding mask images
    """
    images_dir_ext = os.path.join(images_dir, 'marmot_data_ext')
    # images_dir = os.path.join('data', 'marmot_extended')
    img_file_list = [i for i in os.listdir(images_dir_ext) if i.endswith('bmp')]
    img_df = pd.DataFrame(columns=['image_path', 'image_dim', 'tablemask_path', 'columnmask_path'])
    for each_image in tqdm(img_file_list):
        img_path = os.path.join(images_dir_ext, each_image)
        table_annot = os.path.join(images_dir, 'English', 'Positive', 'Labeled', each_image.replace('bmp', 'xml'))
        column_annot = img_path.replace('bmp', 'xml')
        if not os.path.exists(column_annot):
            continue
        width, height, column_bboxes = return_bboxes_extended(column_annot)
        dim = (width, height)
        table_bboxes = return_bboxes_v1(table_annot, width, height)
        image_path, table_mask_path = create_mask(img_path, table_bboxes, dim, mode='table')
        column_mask_path = create_mask(img_path, column_bboxes, dim, mode='column')
        img_df.loc[len(img_df.index)] = [image_path, dim, table_mask_path, column_mask_path]
        csv_fname = 'data.csv'
        img_df.to_csv(csv_fname, index=False)
    return img_df


img_dataframe = compute_masks(data_dir)
