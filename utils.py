import os
import numpy as np
import pandas as pd
import tensorflow as tf


BROKEN_FILES = [
    "39_1_20170116174525125.jpg.chip.jpg",
    "61_1_20170109142408075.jpg.chip.jpg",
    "61_1_20170109150557335.jpg.chip.jpg"
]


def extract_data_from_filename(filepath):
    parts = filepath.split('_')
    age = int(parts[0])
    gender = int(parts[1])
    race = int(parts[2])

    return age, gender, race


def prepare_dataframe(dataset_dir):
    filepaths = []
    ages = []
    genders = []
    ethnicities = []
    age_classes = []

    for file in os.listdir(dataset_dir):
        if file.endswith(".jpg"):
            if file in BROKEN_FILES:
                continue

            filepath = os.path.join(dataset_dir, file)
            filepaths.append(filepath)
            age, gender, race = extract_data_from_filename(file)
            ages.append(age)
            genders.append(gender)
            ethnicities.append(race)
            age_classes.append(age // 10)

    return pd.DataFrame({
        'Image_Path': filepaths,
        'Age': ages,
        'Gender': genders,
        'Ethnicity': ethnicities,
        'Age_Class': age_classes
    })


def load_image_np(filepath, img_dim=200):
    image = tf.io.read_file(filepath)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_dim, img_dim])
    image = tf.cast(image, tf.uint8)
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    return image.numpy()


def df_to_np_arrays(df, img_dim=200):
    images = []
    labels = []
    
    for _, row in df.iterrows():
        filepath = row['Image_Path']
        age_class = row['Age_Class']
        try:
            img = load_image_np(filepath, img_dim=img_dim)
            images.append(img)
            labels.append(age_class)
        except Exception as e:
            print(f"Error loading image at {filepath}: {e}")
    
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels