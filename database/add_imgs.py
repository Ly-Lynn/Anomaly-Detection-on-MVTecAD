import sqlite3
import os
import base64
from tqdm import tqdm

DB_NAME = 'app.db'
conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS images_test (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path TEXT NOT NULL,
    groundtruth_path TEXT NOT NULL,
    label INTEGER NOT NULL,
    data_module TEXT NOT NULL,
    image_base64 TEXT NOT NULL
)
''')

conn.commit()

def save_images_to_db(image_path, data_module, label, groundtruth_path):
    with open(image_path, 'rb') as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

    cursor.execute('INSERT INTO images_test (data_module, groundtruth_path, label, image_path, image_base64) VALUES (?, ?, ?, ?, ?)',
                    (data_module, groundtruth_path, label, image_path, image_base64))
    
    conn.commit()

list_images = []
root_dir = r'D:\codePJ\UIT\CS331\lastterm\final-project-backend\datasets\MVTec'
modules = os.listdir(root_dir)

for module in tqdm(modules):
    module_path = os.path.join(root_dir, module)
    if not os.path.isdir(module_path):  # Skip if it's not a directory
        continue
    test_root = os.path.join(module_path, 'test')
    if not os.path.isdir(test_root):  # Skip if 'test' directory is missing
        continue
    for test_folder in os.listdir(test_root):
        test_folder_root = os.path.join(test_root, test_folder)
        if test_folder == 'good':
            label = 0
        else:
            label = 1
        if not os.path.isdir(test_folder_root):  # Skip if not a directory
            continue
        for img in os.listdir(test_folder_root):
            img_path = os.path.join(test_folder_root, img)
            if test_folder == 'good':
                groundtruth_path = ""
            else: 
                groundtruth_path = os.path.join(module_path, 'ground_truth', test_folder, img.split('.')[0] + '_mask.png')
            if os.path.isfile(img_path):  # Ensure the item is a file (not another directory)
                list_images.append({
                    "module": module,
                    "img_path": img_path,
                    "label": label,
                    "groundtruth_path": groundtruth_path
                })

print('Add images into database\n')

for image_data in tqdm(list_images):
    save_images_to_db(image_path= image_data['img_path'], 
                    data_module= image_data['module'], 
                    label= image_data['label'],
                    groundtruth_path= image_data['groundtruth_path'])

conn.close()
