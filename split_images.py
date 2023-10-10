import os
import shutil


def split_data():

    data_path = "data/image"

    train_folder = 'data/train/image'
    val_folder =  'data/val/image'
    test_folder = 'data/test/image'

    # Define a list of image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    imgs_list = [filename for filename in os.listdir(data_path) if os.path.splitext(filename)[-1] in image_extensions]


    train_size = int(len(imgs_list) * 0.7)
    val_size = int(len(imgs_list) * 0.20)
    test_size = int(len(imgs_list) * 0.10)

    # Create destination folders if they don't exist
    print(train_folder, val_folder, test_folder)
    for folder_path in [train_folder, val_folder, test_folder]:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(folder_path)

    # Copy image files to destination folders
    for i, f in enumerate(imgs_list):
        if i < train_size:
            dest_folder = train_folder
        elif i < train_size + val_size:
            dest_folder = val_folder
        else:
            dest_folder = test_folder
        shutil.copy(os.path.join(data_path, f), os.path.join(dest_folder, f))

if __name__ == '__main__':

    split_data()