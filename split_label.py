import os
import  shutil

def split_label():

    for folder in ['test','train','val']:
        for file in os.listdir(os.path.join('data',folder,'images')):
            filename = file.split('.')[0]+'.json'

            if True:
                new_filepath = os.path.join('data',folder,'labels',filename)
                existing = f'data/labels/{filename}'

                if os.path.exists(new_filepath):

                    print('Found: ',new_filepath)

                else:
                     print('copying......... ',existing,"> -------TO----------->",new_filepath)

                     shutil.copy(existing,new_filepath)

                     print("--------------------DONE COPYING-----------------------")

if __name__ == '__main__':

    split_label()