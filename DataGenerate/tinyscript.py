import os

dirname = "./TRAData"
dirlist = os.listdir(dirname)
for index, file in enumerate(dirlist):
    file_path = os.path.join(dirname, file)
    newfile = "1-" + file
    new_file_path = os.path.join(dirname, newfile)
    os.rename(file_path, new_file_path)
