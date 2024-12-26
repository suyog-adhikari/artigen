import os
import shutil

inp_main = '../Dataset/Aug'
out_main = '../Dataset'

for dir in os.listdir(inp_main):
    dir_path = os.path.join(inp_main, dir)
    
    for subdir in os.listdir(dir_path):
        subdir_path = os.path.join(dir_path, subdir)

        files = [f for f in os.listdir(subdir_path) if f.endswith(".png") or f.endswith(".jpg")]

        destination_path = os.path.join(out_main, dir)
        destination_path = os.path.join(destination_path, subdir)

        for file in files:
            source_path = os.path.join(subdir_path, file)    
            shutil.move(source_path, destination_path)
        
        print(f"All file in {dir}/{subdir} moved")

