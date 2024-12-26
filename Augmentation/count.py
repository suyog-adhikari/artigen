import os

dir = "../Dataset"
count_dict = {}

for subdir in os.listdir(dir):
    temp_dict = {}
    subdir_path = os.path.join(dir, subdir)

    for sub_subdir in os.listdir(subdir_path):
        sub_subdir_path = os.path.join(subdir_path, sub_subdir)
        file_count = sum(1 for item in os.listdir(sub_subdir_path) if os.path.isfile(os.path.join(sub_subdir_path, item)))
        temp_dict[sub_subdir] = file_count
    
    count_dict[subdir] = temp_dict

print(f'-----------------\n| Image Counts: |\n-----------------')
for key in count_dict:
    print(f'{key}:')
    print(count_dict[key])
