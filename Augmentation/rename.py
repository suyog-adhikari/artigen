import os

dir = '../Dataset'

rename_map = {
    '7': 'Cat',
    '8': 'Dog',
    '9': 'Horse',
    '10': 'Sheep',
    '11': 'Cow',
    '12': 'Elephant',
    '13': 'Zebra',
    '14': 'Giraffe'
}

for subdir in os.listdir(dir):
    subdir_path = os.path.join(dir, subdir)

    for sub_subdir in os.listdir(subdir_path):
        sub_subdir_path = os.path.join(subdir_path, sub_subdir)

        if os.path.isdir(sub_subdir_path) and sub_subdir in rename_map:
            new_subdir_path = os.path.join(subdir_path, rename_map[sub_subdir])
            os.rename(sub_subdir_path, new_subdir_path)
            print(f'Renamed "{subdir}" to "{new_subdir_path}"')
        else:
            print(f'"{sub_subdir_path}" not renamed')