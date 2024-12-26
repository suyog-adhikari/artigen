import os
import cv2
import numpy as np
import random

def apply_augmentation(image, rotation_angle, scale_factor, brightness_factor=1.0, max_noise=1):
    # Random rotation
    rows, cols, _ = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), rotation_angle, scale_factor)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows), borderValue=(255, 255, 255))

    # Gaussian noise
    noise = np.random.normal(0, max_noise, rotated_image.shape).astype(np.uint8)
    noisy_image = cv2.add(rotated_image, noise)

    return noisy_image

def resize_image(image, target_size=(256, 256)):
    return cv2.resize(image, target_size)

def augment_dataset(input_directory, output_directory):
    # List all subdirectories in the main dataset directory
    #inp_dir = os.path.join(input_directory, 'Sketch')
    # class_directories = os.listdir(input_directory)

    sketch_class_path_main = os.path.join(input_directory, 'Sketch')
    GT_class_path_main = os.path.join(input_directory, 'GT')
    edge_class_path_main = os.path.join(input_directory, 'Edge')

    sketch_output_class_directory_main = os.path.join(output_directory, 'Sketch')
    GT_output_class_directory_main = os.path.join(output_directory, 'GT')
    edge_output_class_directory_main = os.path.join(output_directory, 'Edge')
    
    for j in os.listdir(sketch_class_path_main):
        print(f"{str(j)}: Augmentation in progess.....")
        sketch_class_path = os.path.join(sketch_class_path_main, str(j))
        GT_class_path = os.path.join(GT_class_path_main, str(j))
        edge_class_path = os.path.join(edge_class_path_main, str(j))

        sketch_output_class_directory = os.path.join(sketch_output_class_directory_main, str(j))
        GT_output_class_directory = os.path.join(GT_output_class_directory_main, str(j))
        edge_output_class_directory = os.path.join(edge_output_class_directory_main, str(j))

        # Create the output directory for the class if it doesn't exist
        if not os.path.exists(sketch_output_class_directory):
            os.makedirs(sketch_output_class_directory)
        if not os.path.exists(GT_output_class_directory):
            os.makedirs(GT_output_class_directory)
        if not os.path.exists(edge_output_class_directory):
            os.makedirs(edge_output_class_directory)

        # List all image files in the class directory
        sketch_files = [f for f in os.listdir(sketch_class_path) if f.endswith(".png") or f.endswith(".jpg")]
        GT_files = [f for f in os.listdir(GT_class_path) if f.endswith(".png") or f.endswith(".jpg")]
        edge_files = [f for f in os.listdir(edge_class_path) if f.endswith(".png") or f.endswith(".jpg")]

        # Calculate oversampling factor for this class
        class_size = len(sketch_files)
        oversample_factor=3917/class_size
        oversample_count = int((class_size * oversample_factor)-class_size)
        for i in range(oversample_count):
            # Randomly select an original image
            #original_sketch_file = random.choice(sketch_files)
            file = random.choice(sketch_files)

            original_sketch_path = os.path.join(sketch_class_path, file)
            #original_GT_file = random.choice(GT_files)
            original_GT_path = os.path.join(GT_class_path, file)
            #original_edge_file = random.choice(edge_files)
            original_edge_path = os.path.join(edge_class_path, file)

            # Read the original image
            original_sketch = cv2.imread(original_sketch_path)
            original_GT = cv2.imread(original_GT_path)
            original_edge = cv2.imread(original_edge_path)

            # Resize the image to the target size
            original_sketch = resize_image(original_sketch)
            original_GT = resize_image(original_GT)
            original_edge = resize_image(original_edge)

            # Random rotation angle and scale factor
            rotation_angle = random.uniform(-30, 30)
            scale_factor = random.uniform(0.8, 1.2)

            # Apply augmentation to the resized image
            augmented_sketch = apply_augmentation(original_sketch, rotation_angle, scale_factor)
            augmented_GT = apply_augmentation(original_GT, rotation_angle, scale_factor)
            augmented_edge = apply_augmentation(original_edge, rotation_angle, scale_factor)

            # Resize the augmented image to the target size
            augmented_sketch = resize_image(augmented_sketch)
            augmented_GT = resize_image(augmented_GT)
            augmented_edge = resize_image(augmented_edge)

            # Save the augmented image to the corresponding output class directory
            output_sketch_path = os.path.join(sketch_output_class_directory, f"{i}_{file}")
            cv2.imwrite(output_sketch_path, augmented_sketch)
            output_GT_path = os.path.join(GT_output_class_directory, f"{i}_{file}")
            cv2.imwrite(output_GT_path, augmented_GT)
            output_edge_path = os.path.join(edge_output_class_directory, f"{i}_{file}")
            cv2.imwrite(output_edge_path, augmented_edge)

        print(f"\t{str(j)} : Augmentation Complete")

if __name__ == "__main__":
    input_directory = '../Dataset/'
    output_directory = '../Dataset/Aug/'

    augment_dataset(input_directory, output_directory)

