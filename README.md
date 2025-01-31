# Artigen: Transfigure Images Via Sketch Guided Generative Model
**Artigen** is an application of **Generative Adversarial Networks (GANs)** for transforming hand-drawn sketches into realistic images. Built using the **Pix2Pix architecture**, this project explores new possibilties in sketch-to-image synthesis, bridging the gap between simplistic outlines and photorealistic representations.

## 🚀 Features
* **Sketch-to-Image Synthesis**: Converts sketches into images
* **GAN-Powered Transfromation**: Utilized Conditional GANs for realistic output
* **User-Friendlt GUI**: Simplified Interaction for intuitive image generation
* **Flexible Applications**: From concept art to education and virtual prototyping

## 📖 Abstract
The project leverages **GANs** to transfigure simple sketch contours into visually compelling images. Using paired datasets like **SketchyCOCO**, it trains a model to understand the semantics of sketches, producing outputs with high fidelity and contextual relevance.

## 🛠️ System Architecture
The architecture integrates:
* **Generator Network**: Creates images from latent representations
* **Discriminator Network**: Validates the realism of generated images
* **Pix2Pix Model**: Enhances sketch-image pair learning.


![image](https://github.com/user-attachments/assets/b44cd1f2-2ce9-4427-9204-c0e9d8ea068c)

## 🔧 Requirements
### Hardware
* GPU with at least **8 GB memory**
### Software
* Python 3.10
* Libraries: TensorFlow, NumPy, Matplotlib, Pillow, Tkinter
* Jupyter Notebook *(Preferred IDE)*
* Dataset: [SketchyCOCO](https://github.com/sysu-imsl/SketchyCOCO "SketchyCOCO")

## 🎯 Objectives 
1. Develop a GAN-based module for realistic sketch-to-image translation
2. Provide an interactive and user-friendly tool for image generation

## ⚙️ Implementation Details
![image](https://github.com/user-attachments/assets/de07c5d8-6ba5-4f57-bfc7-f6c3dac4a72e)

Data preprocessing, model architectural design and training procedures with different techniques serve as the core elements of implementation to provide stable and effective training for the model. This section analyzes the core operational components which enable cGAN to produce images from sketches and noise during the generation process.

1. **Data Preprocessing**: The dataset requires resizing and normalization steps followed by augmentation through random cropping to prepare it before training. The processing methods guarantee the model will receive diverse as well as standardized input.

2. **Model Architecture**: The cGAN consists both generator and discriminator components. The generator generates synthetic images through up-sampling and concatenation operations and the discriminator uses down-sampling analysis to differentiate real from synthetic images with the aid of class labeling information.

3. **Training Process**: The training process includes alternative operations for discriminator updates using real and fake images with their labels followed by generator updates through random noisy inputs and class labels. The loss functions include conditional elements which represent class-specific data.

4. **Stability Techniques**: The model incorporates batch normalization alongside gradient penalty to enhance training stability which helps resolve mode collapse and vanishing gradients. The model is evaluated through relevant assessment metrics which measure both image quality and class conditioning performance.

## 📊 Results and Evaluation
* Image synthesis demonstrated with **FID**, **MSE**, and **SSM** metrics.
* Versatile GUI showcasing various object classes (e.g., animals, cars, landscapes)
* Consistent result across diverse sketch inputs

## 🚀 Quick Start
### Installation
```bash
git clone https://github.com/suyog-adhikari/artigen.git
cd artigen

pip install -r requirements.txt
```
### Usage
```bash
python gui.py
```
***
## Examples
### Input Images
![image](https://github.com/user-attachments/assets/3ac09c50-7535-448a-ace7-3535d7bfc1fc)
### Output Images
![image](https://github.com/user-attachments/assets/20e2edc7-8cb3-43eb-b66a-cc8882492694)
***
## 📚 Literature Review
The project builds on prior research in GANs including:
* **SketchyGAN**
* **Pix2Pix Architecture**
* **Variational Autoencoders (VAEs)**

## 🤝 Contributors
* [**Manoj Paudel**](https://github.com/manozpdel "Manoj Paudel")
* [**Roshan Pokharel**](https://github.com/roshan076 "Roshan Pokharel")
* [**Ujjwol Poudel**](https://github.com/ujjwol112 "Ujjwol Poudel")
* **Suyog Adhikari**

**Supervised By:** Er. Umesh Kanta Ghimire, Thapathali Campus, IOE, TU

## 📜 License
This project is licensed under the **MIT License**

## 🌟 Acknowledgements
Special thanks to the **Department of Electronics and Computer Engineering, Thapathali Campus, Institute of Engineering, Tribhuvan University** for providing guidance and resources.
