# InpaintGAN

## Introduction
This paper presents a novel approach to the image inpainting problem using a Generative Adversarial Network (GAN) architecture. We undertook a formal optimization process on an existing EdgeConnect model to improve its performance in a specific domain.

The method proposed in this paper achieved a precision of 28%, recall of 25% and, feature matching loss of 25% while keeping the number of iterations at bare minimum of 175,000. In contrast, EdgeConnect [1] achieved precision of 27%, recall of 25% and, feature matching loss of 45%.
Our model can be useful for various image editing tasks such as image completion, object removal and image restoration, where fine details are crucial. Our approach addresses the limitation of current state-of-the-art models in producing fine detailed images and opens new possibilities for image inpainting applications.
The book is aimed at researchers, practitioners, and students in the fields of computer vision, image processing, and deep learning who are interested in learning about the latest advancements in image inpainting and how to improve current methods.

## Prerequisites
- Python 3
- PyTorch 1.13
- Eel 0.16
- scikit-image
- opencv 4.5.4.60
### Optional
- The development was utlizing PyCharm IDE by JetBrains

## Installation
- Cloning repo
```bash
git clone https://github.com/MoshPe/InpaintGAN.git
cd InpaintGAN
```
- Install Python dependencies
```bash
pip install -r requirements.txt
```
## Dataset
We use [HumanFaces](https://www.kaggle.com/datasets/ashwingupta3012/human-faces) datasets. To train the model on the dataset please download the dataset from Kaggle website with your registered user.
> **Warning** The dataset must contain only images with any type of jpg, jpeg, png and the folder path Must be in english !!!

## Getting Started
### Training
To train the model simply run `main.py` to open up Eel GUI to operate in system.
- Head over to train tab and configure the model.
  
  ![image](https://github.com/MoshPe/InpaintGAN/assets/57846100/36cf5d79-ee75-42f7-b793-acd91727536f)

- Select what model to train, method of masking an image and which edge detection model to utilize
  
  ![image](https://github.com/MoshPe/InpaintGAN/assets/57846100/2ba26472-c244-4b96-9505-b9d7a73a732c)

- Hit next for further configuring the model

  ![image](https://github.com/MoshPe/InpaintGAN/assets/57846100/7bc986c8-4c95-40ae-bbdb-bcf02d9e3cde)

- In the following section is the configuration of both generator and discriminator for Edge and Inpaint models accordingly.

    Option                 |Default| Description
  -----------------------|-------|------------
  LR                     | 0.0001| learning rate
  D2G_LR                 | 0.1   | discriminator/generator learning rate ratio
  BETA1                  | 0.0   | adam optimizer beta1
  BETA2                  | 0.9   | adam optimizer beta2
  BATCH_SIZE             | 8     | input batch size 
  INPUT_SIZE             | 256   | input image size for training. (0 for original size)
  SIGMA                  | 2     | standard deviation of the Gaussian filter used in Canny edge detector </br>(0: random, -1: no edge)
  MAX_ITERS              | 2e6   | maximum number of iterations to train the model
  EDGE_THRESHOLD         | 0.5   | edge detection threshold (0-1)
  L1_LOSS_WEIGHT         | 1     | l1 loss weight
  FM_LOSS_WEIGHT         | 10    | feature-matching loss weight
  STYLE_LOSS_WEIGHT      | 1     | style loss weight
  CONTENT_LOSS_WEIGHT    | 1     | perceptual loss weight
  INPAINT_ADV_LOSS_WEIGHT| 0.01  | adversarial loss weight
  GAN_LOSS               | nsgan | **nsgan**: non-saturating gan, **lsgan**: least squares GAN, **hinge**: hinge loss GAN

  ![image](https://github.com/MoshPe/InpaintGAN/assets/57846100/b5172cec-17c5-4572-aa10-41c00b33c4e0)

  

- Running and training the model.

  https://github.com/MoshPe/InpaintGAN/assets/57846100/703132af-9ddc-47d1-8d14-a1c70bd1b163

### Inference
In this step the model is utilized for testing. Uploading and image, creating a mask on the uploaded image and run the model on the masked image.

#### 1. Open Inference tab
  ![image](https://github.com/MoshPe/InpaintGAN/assets/57846100/b9f3b822-2d19-4ef2-ba91-c89a253e7795)

#### 2. Upload Image
  ![image](https://github.com/MoshPe/InpaintGAN/assets/57846100/8e804cca-f38e-43db-b5ba-899230cec15b)
  ![image](https://github.com/MoshPe/InpaintGAN/assets/57846100/c9d09519-4294-4823-b00f-f46c5f56aaf7)

#### 3. Mask image
  In this section the user is able to draw the mask onto the image for the model to fill.<br>
  The user can choose between several thicknesses of lines to draw and clear the drawn lines.
  > **Warning** The model was trained upon a square mask so different kind of drawing might won't return the expected results.

![image](https://github.com/MoshPe/InpaintGAN/assets/57846100/b82da3b3-bf4b-43ed-bdd4-c51a071f8d91)
![image](https://github.com/MoshPe/InpaintGAN/assets/57846100/790200ea-295b-4d82-a26e-ce3accaf99c9)

#### 4. Fill missing regions
  ![image](https://github.com/MoshPe/InpaintGAN/assets/57846100/e1ad3628-b91c-4e72-8976-61b390bfceec)

## Credit
Created and used with the help of Edge-LBAM article and github repo implementation. <br>
<a href="https://github.com/wds1998/Edge-LBAM">Image Inpainting with Edge-guided Learnable Bidirectional Attention Maps</a><br>
Created and used with the help of EdgeConnect article and github repo implementation. <br>
<a href="https://arxiv.org/abs/1901.00212">EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning</a><br>
<a href="http://openaccess.thecvf.com/content_ICCVW_2019/html/AIM/Nazeri_EdgeConnect_Structure_Guided_Image_Inpainting_using_Edge_Prediction_ICCVW_2019_paper.html">EdgeConnect: Structure Guided Image Inpainting using Edge Prediction</a>:

```
@inproceedings{nazeri2019edgeconnect,
  title={EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning},
  author={Nazeri, Kamyar and Ng, Eric and Joseph, Tony and Qureshi, Faisal and Ebrahimi, Mehran},
  journal={arXiv preprint},
  year={2019},
}

@InProceedings{Nazeri_2019_ICCV,
  title = {EdgeConnect: Structure Guided Image Inpainting using Edge Prediction},
  author = {Nazeri, Kamyar and Ng, Eric and Joseph, Tony and Qureshi, Faisal and Ebrahimi, Mehran},
  booktitle = {The IEEE International Conference on Computer Vision (ICCV) Workshops},
  month = {Oct},
  year = {2019}
}
```
