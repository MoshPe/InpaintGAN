import eel

from inpaintGAN.Config import Config
from inpaintGAN.inpaintGAN import InpaintGAN
from PIL import Image, ImageTk
from torchvision.transforms import ToPILImage
import base64
from tkinter import filedialog
import tkinter as tk
import os
import imghdr
import cv2

modelConfig = {}

folder_path = "none"

eel.init("web", allowed_extensions=['.js', '.html', '.png', '.txt'])


def is_folder_only_images(f_path):
    for filename in os.listdir(f_path):
        file_path = os.path.join(f_path, filename)
        if os.path.isfile(file_path):
            if imghdr.what(file_path) is None:
                return False
    return True


@eel.expose
def train_model():
    inpaintConfig = Config()
    inpaintConfig.LR = modelConfig["lr"]
    inpaintConfig.D2G_LR = modelConfig["g_d_lr"]
    inpaintConfig.BATCH_SIZE = modelConfig["b_size"]
    inpaintConfig.SIGMA = modelConfig["sigma"]
    inpaintConfig.MAX_ITERS = modelConfig["max_iterations"]

    inpaintConfig.EDGE_THRESHOLD = modelConfig["edge_threshold"]
    inpaintConfig.L1_LOSS_WEIGHT = modelConfig["l1_loss_w"]
    inpaintConfig.FM_LOSS_WEIGHT = modelConfig["fm_loss_w"]
    inpaintConfig.STYLE_LOSS_WEIGHT = modelConfig["style_loss_w"]
    inpaintConfig.CONTENT_LOSS_WEIGHT = modelConfig["content_loss_w"]
    inpaintConfig.INPAINT_ADV_LOSS_WEIGHT = modelConfig["inpaint_adv_loss_w"]

    inpaintGAN = InpaintGAN(inpaintConfig, modelConfig['dataset_path'])
    inpaintGAN.train(eel.setMetrics, eel.addLog)


@eel.expose
def save_model_config(x):
    for key in x:
        try:
            modelConfig[key] = int(x[key])
        except:
            try:
                modelConfig[key] = float(x[key])
            except:
                modelConfig[key] = x[key]
    print(modelConfig)


@eel.expose
def get_dataset_path():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    folder_path = str(filedialog.askdirectory(parent=root))
    eel.writeFolderPath(str(folder_path))
    folder_path += "/"
    print(os.path.isdir(folder_path))


@eel.expose
def test_model():
    inpaintConfig = Config()
    inpaintConfig.MASK = 1
    inpaintConfig.MODE = 4
    inpaintGAN = InpaintGAN(inpaintConfig, "")
    img, edges, masks, outputs = inpaintGAN.fill_image()
    tensor_to_pil = ToPILImage()
    pil_image = tensor_to_pil(img.squeeze())
    pil_image.save("web/inference/test.jpeg")  # Replace "image.jpg" with the desired file path and extension
    pil_image = tensor_to_pil(edges.squeeze())
    pil_image.save("web/inference/edges.jpeg")  # Replace "image.jpg" with the desired file path and extension
    pil_image = tensor_to_pil(masks.squeeze())
    pil_image.save("web/inference/masks_gen.jpeg")  # Replace "image.jpg" with the desired file path and extension
    pil_image = tensor_to_pil(outputs.squeeze())
    pil_image.save("web/inference/outputs.jpeg")  # Replace "image.jpg" with the desired file path and extension


@eel.expose
def download_image_file(url):
    decoded_data = base64.b64decode(url)
    img_file = open('web/inference/image.jpeg', 'wb')
    img_file.write(decoded_data)
    img_file.close()


@eel.expose
def download_mask_file(url):
    decoded_data = base64.b64decode(url)
    img_file = open('web/inference/mask.jpeg', 'wb')
    img_file.write(decoded_data)
    img_file.close()
    image = Image.open('web/inference/mask.jpeg')
    image = image.convert('L')
    image.save("web/inference/mask.jpeg")


if __name__ == '__main__':
    eel.start("index.html")
