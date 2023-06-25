from Config import Config
from inpaintGAN import InpaintGAN


if __name__ == '__main__':
    inpaintConfig = Config()

    inpaintGAN = InpaintGAN(inpaintConfig)
    inpaintGAN.train()