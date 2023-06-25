import os
import numpy as np
import torch
from PIL.Image import Image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms as T
import tensorflow as tf
import torchvision.transforms.functional as F


# from .dataset import Dataset
# from .models import EdgeModel, InpaintingModel
# from .utils import Progbar, create_dir, stitch_images, imsave
# from .metrics import PSNR, EdgeAccuracy
from inpaintGAN.dataset import Dataset
from inpaintGAN.metrics import EdgeAccuracy, PSNR
from inpaintGAN.models import EdgeModel, InpaintingModel
from inpaintGAN.utils import Progbar, create_dir, stitch_images


class InpaintGAN():
    def __init__(self, config, folder_path):
        self.config = config
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)
        if torch.cuda.is_available():
            config.DEVICE = torch.device("cuda")
            torch.backends.cudnn.benchmark = True  # cudnn auto-tuner
        else:
            config.DEVICE = torch.device("cpu")
        if config.MODEL == 1:
            model_name = 'edge'
        elif config.MODEL == 2:
            model_name = 'inpaint'
        elif config.MODEL == 3:
            model_name = 'edge_inpaint'
        elif config.MODEL == 4:
            model_name = 'joint'
        self.debug = True
        self.model_name = model_name
        # detect and init the TPU
        #         tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
        #         # instantiate a distribution strategy
        #         tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
        #         # instantiating the model in the strategy scope creates the model on the TPU
        #         with tpu_strategy.scope():
        self.edge_model = EdgeModel(config).to(config.DEVICE)
        self.inpaint_model = InpaintingModel(config).to(config.DEVICE)

        self.edgeacc = EdgeAccuracy(config.EDGE_THRESHOLD).to(config.DEVICE)
        self.psnr = PSNR(255.0).to(config.DEVICE)
        if self.config.MODE != 4:
            if self.config.MODE == 2:
                self.test_dataset = Dataset(256, config, '/kaggle/input/train-test-set/val_256/', augment=False,
                                            training=True, isVal=True)
            else:
                self.train_dataset = Dataset(config, 256, folder_path, augment=True, training=True, isVal=False, facesDataSet=True)
                self.val_dataset = Dataset(config, 256, folder_path, augment=False, training=True, isVal=True, facesDataSet=True)
                self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)

        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')

        self.log_file = os.path.join(config.PATH, 'log_' + model_name + '.dat')

    def train(self, setMetrics, addLog):
        # Train
        epoch = 0
        keep_training = True
        model = self.config.MODEL
        max_iteration = int(float((self.config.MAX_ITERS)))

        total = len(self.train_dataset)

        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=4,
            drop_last=True,
            shuffle=True
        )

        # self.load()
        while (keep_training):
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)
            addLog('Training epoch: ' + str(epoch))
            progbar = Progbar(total, width=20, verbose=1, stateful_metrics=['epoch', 'iter'])
            for items in train_loader:

                self.edge_model.train()
                self.inpaint_model.train()
                images, images_gray, edges, masks = self.cuda(*items)
                # train
                outputs, gen_loss, dis_loss, logs = self.edge_model.process(images_gray, edges, masks)

                # metrics
                precision, recall = self.edgeacc(edges * masks, outputs * masks)
                logs.append(('precision', precision.item()))
                logs.append(('recall', recall.item()))
                setMetrics(str(precision.item()), str(recall.item()))
                # backward
                self.edge_model.backward(gen_loss, dis_loss)

                outputs, gen_loss, dis_loss, logsInpaint = self.inpaint_model.process(images, edges, masks)
                logs += logsInpaint
                outputs_merged = (outputs * masks) + (images * (1 - masks))

                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))

                # backward
                self.inpaint_model.backward(gen_loss, dis_loss)
                iteration = self.inpaint_model.iteration

                if iteration >= max_iteration:
                    keep_training = False
                    break

                logs = [
                           ("epoch", epoch),
                           ("iter", iteration),
                       ] + logs

                progbar.add(addLog, len(images),
                            values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])

                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs)

                # sample model at checkpoints
                if self.config.SAMPLE_INTERVAL and iteration % self.config.SAMPLE_INTERVAL == 0:
                    self.sample()

                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                    self.save()

            print('\nEnd training....')

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def load(self):
        if self.config.MODEL == 1:
            self.edge_model.load()

        elif self.config.MODEL == 2:
            self.inpaint_model.load()

        else:
            self.edge_model.load()
            self.inpaint_model.load()

    def save(self):
        if self.config.MODEL == 1:
            self.edge_model.save()

        elif self.config.MODEL == 2 or self.config.MODEL == 3:
            self.inpaint_model.save()

        else:
            self.edge_model.save()
            self.inpaint_model.save()

    def log(self, logs, addLog):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()

    def eval(self):
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.BATCH_SIZE,
            drop_last=True,
            shuffle=True
        )

        model = self.config.MODEL
        total = len(self.val_dataset)

        self.edge_model.eval()
        #         self.inpaint_model.eval()

        progbar = Progbar(total, width=20, stateful_metrics=['it'])
        iteration = 0

        for items in val_loader:
            iteration += 1
            images, images_gray, edges, masks = self.cuda(*items)

            # edge model
            if model == 1:
                # eval
                outputs, gen_loss, dis_loss, logs = self.edge_model.process(images_gray, edges, masks)

                # metrics
                precision, recall = self.edgeacc(edges * masks, outputs * masks)
                logs.append(('precision', precision.item()))
                logs.append(('recall', recall.item()))
        logs = [("it", iteration), ] + logs
        progbar.add(len(images), values=logs)

    def fill_image(self, size):
        augment = True
        self.edge_model.eval()
        self.inpaint_model.eval()

        fill_dataset = Dataset(self.config, 256, 'web/inference/image.jpeg', augment=False, training=True, isVal=True, facesDataSet=False,
                               isSpecific=True, mask_flist='web/inference/mask.jpeg')
        dataset_iterator = fill_dataset.create_iterator(1)
        items = next(dataset_iterator)
        images, images_gray, edges, masks = self.cuda(*items)
        outputs = self.edge_model(images_gray, edges, masks).detach()
        edges = (outputs * masks + edges * (1 - masks)).detach()
        outputs = self.inpaint_model(images, edges, masks)
        outputs_merged = (outputs * masks) + (images * (1 - masks))

        return outputs_merged

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def test(self):
        self.edge_model.eval()
        self.inpaint_model.eval()

        model = self.config.MODEL
        create_dir(self.results_path)

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )

        index = 0
        for items in test_loader:
            name = self.test_dataset.load_name(index)
            images, images_gray, edges, masks = self.cuda(*items)
            index += 1

            # edge model
            if model == 1:
                outputs = self.edge_model(images_gray, edges, masks)
                outputs_merged = (outputs * masks) + (edges * (1 - masks))

            # inpaint model
            elif model == 2:
                outputs = self.inpaint_model(images, edges, masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

            # inpaint with edge model / joint model
            else:
                inputs = (images * (1 - masks)) + masks
                edges = self.edge_model(images_gray, edges, masks).detach()
                outputs = self.inpaint_model(images, edges, masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

            output = self.postprocess(outputs_merged)[0]

            images = stitch_images(
                self.postprocess(images),
                self.postprocess(inputs),
                self.postprocess(edges),
                self.postprocess(outputs_merged)
            )

            path = os.path.join(self.results_path, name)
            print(index, name)
            images.save(path)

        print('\nEnd test....')

    def sample(self, it=None):
        # do not sample when validation set is empty
        if len(self.val_dataset) == 0:
            return

        self.edge_model.eval()
        self.inpaint_model.eval()

        model = self.config.MODEL
        items = next(self.sample_iterator)
        images, images_gray, edges, masks = self.cuda(*items)

        # edge model
        if model == 1:
            iteration = self.edge_model.iteration
            inputs = (images_gray * (1 - masks)) + masks
            outputs = self.edge_model(images_gray, edges, masks)
            outputs_merged = (outputs * masks) + (edges * (1 - masks))

        # inpaint model
        elif model == 2:
            iteration = self.inpaint_model.iteration
            inputs = (images * (1 - masks)) + masks
            outputs = self.inpaint_model(images, edges, masks)
            outputs_merged = (outputs * masks) + (images * (1 - masks))

        # inpaint with edge model / joint model
        else:
            iteration = self.inpaint_model.iteration
            inputs = (images * (1 - masks)) + masks
            outputs = self.edge_model(images_gray, edges, masks).detach()
            edges = (outputs * masks + edges * (1 - masks)).detach()
            outputs = self.inpaint_model(images, edges, masks)
            outputs_merged = (outputs * masks) + (images * (1 - masks))

        if it is not None:
            iteration = it

        image_per_row = 2
        if self.config.SAMPLE_SIZE <= 6:
            image_per_row = 1

        images = stitch_images(
            self.postprocess(images),
            self.postprocess(inputs),
            self.postprocess(edges),
            self.postprocess(outputs),
            self.postprocess(outputs_merged),
            img_per_row=image_per_row
        )

        path = os.path.join(self.samples_path, self.model_name)
        name = os.path.join(path, str(iteration).zfill(5) + ".png")
        create_dir(path)
        print('\nsaving sample ' + name)
        images.save(name)
