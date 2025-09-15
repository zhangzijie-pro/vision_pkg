import os
import logging

import cv2
import numpy as np
# import pkg_resources

# from deep_sort_realtime.embedder.mobilenetv2_bottle import MobileNetV2_bottle

logger = logging.getLogger(__name__)

# MOBILENETV2_BOTTLENECK_WTS = pkg_resources.resource_filename(
#     "deep_sort_realtime", "embedder/weights/mobilenetv2_bottleneck_wts.pt"
# )

# TORCHREID_OSNET_AIN_X1_0_MS_D_C_WTS = pkg_resources.resource_filename(
#     "deep_sort_realtime", "embedder/weights/osnet_ain_ms_d_c_wtsonly.pth"
# )

INPUT_WIDTH = 224


# def batch(iterable, bs=1):
#     l = len(iterable)
#     for ndx in range(0, l, bs):
#         yield iterable[ndx : min(ndx + bs, l)]

import bpu_infer_lib
import numpy as np

class MobileNetv2_Embedder(object):
    def __init__(
        self, model_wts_path=None
    ):
        if model_wts_path is None:
            model_wts_path = "app/pydev_demo/models/mobilenetv2_224x224_nv12.bin"
        assert os.path.exists(
            model_wts_path
        ), f"Mobilenetv2 model path {model_wts_path} does not exists!"

        self.input_size = INPUT_WIDTH
        self.infer_obj = bpu_infer_lib.Infer(False)
        self.infer_obj.load_model(model_wts_path)

        logger.info(f"[INFO] MobileNetV2 BPU Embedder initialised.")
        logger.info(f"- model: {model_wts_path}")
        logger.info(f"- max batch size: {self.max_batch_size}")

    def preprocess(self, frame: np.ndarray):
        img_resized = cv2.resize(frame, (self.input_size, self.input_size))

        # BGR → YUV I420
        img_yuv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2YUV_I420)
        h, w = self.input_size, self.input_size
        y_size = h * w
        uv_size = y_size // 2

        # NV12 格式缓冲区
        nv12 = np.zeros((y_size + uv_size,), dtype=np.uint8)
        nv12[:y_size] = img_yuv[0:y_size]

        u = img_yuv[y_size : y_size + (y_size // 4)].reshape(-1, 1)
        v = img_yuv[y_size + (y_size // 4) :].reshape(-1, 1)
        uv = np.hstack((u, v)).flatten()
        nv12[y_size:] = uv

        # 写入 BPU 输入缓存
        ret = self.infer_obj.set_input_data(0, nv12)
        if ret != 0:
            raise RuntimeError("Failed to set NV12 data to BPU input buffer")

    def extract_feature(self, frame: np.ndarray):
        self.preprocess(frame)
        self.infer_obj.forward()

        feat_ptr = self.infer_obj.get_infer_res(0, 1280)
        feat = np.array([feat_ptr[i] for i in range(1280)], dtype=np.float32)

        return feat

# class MobileNetv2_Embedder(object):
#     def __init__(
#         self, model_wts_path=None, half=True, max_batch_size=16, bgr=True, gpu=True
#     ):
#         if model_wts_path is None:
#             model_wts_path = MOBILENETV2_BOTTLENECK_WTS
#         assert os.path.exists(
#             model_wts_path
#         ), f"Mobilenetv2 model path {model_wts_path} does not exists!"
#         self.model = MobileNetV2_bottle(input_size=INPUT_WIDTH, width_mult=1.0)
#         self.model.load_state_dict(torch.load(model_wts_path))

#         self.model.eval()  # inference mode, deactivates dropout layers

#         self.max_batch_size = max_batch_size
#         self.bgr = bgr

#         logger.info("MobileNetV2 Embedder for Deep Sort initialised")
#         logger.info(f"- gpu enabled: {self.gpu}")
#         logger.info(f"- half precision: {self.half}")
#         logger.info(f"- max batch size: {self.max_batch_size}")
#         logger.info(f"- expects BGR: {self.bgr}")

#         zeros = np.zeros((100, 100, 3), dtype=np.uint8)
#         self.predict([zeros])  # warmup

#     def preprocess(self, np_image):
#         if self.bgr:
#             np_image_rgb = np_image[..., ::-1]
#         else:
#             np_image_rgb = np_image

#         input_image = cv2.resize(np_image_rgb, (INPUT_WIDTH, INPUT_WIDTH))
#         trans = transforms.Compose(
#             [
#                 transforms.ToTensor(),
#                 transforms.Normalize(
#                     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
#                 ),
#             ]
#         )
#         input_image = trans(input_image)
#         input_image = input_image.view(1, 3, INPUT_WIDTH, INPUT_WIDTH)

#         return input_image

#     def predict(self, np_images):
#         all_feats = []

#         preproc_imgs = [self.preprocess(img) for img in np_images]

#         for this_batch in batch(preproc_imgs, bs=self.max_batch_size):
#             this_batch = torch.cat(this_batch, dim=0)
#             output = self.model.forward(this_batch)

#             all_feats.extend(output.data.numpy())

#         return all_feats

# class TorchReID_Embedder(object):
#     """
#     Embedder that works with torchreid (https://github.com/KaiyangZhou/deep-person-reid). Model zoo: https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO

#     Params
#     ------
#     - model_name (optional, str): name of model, see torchreid model zoo. defaults to osnet_ain_x1_0 
#     - model_wts_path (optional, str) : path to torchreid model weights, defaults to TORCHREID_OSNET_AIN_X1_0_MS_D_C_WTS if model_name=='osnet_ain_x1_0' (default) and else, imagenet pretrained weights of given model 
#     - bgr (optional, Bool) : boolean flag indicating if input frames are bgr or not, defaults to True
#     - gpu (optional, Bool) : boolean flag indicating if gpu is enabled or not
#     - max_batch_size: Does nothing, just for compatibility to other embedder classes
#     """

#     def __init__(
#         self, model_name=None, model_wts_path=None, bgr=True, gpu=True, max_batch_size=None,
#     ):
#         try: 
#             import torchreid 
#         except ImportError: 
#             raise Exception('ImportError: torchreid is not installed, please install and try again or choose another embedder')
        
#         from torchreid.utils import FeatureExtractor
        
#         if model_name is None: 
#             model_name = 'osnet_ain_x1_0'

#         if model_wts_path is None: 
#             model_wts_path = ''

#         if model_name=='osnet_ain_x1_0' and model_wts_path=='':
#             model_wts_path = TORCHREID_OSNET_AIN_X1_0_MS_D_C_WTS

#         self.gpu = gpu and torch.cuda.is_available()
#         if self.gpu:
#             device = 'cuda'
#         else:
#             device = 'cpu'

#         self.model = FeatureExtractor(
#             model_name=model_name, 
#             model_path=model_wts_path,
#             device=device,
#         )

#         self.bgr = bgr

#         logger.info("TorchReID Embedder for Deep Sort initialised")
#         logger.info(f"- gpu enabled: {self.gpu}")
#         logger.info(f"- expects BGR: {self.bgr}")

#         zeros = np.zeros((100, 100, 3), dtype=np.uint8)
#         self.predict([zeros])  # warmup

#     def preprocess(self, np_image):
#         """
#         Preprocessing for embedder network: Flips BGR to RGB, resize, convert to torch tensor, normalise with imagenet mean and variance, reshape. Note: input image yet to be loaded to GPU through tensor.cuda()

#         Parameters
#         ----------
#         np_image : ndarray
#             (H x W x C)

#         Returns
#         -------
#         Torch Tensor

#         """
#         if self.bgr:
#             np_image_rgb = np_image[..., ::-1]
#         else:
#             np_image_rgb = np_image
#         # torchreid handles the rest of the preprocessing
#         return np_image_rgb

#     def predict(self, np_images):
#         """
#         batch inference

#         Params
#         ------
#         np_images : list of ndarray
#             list of (H x W x C), bgr or rgb according to self.bgr

#         Returns
#         ------
#         list of features (np.array with dim = 1280)

#         """
#         preproc_imgs = [self.preprocess(img) for img in np_images]
#         output =  self.model(preproc_imgs)
#         return output.cpu().data.numpy()
