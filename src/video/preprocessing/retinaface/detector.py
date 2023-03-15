from __future__ import print_function

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from .data import cfg_mnet
from .layers.functions.prior_box import PriorBox
from .loader import load_model
from .utils.box_utils import decode, decode_landm
from .utils.nms.py_cpu_nms import py_cpu_nms


class RetinafaceDetector:
    def __init__(self, net='mnet', type='cuda'):
        cudnn.benchmark = True
        self.net = net
        self.device = torch.device(type)
        self.model = load_model(self.net).to(self.device)
        self.model.eval()

    def detect_faces(self, img_raw:np.ndarray, confidence_threshold=0.9, top_k=5000, nms_threshold=0.4, keep_top_k=750, resize=1):
        img = torch.Tensor(img_raw).float().to(self.device)
        im_height, im_width = img.shape[:2]
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        scale = scale.to(self.device)

        img = torch.subtract(img, torch.Tensor((104., 117., 123.)).to(self.device))
        img = img.permute(2, 0, 1)
        img = img.unsqueeze(0)



        with torch.no_grad():
            loc, conf, landms = self.model(img)  # forward pass

        priorbox = PriorBox(cfg_mnet, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg_mnet['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]

        # keep top-K faster NMS
        dets = dets[:keep_top_k, :]

        return dets

