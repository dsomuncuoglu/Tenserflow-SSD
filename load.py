# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 14:17:24 2021

@author: DenizS
"""


import os
import klasor as k
import config as c
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

# Load pipeline config and build a detection model
configs = c.config_util.get_configs_from_pipeline_file(k.CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = c.tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(k.CHECKPOINT_PATH, 'ckpt-101')).expect_partial()


def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections