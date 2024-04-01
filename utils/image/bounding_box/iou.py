# Copyright (C) 2022-now yui-mhcp project author. All rights reserved.
# Licenced under a modified Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from loggers import timer, time_logger
from utils.keras_utils import ops
from utils.distance.distance_method import similarity_method_wrapper
from .converter import BoxFormat, box_converter_wrapper

@similarity_method_wrapper(name = 'iou', expand = False)
@timer
@box_converter_wrapper(
    (BoxFormat.XYXY, BoxFormat.YXYX), as_dict = False, as_list = True, dtype = 'float'
)
def compute_iou(boxes1, boxes2 = None, *, as_matrix = None, ** kwargs):
    if as_matrix is None:   as_matrix = boxes2 is None
    if boxes2 is None:      boxes2 = boxes1

    if as_matrix:
        boxes1 = [b[..., None, :] for b in boxes1]
        boxes2 = [b[..., None] for b in boxes2]

    xmin_1, ymin_1, xmax_1, ymax_1 = boxes1
    xmin_2, ymin_2, xmax_2, ymax_2 = boxes2

    areas_1 = (ymax_1 - ymin_1) * (xmax_1 - xmin_1)
    areas_2 = (ymax_2 - ymin_2) * (xmax_2 - xmin_2)

    xmin, ymin = ops.maximum(xmin_1, xmin_2), ops.maximum(ymin_1, ymin_2)
    xmax, ymax = ops.minimum(xmax_1, xmax_2), ops.minimum(ymax_1, ymax_2)

    _zero = 0. if isinstance(xmin, np.ndarray) else ops.convert_to_tensor(0, xmin.dtype)
    inter_w, inter_h = ops.maximum(_zero, xmax - xmin), ops.maximum(_zero, ymax - ymin)

    # making the `- inter` in the middle reduces value overflow when using `float16` computation
    inter = inter_w * inter_h
    union = areas_1 - inter + areas_2

    return ops.divide_no_nan(inter, union)

@similarity_method_wrapper(name = 'ioa', expand = False)
@timer
@box_converter_wrapper((BoxFormat.XYXY, BoxFormat.YXYX), as_dict = False)
def compute_ioa(boxes1, boxes2 = None, *, as_matrix = None, ** kwargs):
    if as_matrix is None:   as_matrix = boxes2 is None
    if boxes2 is None:      boxes2 = boxes1
    
    boxes1, boxes2 = boxes1[..., None, :], boxes2[..., None, :]
    if as_matrix: boxes1, boxes2 = boxes1[..., None, :, :], boxes2[..., None, :, :, :]
    
    xmin_1, ymin_1, xmax_1, ymax_1 = ops.unstack(boxes1, axis = -1, num = 4)
    xmin_2, ymin_2, xmax_2, ymax_2 = ops.unstack(boxes2, axis = -1, num = 4)

    areas_1 = (ymax_1 - ymin_1) * (xmax_1 - xmin_1)

    xmin, ymin = ops.maximum(xmin_1, xmin_2), ops.maximum(ymin_1, ymin_2)
    xmax, ymax = ops.minimum(xmax_1, xmax_2), ops.minimum(ymax_1, ymax_2)

    inter_w, inter_h = ops.maximum(0., xmax - xmin), ops.maximum(0., ymax - ymin)
    inter = inter_w * inter_h
    dtype = 'float32' if ops.is_int(boxes1) else boxes1.dtype
    return ops.cast(ops.divide_no_nan(inter, areas_1)[..., 0], dtype)
