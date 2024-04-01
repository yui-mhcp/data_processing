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

import os
import cv2
import keras
import unittest
import numpy as np
import keras.ops as K

from absl.testing import parameterized

from utils.keras_utils import ops
from utils.image import load_image, apply_mask, normalize_color
from utils.image.bounding_box import *
from unitests import CustomTestCase, data_dir

def is_tensorflow_available():
    try:
        import tensorflow
        return True
    except:
        return False

_box_formats    = ('xywh', 'xyxy', 'yxyx')

class TestBoxesConvertion(CustomTestCase, parameterized.TestCase):
    def setUp(self):
        self.image_h, self.image_w = 720, 1024
        
        self.relative_boxes_xywh = np.array([
            [0, 0, 1, 1], [0.25, 0.2, 0.1, 0.2], [0.5, 0.5, 0.5, 0.5]
        ], dtype = np.float32)
        self.relative_boxes_xyxy = np.array([
            [0, 0, 1, 1], [0.25, 0.2, 0.35, 0.4], [0.5, 0.5, 1,1]
        ], dtype = np.float32)
        self.relative_boxes_yxyx = np.array([
            [0, 0, 1, 1], [0.2, 0.25, 0.4, 0.35], [0.5, 0.5, 1,1]
        ], dtype = np.float32)

        factor  = np.array([[self.image_w, self.image_h, self.image_w, self.image_h]])
        self.absolute_boxes_xywh = np.array([
            [0, 0, 1, 1], [0.25, 0.2, 0.1, 0.2], [0.5, 0.5, 0.5, 0.5]
        ], dtype = np.float32) * factor
        self.absolute_boxes_xyxy = np.array([
            [0, 0, 1, 1], [0.25, 0.2, 0.35, 0.4], [0.5, 0.5, 1,1]
        ], dtype = np.float32) * factor
        self.absolute_boxes_yxyx = np.array([
            [0, 0, 1, 1], [0.2, 0.25, 0.4, 0.35], [0.5, 0.5, 1,1]
        ], dtype = np.float32) * factor[:, ::-1]
        
        self.absolute_boxes_xywh = self.absolute_boxes_xywh.astype(np.int32)
        self.absolute_boxes_xyxy = self.absolute_boxes_xyxy.astype(np.int32)
        self.absolute_boxes_yxyx = self.absolute_boxes_yxyx.astype(np.int32)

    @parameterized.product(
        source = _box_formats, target = _box_formats, to_tensor = (True, False)
    )
    def test_converter(self, source, target, to_tensor):
        rel_in_boxes    = getattr(self, 'relative_boxes_{}'.format(source))
        rel_out_boxes   = getattr(self, 'relative_boxes_{}'.format(target))
        abs_in_boxes    = getattr(self, 'absolute_boxes_{}'.format(source))
        abs_out_boxes   = getattr(self, 'absolute_boxes_{}'.format(target))

        if to_tensor:
            rel_in_boxes    = K.convert_to_tensor(rel_in_boxes)
            rel_out_boxes   = K.convert_to_tensor(rel_out_boxes)
            abs_in_boxes    = K.convert_to_tensor(abs_in_boxes)
            abs_out_boxes   = K.convert_to_tensor(abs_out_boxes)

        if source == target:
            self.assertTrue(
                convert_box_format(rel_in_boxes, source = source, target = target) is rel_in_boxes,
                'The function should return the same instance when `source == target`'
            )

        self.assertEqual(
            convert_box_format(rel_in_boxes, source = source, target = target),
            rel_out_boxes
        )
        self.assertEqual(convert_box_format(
            rel_in_boxes, source = source, target = target, as_list = True
        ), ops.unstack(rel_out_boxes, axis = -1, num = 4))

        self.assertEqual(convert_box_format(
            abs_in_boxes,
            source = source,
            target = target,
            image_shape = (self.image_h, self.image_w)
        ), abs_out_boxes)
        self.assertEqual(convert_box_format(
            rel_in_boxes,
            source = source,
            target = target,
            normalize_mode  = NORMALIZE_WH,
            image_h = self.image_h,
            image_w = self.image_w
        ), abs_out_boxes)
        self.assertEqual(convert_box_format(
            abs_in_boxes,
            source = source,
            target = target,
            normalize_mode  = NORMALIZE_01,
            image_h = self.image_h,
            image_w = self.image_w
        ), rel_out_boxes, max_err = 1e-3)
    
    def test_dezoom(self):
        kwargs = {'normalize_mode' : 'standardize'}
        self.assertEqual(
            convert_box_format([0., 0., 1., 1.], dezoom_factor = 0.5, ** kwargs),
            np.array([[0.25, 0.25, 0.5, 0.5]], dtype = 'float64')
        )
        self.assertEqual(
            convert_box_format([0., 0., 1., 1.], dezoom_factor = 2, ** kwargs),
            np.array([[0, 0, 1, 1]], dtype = 'float64')
        )
        
        self.assertEqual(
            convert_box_format([0.25, 0.25, .5, .5], dezoom_factor = 2, ** kwargs),
            np.array([[0, 0, 1, 1]], dtype = 'float64')
        )
        self.assertEqual(
            convert_box_format([0.5, 0.5, .5, .5], dezoom_factor = 2, ** kwargs),
            np.array([[0.25, 0.25, .75, .75]], dtype = 'float64')
        )
        
    @parameterized.parameters(
        {'method' : 'x', 'expected' : [0, 1, 2]},
        {'method' : 'y', 'expected' : [0, 1, 2]},
        {'method' : 'w', 'expected' : [0, 2, 1]},
        {'method' : 'h', 'expected' : [0, 2, 1]},
        {'method' : 'area', 'expected' : [0, 2, 1]},
        {'method' : 'center', 'expected' : [1, 0, 2]},
        {'method' : 'corner', 'expected' : [0, 1, 2]},
    )
    def test_sort(self, method, expected):
        for to_tensor in (False, True):
            with self.subTest(to_tensor = to_tensor):
                for source in _box_formats:
                    kwargs = {
                        'return_indices' : True,
                        'image_shape' : (self.image_h, self.image_w),
                        'source' : source
                    }
                    rel_boxes    = getattr(self, 'relative_boxes_{}'.format(source))
                    abs_boxes    = getattr(self, 'absolute_boxes_{}'.format(source))

                    if to_tensor:
                        rel_boxes    = K.convert_to_tensor(rel_boxes)
                        abs_boxes    = K.convert_to_tensor(abs_boxes)

                    self.assertEqual(
                        sort_boxes(rel_boxes, method = method, ** kwargs), expected
                    )
                    self.assertEqual(
                        sort_boxes(abs_boxes, method = method, ** kwargs), expected
                    )


class TestBoxProcessing(CustomTestCase, parameterized.TestCase):
    def setUp(self):
        self.image = np.arange(16).reshape(4, 4)
    
    def test_crop_empty_box(self):
        self.assertEqual(crop_box(self.image, [])[1], None)

    def test_crop_single_box(self):
        self.assertEqual(
            crop_box(self.image, [0, 1, 1, 3])[1], self.image[1: 4, :1]
        )
        self.assertEqual(
            crop_box(self.image, [0, 1, 1, 3], source = 'xyxy')[1], self.image[1 : 3, :1]
        )

    def test_crop_multiple_boxes(self):
        self.assertEqual(
            crop_box(self.image, [[0, 1, 1, 3], [2, 2, 1, 2]])[1],
            [self.image[1:4, :1], self.image[2:4, 2:3]]
        )


class TestDrawing(CustomTestCase, parameterized.TestCase):
    def setUp(self):
        self.image = np.random.uniform(size = (720, 1024, 3)).astype('float32')
        self.mask_box = [250, 200, 500, 500]
        self.kwargs = {'color' : 'blue', 'thickness' : 3}
    
    def test_rectangle(self):
        x, y, w, h = self.mask_box
        color = normalize_color('blue', dtype = self.image.dtype.name).tolist()
        
        self.assertEqual(
            draw_boxes(
                self.image.copy(), self.mask_box, shape = 'rectangle', ** self.kwargs
            ),
            cv2.rectangle(
                self.image.copy(), (x, y), (x + w, y + h), color, 3

            )
        )
        
    def test_circle(self):
        x, y, w, h = self.mask_box
        color = normalize_color('blue', dtype = self.image.dtype.name).tolist()

        self.assertEqual(
            draw_boxes(
                self.image, self.mask_box, shape = 'circle', ** self.kwargs
            ),
            cv2.circle(
                ops.convert_to_numpy(self.image), (x + w // 2, y + h // 2),
                min(w, h) // 2, color, 3

            )
        )
    
    def test_ellipse(self):
        x, y, w, h = self.mask_box
        color = normalize_color('blue', dtype = self.image.dtype.name).tolist()

        self.assertEqual(
            draw_boxes(
                self.image, self.mask_box, color = 'blue', shape = 'ellipse', thickness = 3
            ),
            cv2.ellipse(
                ops.convert_to_numpy(self.image),
                angle       = 0,
                startAngle  = 0,
                endAngle    = 360, 
                center      = (x + w // 2, y + h // 2),
                thickness   = 3,
                axes    = (w // 2, int(h / 1.5)),
                color   = color
            )
        )

    @parameterized.parameters('rectangle', 'ellipse', 'circle')
    def test_box_to_mask(self, shape):
        self.assertEqual(
            box_as_mask(self.image, self.mask_box, shape = shape),
            draw_boxes(
                np.zeros(self.image.shape), self.mask_box, shape = shape, thickness = -1
            )[..., :1] != 0
        )

    def test_box_masking(self):
        box_mask = box_as_mask(self.image, self.mask_box)
        
        for method in apply_mask.methods.keys():
            if method == 'replace': continue
            for on_background in (False, True):
                with self.subTest(method = method, on_background = on_background):
                    mask = box_mask if not on_background else ~box_mask
                    if method == 'keep':
                        target = np.where(mask, self.image, 0.)
                    elif method == 'remove':
                        target = np.where(mask, 0., self.image)
                    elif method == 'blur':
                        target = np.where(
                            mask,
                            cv2.blur(ops.convert_to_numpy(self.image), (21, 21)),
                            self.image
                        )

                    self.assertEqual(apply_mask(
                        self.image, box_mask, method = method, on_background = on_background
                    ), target)


class TestIoU(CustomTestCase):
    def test_single_iou(self):
        box1 = np.array([[100, 101, 200, 201]])
        box2 = box1 + 1
        # area of bb1 and bb1_off_by_1 are each 10000.
        # intersection area is 99*99=9801
        # iou=9801/(2*10000 - 9801)=0.96097656633
        self.assertEqual(
            compute_iou(box1[0], box2[0], source = "yxyx"), [0.96097656633]
        )
        self.assertEqual(
            compute_iou(box1, box2, source = "yxyx"), [0.96097656633]
        )

    def test_iou(self):
        bb1 = [100, 101, 200, 201]
        bb1_off_by_1_pred = [101, 102, 201, 202]
        iou_bb1_bb1_off = 0.96097656633
        top_left_bounding_box = [0, 2, 1, 3]
        far_away_box = [1300, 1400, 1500, 1401]
        another_far_away_pred = [1000, 1400, 1200, 1401]

        # Rows represent predictions, columns ground truths
        expected_matrix_result = np.array(
            [[iou_bb1_bb1_off, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        )

        sample_y_true = np.array([bb1, top_left_bounding_box, far_away_box], dtype = 'int32')
        sample_y_pred = np.array(
            [bb1_off_by_1_pred, top_left_bounding_box, another_far_away_pred], dtype = 'int32'
        )
        self.assertEqual(
            compute_iou(sample_y_true, sample_y_pred, source = "yxyx", as_matrix = True),
            expected_matrix_result
        )
        self.assertEqual(
            compute_iou(sample_y_true, sample_y_pred, source = "yxyx"),
            np.diagonal(expected_matrix_result)
        )
        
        
        batch_y_true = np.stack([
            sample_y_true, sample_y_true[::-1]
        ], axis = 0)
        batch_y_pred = np.stack([
            sample_y_pred, sample_y_pred[::-1]
        ], axis = 0)
        batch_matrix    = np.stack([
            expected_matrix_result, expected_matrix_result[::-1, ::-1]
        ], axis = 0)
        self.assertEqual(
            compute_iou(batch_y_true, batch_y_pred, source = "yxyx", as_matrix = True),
            batch_matrix
        )
        self.assertEqual(
            compute_iou(batch_y_true, batch_y_pred, source = "yxyx"),
            np.stack([
                np.diagonal(batch_matrix[0]), np.diagonal(batch_matrix[1])
            ], axis = 0)
        )

    def test_ioa(self):
        box1 = np.array([[1, 1, 5, 10]])
        box2 = box1 * 2
        box3 = np.array([[0, 0, 2, 2]])

        self.assertEqual(
            compute_ioa(box1[0], box2[0], source = "xywh"), [36 / 50]
        )
        self.assertEqual(
            compute_ioa(box1, box2, source = "xywh"), np.array([36 / 50], dtype = 'float32')
        )
        
        boxes = np.concatenate([box1, box2, box3], axis = 0)
        self.assertEqual(
            compute_ioa(boxes, source = "xywh", as_matrix = True),
            np.array([
                [1., 36 / 50, 1 / 50],
                [36 / 200, 1, 0],
                [1 / 4, 0, 1]
            ], dtype = 'float32')
        )
        self.assertEqual(
            compute_ioa(boxes, boxes[[0, 2]], source = "xywh", as_matrix = True),
            np.array([
                [1, 1 / 50],
                [36 / 200, 0],
                [1 / 4, 1]
            ], dtype = 'float32')
        )
        self.assertEqual(
            compute_ioa(boxes[[0, 2]], boxes, source = "xywh", as_matrix = True),
            np.array([
                [1, 36 / 50, 1 / 50],
                [1 / 4, 0, 1]
            ], dtype = 'float32')
        )


class TestNonMaxSuppression(CustomTestCase, parameterized.TestCase):
    def setUp(self):
        self.boxes = np.array([
            [0, 0, 0.2, 0.2],
            [0.1, 0.1, 0.3, 0.3],
            [0.2, 0.2, 0.4, 0.4],
            [0.3, 0.3, 0.5, 0.5]
        ], dtype = 'float32')

    @parameterized.parameters('tensorflow', 'nms', 'fast', 'padded')
    def test_standard_nms(self, method):
        if method == 'tensorflow' and not is_tensorflow_available():
            self.skip('Tensorflow is not available, skipping the `tensorflow` nms method')
            return
        
        boxes, scores, valids = nms(
            self.boxes, nms_threshold = 0.1, source = 'yxyx', method = method, run_eagerly = True
        )
        boxes = ops.convert_to_numpy(boxes)[ops.convert_to_numpy(valids)]
        self.assertEqual(boxes, self.boxes[[0, 2]])
        
        boxes, scores, valids = nms(
            self.boxes, nms_threshold = 0.1, source = 'yxyx', method = method, run_eagerly = False
        )
        boxes = ops.convert_to_numpy(boxes)[ops.convert_to_numpy(valids)]
        self.assertEqual(boxes, self.boxes[[0, 2]])
    
    def test_locality_aware_nms(self):
        boxes, scores, valids = nms(
            self.boxes,
            nms_threshold   = 0.1,
            merge_threshold = 0.1,
            source  = 'yxyx',
            method  = 'lanms',
            run_eagerly = True
        )
        boxes = ops.convert_to_numpy(boxes)[ops.convert_to_numpy(valids)]
        self.assertEqual(
            boxes, np.array([[0, 0, 0.3, 0.3], [0.2, 0.2, 0.5, 0.5]], dtype = 'float32')
        )
        
        boxes, scores, valids = nms(
            self.boxes,
            nms_threshold   = 0.1,
            merge_threshold = 0.1,
            merge_method    = 'average',
            source  = 'yxyx',
            method  = 'lanms',
            run_eagerly = True
        )
        boxes = ops.convert_to_numpy(boxes)[ops.convert_to_numpy(valids)]
        self.assertEqual(
            boxes, np.array([[0.05, 0.05, 0.25, 0.25], [0.25, 0.25, 0.45, 0.45]], dtype = 'float32')
        )


        boxes, scores, valids = nms(
            self.boxes,
            nms_threshold   = 0.01,
            merge_threshold = 0.1,
            source  = 'yxyx',
            method  = 'lanms',
            run_eagerly = True
        )
        boxes = ops.convert_to_numpy(boxes)[ops.convert_to_numpy(valids)]
        self.assertEqual(boxes, np.array([[0, 0, 0.3, 0.3]], dtype = 'float32'))

        boxes, scores, valids = nms(
            self.boxes,
            nms_threshold   = 0.1,
            merge_threshold = 0.01,
            source  = 'yxyx',
            method  = 'lanms',
            run_eagerly = True
        )
        boxes = ops.convert_to_numpy(boxes)[ops.convert_to_numpy(valids)]
        self.assertEqual(
            boxes, np.array([[0, 0, 0.5, 0.5]], dtype = 'float32'), 'The LANMS should be iterative'
        )
