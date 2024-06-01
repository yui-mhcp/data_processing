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

import keras
import logging

from .ops import build_custom_op

logger = logging.getLogger(__name__)

_limited_memory = False

def split_gpus(n, memory = 2048):
    if keras.backend.backend() != 'tensorflow': return
    
    import tensorflow as tf
    """ Splits each physical GPU into `n` virtual devices with `memory` available gpu memory """
    gpus = tf.config.list_physical_devices('GPU')
    try:
        for gpu in gpus:
            tf.config.set_logical_device_configuration(gpu, [
                tf.config.LogicalDeviceConfiguration(memory_limit = memory)
                for _ in range(n)
            ])
    except RuntimeError as e:
        logger.error(e)
    
    logger.info("# physical GPU : {}\n# logical GPU : {}".format(
        len(tf.config.list_physical_devices('GPU')), len(tf.config.list_logical_devices('GPU'))
    ))

def limit_gpu_memory(limit = 4096):
    if keras.backend.backend() != 'tensorflow': return
    import tensorflow as tf
    """ Limits the tensorflow visible GPU memory on each available physical device """
    global _limited_memory
    if _limited_memory or not limit: return
    
    gpus = tf.config.list_physical_devices('GPU')
    try:
        for gpu in gpus:
            tf.config.set_logical_device_configuration(gpu, [
                tf.config.LogicalDeviceConfiguration(memory_limit = limit)
            ])
        _limited_memory = True
    except Exception as e:
        logger.error("Error while limiting tensorflow GPU memory : {}".format(e))

def set_memory_growth(memory_growth = True):
    if keras.backend.backend() != 'tensorflow': return
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, memory_growth)
    except Exception as e:
        logger.error("Error while setting memory growth : {}".format(e))

def get_memory_stats_tf(gpu = 'GPU:0'):
    import tensorflow as tf

    if isinstance(gpu, int): gpu = 'GPU:{}'.format(gpu)
    
    mem_usage = tf.config.experimental.get_memory_info(gpu)
    tf.config.experimental.reset_memory_stats(gpu)
    return mem_usage

def get_memory_stats_jax(gpu = 0):
    import jax

    if isinstance(gpu, str): gpu = int(gpu.split(':')[-1])
    
    mem_usage = ...
    return {'current' : mem_usage['bytes_in_use'], 'peak' : mem_usage['peak_bytes_in_use']}

get_memory_stats = build_custom_op(
    tf_fn   = get_memory_stats_tf,
    jax_fn  = get_memory_stats_jax,
    name    = 'get_memory_stats'
)

def show_memory(message = '', gpu = 0):
    mem_usage = get_memory_stats(gpu = gpu)
    
    logger.info('{}{}'.format(message if not message else message + '\t: ', {
        k : '{:.3f} Gb'.format(v / 1024 ** 3) for k, v in mem_usage.items()
    }))
    return mem_usage

