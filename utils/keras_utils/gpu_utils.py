# Copyright (C) 2023-now yui-mhcp project's author. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
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

def show_memory(gpu = 'GPU:0', message = ''):
    if keras.backend.backend() != 'tensorflow': return
    import tensorflow as tf
    """ Displays the memory stats computed by `tensorflow`, then resets the stats """
    mem_usage = tf.config.experimental.get_memory_info(gpu)
    logger.info('{}{}'.format(message if not message else message + '\t: ', {
        k : '{:.3f} Gb'.format(v / 1024 ** 3) for k, v in mem_usage.items()
    }))
    tf.config.experimental.reset_memory_stats(gpu)
    return mem_usage
