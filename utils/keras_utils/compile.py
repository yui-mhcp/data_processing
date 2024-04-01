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
import inspect
import logging
import warnings
import threading
import pandas as pd
import keras.ops as K

from keras import tree
from typing import Union, Tuple
from dataclasses import dataclass
from functools import wraps, cache

from . import ops
from loggers import timer, time_logger
from utils.wrapper_utils import ContextManager, update_signature
from utils.generic_utils import convert_to_str, get_annotations, get_args, get_kwargs, has_kwargs

logger  = logging.getLogger(__name__)

_jit_compile    = keras.backend.backend() != 'tensorflow'
_should_execute_eagerly = {}

@dataclass
class TensorSpec:
    shape   : Union[None, Tuple[int]] = None
    dtype   : str   = None
    name    : str   = None
    
    def __hash__(self):
        return hash((self.shape, self.dtype))
    
    def __eq__(self, o):
        return self.name == o.name and self.dtype == o.dtype

class XLAExecution(ContextManager):
    def __init__(self, force_tensorflow = False):
        if not force_tensorflow:
            super().__init__(
                enter = lambda: ops.set_xla_execution(True),
                exit  = lambda * _: ops.set_xla_execution(False)
            )
        else:
            super().__init__(enter = lambda: None, exit = lambda *_: None)

class EagerExecution(ContextManager):
    def __init__(self):
        super().__init__(
            enter = lambda: set_eager_execution(True),
            exit  = lambda * _: set_eager_execution(False)
        )

def jit_compile():
    global _jit_compile
    return _jit_compile

def set_jit_compile(jit_compile):
    global _jit_compile
    _jit_compile = jit_compile

def should_execute_eagerly():
    """
        This function returns whether a graph-compiled function should be executed eagerly or not
        The typical use case is to execute eagerly all nested functions if it has been required via the `execute_eagerly` argument from a `graph_compile`d function
        
        Note that the function is thread-safe, meaning that if one function thread A is requested to run eagerly, it will not afect functions from other threads
    """
    return _get_thread_id() in _should_execute_eagerly

def set_eager_execution(eager):
    if eager:   _should_execute_eagerly[_get_thread_id()] = True
    else:       _should_execute_eagerly.pop(_get_thread_id())

def graph_compile(fn    = None,
                  
                  cast_kwargs   = True,
                  cast_defaults = True,
                  follow_type_hints = True,
                  
                  support_xla   = True,
                  prefer_xla    = None,
                  
                  prepare   = None,
                  prepare_for_xla   = None,
                  prepare_for_graph = None,
                  
                  force_tensorflow  = False,
                  
                  static_args   = 'auto',
                  reduce_retracing  = True,
                  
                  ** compile_kwargs
                 ):
    """
        This function is equivalent to `tf.function` except that it pre-processes the arguments to cast them to `Tensor` (only if `ollow_type_hints == True`). It also casts numeric kwargs, and default values
        
        Example :
        ```python
            @graph_compile(follow_type_hints = True, reduce_retracing = True)
            def square(x : tf.Tensor):
                return x ** 2

            for i in range(5): print(square(i))
            # Is strictly equivalent to
            for i in range(5): print(square(tf.cast(i, tf.int32)))
        ```
        This feature has been removed after tensorflow 2.10 but it is useful to avoid retracing
        
        Another fancy feature which was not implemented originally is that you can specify expected shapes and types by setting `tf.TensorSpec` as annotation
        This will automatically cast the input to the right dtype, which may be useful to standardize the inputs and further reduce retracing !
        It is a bit equivalent to a partially known `input_signature` in the `tf.function`
        ```python
            @tf_compile(experimental_follow_type_hints = True, reduce_retracing = True)
            def square(x : tf.TensorSpec(shape = None, dtype = tf.float32)):
                return x ** 2

            for i in range(5): print(square(i))
            # Is strictly equivalent to
            for i in range(5): print(square(tf.cast(i, tf.float32)))
        ```
    """
    def wrapper(fn):
        def log_retracing(args, kwargs):
            if logger.isEnabledFor(logging.RETRACING):
                logger.retracing('Retracing {} with args : {} - kwargs : {}'.format(
                    fn_name, args, kwargs
                ))
        
        def fn_with_retracing_logs(* args, ** kwargs):
            if ops.is_tensorflow_graph():
                import tensorflow as tf
                tf.autograph.experimental.do_not_convert(log_retracing)(args, kwargs)
            return fn(* args, ** kwargs)
        
        @wraps(fn)
        def inner(* args, run_eagerly = None, use_xla = None, recompile = False, ** kwargs):
            if skip_kwargs: kwargs = {k : v for k, v in kwargs.items() if k in fn_all_names}
            if prepare is not None: args, kwargs = prepare(* args, ** kwargs)
            
            if follow_type_hints and fn_annots:
                with time_logger.timer('follow_type_hints', debug = True):
                    args = tuple([
                        _cast_arg(arg, fn_annots[name], force_tensorflow) if name in fn_annots else arg
                        for name, arg in zip(fn_args[: len(args)], args)
                    ])
                    kwargs  = {
                        k : _cast_arg(v, fn_annots[k], force_tensorflow) if k in fn_annots else v
                        for k, v in kwargs.items()
                    }
            
            if not skip_kwargs and cast_kwargs:
                with time_logger.timer('cast_kwargs', debug = True):
                    for k, v in kwargs.items():
                        if k not in fn_all_names:
                            kwargs[k] = _cast_arg(v, None, force_tensorflow)
            
            if cast_defaults and defaults_to_cast:
                with time_logger.timer('cast_defaults', debug = True):
                    for k, annot in defaults_to_cast.items():
                        if k not in fn_args[:len(args)] and k not in kwargs:
                            kwargs[k] = _cast_arg(fn_kwargs[k], annot, force_tensorflow)
            
            if not ops.executing_eagerly(): return fn(* args, ** kwargs)

            if should_execute_eagerly() and run_eagerly is not False:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('The function is requested to run eagerly by another function')
                return fn(* args, ** kwargs)
            elif run_eagerly:
                with EagerExecution():
                    return fn(* args, ** kwargs)

            if use_xla is None:
                use_xla = (support_xla and jit_compile()) or prefer_xla
            elif use_xla and not support_xla:
                warnings.warn('`use_xla = True` but the function {} does not support XLA\nSet `support_xla = True` in the decorator if you want to enable it'.format(fn))
                use_xla = False

            if not use_xla and not ops.is_tensorflow_backend():
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('The function should be executed in graph without XLA which is not supported by this backend')
                return fn(* args, ** kwargs)
            
            _compile_kwargs = compile_kwargs
            if use_xla and ops.is_jax_backend() and static_args == 'auto':
                _compile_kwargs = _compile_kwargs.copy()
                for k, v in kwargs.items():
                    if k not in fn_all_names and not ops.is_tensor(v):
                        _compile_kwargs['static_argnames'] += [k]
            
            key = 'xla' if use_xla else 'graph'
            if recompile or key not in _compiled:
                _fn = fn_with_retracing_logs if force_tensorflow or ops.is_tensorflow_backend() else fn
                _compiled[key] = timer(compile_function(
                    _fn, jit_compile = use_xla, ** _compile_kwargs
                ), name = '{}_{}'.format(key, fn_name), debug = True)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('The function is executed in {} mode'.format(key))

            if use_xla and prepare_for_xla is not None:
                args, kwargs = prepare_for_xla(* args, ** kwargs)
            elif prepare_for_graph is not None:
                args, kwargs = prepare_for_graph(* args, ** kwargs)

            with XLAExecution(force_tensorflow = force_tensorflow):
                return _compiled[key](* args, ** kwargs)
        
        _compiled   = {}
        
        fn_name     = fn.__name__ if hasattr(fn, '__name__') else fn.__class__.__name__
        fn_args     = get_args(fn)
        fn_kwargs   = get_kwargs(fn)
        fn_all_names    = set(fn_args + list(fn_kwargs.keys()))

        skip_kwargs = not has_kwargs(fn)
        
        fn_annots = {
            k : v for k, v in get_annotations(fn).items() if isinstance(v, TensorSpec)
        }
        if 'input_signature' in compile_kwargs:
            _names = fn_args if fn_args[0] != 'self' else fn_args[1:]
            fn_annots.update({
                name : sign for name, sign in zip(_names, compile_kwargs.pop('input_signature'))
            })
        
        if static_args == 'auto':
            compile_kwargs['static_argnames'] = [
                k for k in fn_all_names if k not in fn_annots
            ]
        
        defaults_to_cast    = {
            k : v for k, v in fn_annots.items() if fn_kwargs.get(k, None) is not None
        }
        
        return inner
    
    compile_kwargs.update({
        'reduce_retracing'  : reduce_retracing,
        'force_tensorflow'  : force_tensorflow
    })
    follow_type_hints = compile_kwargs.pop('experimental_follow_type_hints', follow_type_hints)
    return wrapper if fn is None else wrapper(fn)

def execute_eagerly(fn  = None,
                    Tout    = None,
                    signature   = None,
                    default_key = None,
                    numpy   = False,
                    name    = None
                   ):
    """
        This function wraps `fn`, a regular python function, such that it can be used transparently inside a ``tf.function` executing by automatically calling `tf.py_function` or `tf.numpy_function`
        
        Arguments :
            - fn    : the python function to wrap
            - Tout  : output types of `fn`
            - signature : (list of) `tf.TensorSpec` that gives both shapes and types information
            - default_key   : the key to use if the 1st argument is a `dict / pd.Series`
            - numpy : whether to use `numpy_function` instead of `py_function`
            - name  : the operation name
        Return :
            If `fn is None`:
                - wraper    : a decorator function
            Else :
                - decorated : the function wrapped
        
        Note : if the function is executed eagerly (i.e. `tf.executing_eagerly() == True`), `fn` is simply called, meaning that the output is **not** forced to be a `tf.Tensor` !
        
        Note 2 : the benefit of passing `signature` instead of `Tout` is that it will fix the static shape of the output tensor with `tf.ensure_shape`, which may be required for some future usage
        
        Note 3 : when calling the decorated function, 2 new arguments are supported :
            - key       : to override the `default_key` argument
            - shape     : to specify a more precise shape (or in complement to `Tout`), see Note 2
        
        Known limitation : the decorator cannot be used on class method directly
        Example :
        ```python
        # This will raise an error when called in graph-mode due to the `self` parameter
        # which is not convertible to `tf.Tensor`, and thus not usable in `tf.numpy_function`
        class TextEncoderV1(object):
            @execute_eagerly(...)
            def encode(self, text, ...):
                ...
        
        # This will work properly both in eager and graph modes !
        # This way, the `self` argument is not *explicitely* passed, which makes `tf.numpy_function` happy :)
        class TextEncoderV2(object):
            def __init__(...):
                ...
                self.encode = execute_eagerly(self.encode, ...)
            
            def encode(self, text, ...):
                ...
        ```
    """
    def wrapper(fn):
        @wraps(fn)
        def inner(* args, shape = Sout, key = default_key, ** kwargs):
            if ops.is_tensorflow_graph() and is_class_method:
                function, args = getattr(args[0], fn.__name__), args[1:]
            else:
                function = fn
            
            if len(args) > 0 and isinstance(args[0], (dict, pd.Series)):
                if key and key in args[0]:
                    args = (args[0][key], ) + args[1:]
                elif default_key and default_key in args[0]:
                    args = (args[0][default_key], ) + args[1:]

            if not ops.is_tensorflow_graph():
                if numpy:
                    args = tree.map_structure(
                        lambda a: ops.convert_to_numpy(a) if K.is_tensor(a) else a, args
                    )
                    kwargs = tree.map_structure(
                        lambda a: ops.convert_to_numpy(a) if K.is_tensor(a) else a, kwargs
                    )
                return function(* args, ** kwargs)

            import tensorflow as tf
            
            python_function = tf.py_function if not numpy else tf.numpy_function
            
            if not kwargs:
                result = python_function(function, args, Tout = Tout)
            else:
                def fn_with_kwargs(n, * args_and_kwargs):
                    args    = args_and_kwargs[:n]
                    keys    = convert_to_str(args_and_kwargs[n])
                    kwargs  = {k : v for k, v in zip(keys, args_and_kwargs[n + 1 :])}
                    return function(* args, ** kwargs)

                keys    = list(kwargs.keys())
                vals    = [kwargs[k] for k in keys]
                args_with_kv = list(args) + [tf.cast(keys, tf.string)] + vals
                result = python_function(
                    fn_with_kwargs, [len(args)] + args_with_kv, Tout = Tout
                )

            if isinstance(shape, tuple):    shape = tf.TensorShape(shape)
            elif isinstance(shape, list):   shape = [tf.TensorShape(s) for s in shape]
            if shape is not None:
                result = tf.nest.map_structure(tf.ensure_shape, result, shape)
            elif Sout is not None:
                result = tf.nest.map_structure(tf.ensure_shape, result, Sout)

            return result
        
        is_class_method     = 'self' == list(inspect.signature(fn).parameters.keys())[0]
        inner.__signature__ = update_signature(fn, shape = Sout, key = default_key)

        inner.signature = signature
        inner.Tout      = Tout
        inner.default_key   = default_key

        return inner
    
    assert Tout is not None or signature is not None

    Sout = None
    if signature is not None:
        Tout    = tree.map_structure(lambda s: s.dtype, signature)
        Sout    = tree.map_structure(lambda s: s.shape, signature)
    
    return wrapper if fn is None else wrapper(fn)

def compile_function(fn, jit_compile, force_tensorflow = False, ** kwargs):
    compile_fn = None
    if ops.is_tensorflow_backend() or force_tensorflow:
        import tensorflow as tf
        compile_fn = tf.function
        kwargs['jit_compile'] = jit_compile
    elif ops.is_torch_backend() and jit_compile:
        import torch
        compile_fn = torch.compile
    elif ops.is_jax_backend() and jit_compile:
        import jax
        compile_fn = jax.jit
    
    if compile_fn is not None:
        kwargs = {
            k : v for k, v in kwargs.items() if k in inspect.signature(compile_fn).parameters
        }
        logger.debug('Compiling {} with kwargs {}'.format(fn, kwargs))
        return compile_fn(fn, ** kwargs)
    
    return fn

def _should_cast_kwarg(x):
    if isinstance(x, dict): return all(_should_cast_kwarg(vi) for vi in x.values())
    if isinstance(x, list): return all(_should_cast_kwarg(xi) for xi in x)
    return not isinstance(x, (str, bool)) and not callable(x)

def _cast_arg(value, annot, force_tensorflow = False, *, cache = True):
    if value is None: return None
    
    if cache and isinstance(value, (int, float, bool)):
        return _cached_cast_arg(value, annot, force_tensorflow)
    
    if not force_tensorflow:
        convert_to_tensor = ops.convert_to_tensor
    else:
        import tensorflow as tf
        convert_to_tensor = tf.convert_to_tensor
    
    if isinstance(annot, TensorSpec):
        return convert_to_tensor(value, annot.dtype)
        #return ops.ensure_shape(ops.convert_to_tensor(value, annot.dtype), annot.shape)
    elif annot is None and _should_cast_kwarg(value):
        return convert_to_tensor(value)
    return value

@cache
def _cached_cast_arg(value, annot, force_tensorflow = False):
    return _cast_arg(value, annot, force_tensorflow, cache = False)

def _get_thread_id():
    return threading.current_thread().ident
