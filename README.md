# :yum: Data processing utilities

Check the [CHANGELOG](https://github.com/yui-mhcp/yui-mhcp/blob/main/CHANGELOG.md) file to have a global overview of the latest updates / new features ! :yum:

## Project structure

Check the provided notebooks to have an overview of the available features !

```bash
├── example_data        : data used for the demonstrations
├── loggers             : custom utilities for the `logging` module
│   ├── utils       : subset of the utils module to make loggers fully independant
│   ├── __init__.py         : defines useful utilities to control `logging`
│   ├── telegram_handler.py : custom logger using the telegram bot api
│   ├── time_logging.py     : custom timer features
│   └── tts_handler.py      : custom logger using the Text-To-Speech models
├── unitests            : custom unit-testing for the different `utils` modules
│   ├── __init__.py
│   ├── test_custom_objects.py  : not executed in this project (requires `custom_train_objects`)
│   ├── test_layers_masking.py  : not executed in this project (requires `custom_layers`)
│   ├── test_transformers.py    : test `transformers.AutoTokenizer` convertion to `TextEncoder`
│   ├── test_utils_audio.py
│   ├── test_utils_boxes.py
│   ├── test_utils_compile.py
│   ├── test_utils_clustering.py
│   ├── test_utils_distance.py
│   ├── test_utils_embeddings.py
│   ├── test_utils_image.py
│   ├── test_utils_ops.py
│   ├── test_utils_text.py
│   └── test_utils_threading.py
├── utils
│   ├── audio                   : audio utilities
│   │   ├── __init__.py
│   │   ├── audio_annotation.py     : annotation features for new TTS/STT dataset creation
│   │   ├── audio_augmentation.py   : augmentation methods for audio / mel data
│   │   ├── audio_io.py             : audio loading / writing
│   │   ├── audio_processing.py     : audio normalization / processing
│   │   ├── audio_search.py         : custom search in audio / video based on transcript
│   │   ├── mkv_utils.py            : processing for .mkv video format
│   │   ├── noisereducev1.py        : maintained version of the old `noisereduce` library
│   │   └── stft.py                 : implementations of various mel-spectrogram methods
│   ├── distance                : distance / similarity functions
│   │   ├── __init__.py
│   │   ├── clustering.py           : clustering wrappers / base features
│   │   ├── distance_method.py      : distance / similarity metrics computation
│   │   ├── kmeans_method.py        : K-Means implementation
│   │   ├── knn_method.py           : K-Nearest Neighbor implementation
│   │   ├── label_propagation_method.py : custom clustering algorithm
│   │   ├── spectral_clustering_method.py   : spectral-clustering implementation
│   │   └── text_distance_method.py     : text-based similarity metrics (e.g., F1)
│   ├── image                   : image features
│   │   ├── bounding_box            : features for bounding box manipulation (for object detection)
│   │   │   ├── __init__.py
│   │   │   ├── combination.py          : (experimental) combines group of boxes
│   │   │   ├── converter.py            : box convertion format
│   │   │   ├── iou.py                  : Intersection over Union implementation
│   │   │   ├── locality_aware_nms.py   : (experimental) LA-NMS implementation
│   │   │   ├── non_max_suppression.py  : (experimental) non-max suppression (NMS) implementation
│   │   │   ├── polygons.py             : polygon manipulation for the EAST model
│   │   │   ├── processing.py           : box processing
│   │   │   └── visualization.py        : box extraction / drawing
│   │   ├── __init__.py
│   │   ├── custom_cameras.py       : custom camera for the HTTPScreenMirror app
│   │   ├── image_augmentation.py   : image augmentation methods
│   │   ├── image_io.py             : image loading / writing / camera streaming features
│   │   ├── image_normalization.py  : normalization schema
│   │   ├── image_utils.py          : custom image functions (resizing, padding, ...)
│   │   ├── mask_utils.py           : masking utilities
│   │   └── video_utils.py          : (experimental) basic functions for video manipulation
│   ├── keras_utils             : custom keras operations (see `example_custom_operations.ipynb`)
│   │   ├── ops                     : interfaces the main keras / numpy operations
│   │   │   ├── __init__.py
│   │   │   ├── core.py
│   │   │   ├── image.py
│   │   │   ├── linalg.py
│   │   │   ├── math.py
│   │   │   ├── nn.py
│   │   │   ├── numpy.py
│   │   │   ├── ops_builder.py
│   │   │   └── random.py
│   │   ├── __init__.py
│   │   ├── compile.py          : custom graph compilation features
│   │   └── gpu_utils.py        : (experimental) custom gpu features
│   ├── search              : utilities related to information retrieval
│   │   ├── vectors
│   │   │   ├── __init__.py
│   │   │   ├── base_vectors_db.py  : BaseVectorsDB abstraction class
│   │   │   └── dense_vectors.py    : DenseVectors class
│   ├── text                : text-related features
│   │   ├── abreviations
│   │   │   └── en.json
│   │   ├── document_parser         : text extraction from documents
│   │   │   ├── pdf_parser
│   │   │   │   ├── __init__.py         : main parsing method
│   │   │   │   ├── combination.py      : subset of utils/image/bounding_box/combination.py
│   │   │   │   ├── pdfminer_parser.py  : parser based on the pdfminer extraction library
│   │   │   │   ├── post_processing.py  : post processing for pypdfium2_parser
│   │   │   │   ├── pypdf_parser.py     : parser based on the pypdf extraction library
│   │   │   │   └── pypdfium2_parser.py : parser based on the pypdfium2 extraction library
│   │   │   ├── __init__.py
│   │   │   └── docx_parser.py
│   │   │   ├── html_parser.py
│   │   │   └── md_parser.py
│   │   │   ├── parser.py
│   │   │   ├── parser_utils.py
│   │   │   └── txt_parser.py
│   │   ├── __init__.py
│   │   ├── byte_pair_encoding.py   : BPE implementation
│   │   ├── cleaners.py             : text cleaning methods
│   │   ├── ctc_decoder.py          : (experimental) CTC-decoding
│   │   ├── numbers.py              : numbers cleaning methods
│   │   ├── sentencepiece_encoder.py    : custom encoder interfacing with the sentencepiece library
│   │   ├── text_augmentation.py    : (experimental) token masking methods
│   │   ├── text_encoder.py         : TextEncoder class
│   │   └── text_processing.py      : custom text / logits processing functions
│   ├── threading               : custom producer-consumer methods
│   │   ├── __init__.py
│   │   ├── consumer.py         : multi-threaded consumer with observers
│   │   ├── priority_queue.py   : custom `PriorityQueue` with order consistency
│   │   ├── producer.py         : multi-threaded generator with observers
│   │   └── threaded_dict.py    : thread-safe `dict` with blocking get
│   ├── __init__.py
│   ├── comparison_utils.py     : convenient comparison features for various data types
│   ├── embeddings.py           : embeddings saving / loading
│   ├── file_utils.py           : data saving / loading
│   ├── generic_utils.py        : generic features 
│   ├── pandas_utils.py         : pandas custom features
│   ├── plot_utils.py           : plotting functions
│   ├── sequence_utils.py       : sequence manipulation
│   ├── stream_utils.py         : function streaming interface
│   └── wrapper_utils.py        : custom wrappers
├── LICENSE
├── Makefile
├── README.md
├── example_audio.ipynb
├── example_clustering.ipynb
├── example_custom_operations.ipynb
├── example_generic.ipynb
├── example_image.ipynb
├── example_producer_consumer.ipynb
├── example_text.ipynb
└── requirements.txt

```

The `loggers` module is independant from the `utils` one, making it easily reusable / extractable.

## Installation and usage

1. Clone this repository : `git clone https://github.com/yui-mhcp/data_processing.git`
2. Go to the root of this repository : `cd data_processing`
3. Install requirements : `pip install -r requirements.txt`
4. Open an example notebook and follow the instructions ! The `make` command starts a `jupyter lab` server
5. (Optional) run the tests : `python3 -m unittest discover -v -t . -s unitests -p test_*`

**Some tests fail with the JAX backend, this will be solved in the next updates**

The `utils/{audio / image / text}` modules are not loaded by default, meaning that it is not required to install the requirements for a given submodule if you do not want to use it. In this case, you can simply remove the submodule and run the `pipreqs` command to compute a new `requirements.txt` file !

**For audio processing** : `ffmpeg` is required for some audio processing functions (especially the `.mp3` support).

**Important Note** : no backend (i.e., `tensorflow`, `torch`, ...) is installed by default, so make sure to properly install them before ! The `tensorflow` library is currently required when importing `utils.text` (other modules do not import `tensorflow by default`). The `tensorflow` removal is in progress and may not be perfectly working ;)

## TO-DO list

- [x] Make example for audio processing
- [x] Make example for image processing
- [x] Make example for text processing
- [x] Make example for plot utils
- [x] Make example for embeddings manipulation
- [x] Make example for the `producer-consumer` utility
- [x] Make the code keras-3 compatible
    - [x] Update the `audio` module
    - [x] Update the `image` module
    - [x] Update the `text` module
    - [x] Update the `utils` module
    - [x] Make unit-testing to check correctness for the different keras backends
    - [x] Make unit-testing for the `graph_compile` and `executing_eagerly` in all backends
    - [x] Make every function compatible with `tf.data`, no matter the keras backend (see `example_custom_ops.ipynb` for more information)
- [x] Enable any backend to be aware of XLA/eager execution (i.e., `executing_eagerly` function)
- [x] Enable `graph_compile` to support all backends compilation
    - [x] `tensorflow` backend (`tf.function`)
    - [x] `torch` backend (`torch.compile`)
    - [x] `jax` backend (`jax.jit`)
- [x] Auto-detect `static_argnames` for the `jax.jit` compilation
- [x] Allow `tf.function` with `graph_compile` regardless of the `keras` backend
- [ ] Add GPU features for all backends
    - [x] `tensorflow` backend
    - [ ] `torch` backend
    - [ ] `jax` backend

### Audio 

- [x] Extract audio from videos
- [x] Enables audio playing without `IPython.display` autoplay feature
- [x] Implement specific `Mel spectrogram` implementations
    - [x] [Tacotron-2](https://github.com/NVIDIA/tactron2)
    - [x] [Whisper](https://github.com/whisper)
    - [x] DeepSpeech 2
    - [x] Conformer
- [x] Run the `read_audio` function in `tf.data` pipeline
- [x] Support audio formats :
    - [x] `wav`
    - [x] `mp3`
    - [x] Any `librosa` format
    - [x] Any `ffmpeg` format (video support)

### Image

- [x] Add image loading / writing support
- [x] Add video loading / writing support
- [x] Add support for segmentation masking
    - [x] Add support for polygon masks
    - [ ] Add support for RLE masks
- [x] Add support for rotated bounding boxes
- [x] Implement a keras 3 Non-Maximal Suppression (NMS)
- [x] Implement the [Locality-Aware NMS (LaNMS)](https://github.com/argman/EAST)

### Text

- [x] Support text encoding in `tf.data` pipeline
- [x] Implement text cleaning
    - [x] Abreviation extensions
    - [x] Time / dollar / number extensions
    - [x] unicode convertion
- [x] Support token-splitting instead of word-splitting in `TextEncoder`
- [x] Support `transformers` tokenizers convertion 
- [x] Support `sentencepiece` encoders
- [x] Extract text from documents
    - [x] `.txt`
    - [x] `.md`
    - [x] `.pdf`
    - [x] `.docx`
    - [x] `.html`
    - [ ] `.epub`
- [x] Implement token-based logits masking
- [x] Implement batch text encoding
- [x] Add custom tokens to `TextEncoder`
- [x] Implement CTC-decoding in keras 3 (µalready implemented in `keras 3.3`*)

### Generic utilities

- [x] Make subplots easier to use via `args` and `kwargs`
- [x] Make custom plot functions usable with `plot_multiple`
- [x] Add 3D plot / subplot support
- [x] Implement custom plotting functions
    - [x] Spectrogram / attention weights
    - [x] Audio waveform
    - [x] Embeddings (*d*-dimensional vectors projected in 2D space)
    - [x] 3D volumes
    - [x] Classification result
    - [x] Confusion matrix (or any matrix)

## Contacts and licence

Contacts :
- **Mail** : `yui-mhcp@tutanota.com`
- **[Discord](https://discord.com)** : yui0732

### Terms of use

The goal of these projects is to support and advance education and research in Deep Learning technology. To facilitate this, all associated code is made available under the [GNU Affero General Public License (AGPL) v3](AGPLv3.licence), supplemented by a clause that prohibits commercial use (cf the [LICENCE](LICENCE) file).

These projects are released as "free software", allowing you to freely use, modify, deploy, and share the software, provided you adhere to the terms of the license. While the software is freely available, it is not public domain and retains copyright protection. The license conditions are designed to ensure that every user can utilize and modify any version of the code for their own educational and research projects.

If you wish to use this project in a proprietary commercial endeavor, you must obtain a separate license. For further details on this process, please contact me directly.

For my protection, it is important to note that all projects are available on an "As Is" basis, without any warranties or conditions of any kind, either explicit or implied. However, do not hesitate to report issues on the repository's project, or make a Pull Request to solve it :smile: 

### Citation

If you find this project useful in your work, please add this citation to give it more visibility ! :yum:

```
@misc{yui-mhcp
    author  = {yui},
    title   = {A Deep Learning projects centralization},
    year    = {2021},
    publisher   = {GitHub},
    howpublished    = {\url{https://github.com/yui-mhcp}}
}
```

## Notes and references 

- The text cleaning module (`text.cleaners`) is inspired from [NVIDIA tacotron2](https://github.com/NVIDIA/tacotron2) repository. Their implementation of `Short-Time Fourrier Transform (STFT)` is also available in `audio/stft.py`, adapted in `keras 3`.

- The provided embeddings in `example_data/embeddings/embeddings_256_voxforge.csv` has been generated based on samples of the [VoxForge](http://www.voxforge.org/) dataset, and embedded with an [AudioSiamese](https://github.com/yui-mhcp/siamese_networks) model (`audio_siamese_256_mel_lstm`).

Tutorials :
- The [Keras 3 API](https://keras.io/api/) which has been (*partially*) adapted in the `keras_utils/ops` module to enable `numpy` backend, and `tf.data` compatibility
- The [tf.function](https://www.tensorflow.org/guide/function?hl=fr) guide 