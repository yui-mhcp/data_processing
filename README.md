# :yum: Data processing utilities

Check the [CHANGELOG](https://github.com/yui-mhcp/yui-mhcp/blob/main/CHANGELOG.md) file to have a global overview of the latest updates / new features ! :yum:

## Project structure

Check the provided notebooks to have an overview of the available features !

```bash
├── example_data        : data used for the demonstrations
├── loggers             : custom utilities for the `logging` module
│   ├── __init__.py         : defines useful utilities to control `logging`
│   ├── telegram_handler.py : custom logger using the telegram bot api
│   ├── time_logging.py     : custom timer features
│   └── tts_handler.py      : custom logger using the Text-To-Speech models
├── tests               : custom unit-testing for the different modules
│   ├── data               : test data files
│   ├── __reproduction     : expected output files for reproducibility tests
│   └── test_*.py          : test file
├── utils
│   ├── audio                   : audio utilities
│   │   ├── audio_annotation.py     : annotation features for new TTS/STT dataset creation
│   │   ├── audio_io.py             : audio loading / writing
│   │   ├── audio_player.py         : audio playback functionality
│   │   ├── audio_processing.py     : audio normalization / processing
│   │   ├── audio_recorder.py       : audio recording functionality
│   │   ├── audio_stream.py         : audio streaming support
│   │   ├── mkv_utils.py            : processing for .mkv video format
│   │   ├── noisereducev1.py        : maintained version of the old `noisereduce` library
│   │   └── stft.py                 : implementations of various mel-spectrogram methods
│   ├── callbacks               : callback management system
│   │   ├── __init__.py
│   │   ├── callback.py             : base callback implementation
│   │   ├── displayer.py            : display-related callbacks
│   │   ├── file_saver.py           : file saving callbacks
│   │   └── function_callback.py    : function-based callbacks
│   ├── databases               : custom storage features
│   │   ├── vectors                 : vector storage
│   │   │   ├── faiss_index.py          : vector index using `faiss`
│   │   │   ├── keras_index.py          : vector index using `keras`
│   │   │   ├── numpy_index.py          : vector index using `numpy`
│   │   │   ├── torch_index.py          : vector index using `torch`
│   │   │   └── vector_index.py         : abstract vector index
│   │   ├── database.py             : abstract database
│   │   ├── database_wrapper.py     : database wrapping another database
│   │   ├── json_dir.py             : database storing each entry in a `.json` file
│   │   ├── json_file.py            : basic implementation storing all data in a `dict`
│   │   ├── json.py                 : optimized `json`-based data storage
│   │   ├── ordered_database_wrapper.py : wrapper that keeps track of insertion order (like an `OrderedDict`)
│   │   └── vector_database.py      : wrapper storing both data and vectors
│   ├── datasets                : dataset utilities
│   │   ├── audio_datasets          : audio dataset implementations
│   │   │   ├── common_voice.py         : Mozilla Common Voice dataset
│   │   │   ├── libri_speech.py         : LibriSpeech dataset
│   │   │   ├── processing.py           : audio dataset processing
│   │   │   ├── siwis.py                : SIWIS dataset
│   │   │   └── voxforge.py             : VoxForge dataset
│   │   ├── builder.py               : dataset building utilities
│   │   ├── loader.py                : dataset loading utilities
│   │   └── summary.py               : dataset summary tools
│   ├── image                   : image features
│   │   ├── bounding_box            : features for bounding box manipulation
│   │   │   ├── combination.py          : combines group of boxes
│   │   │   ├── converter.py            : box format conversion
│   │   │   ├── filters.py              : box filtering
│   │   │   ├── locality_aware_nms.py   : LA-NMS implementation
│   │   │   ├── metrics.py              : box metrics (IoU, etc.)
│   │   │   ├── non_max_suppression.py  : NMS implementation
│   │   │   ├── processing.py           : box processing
│   │   │   └── visualization.py        : box extraction / drawing
│   │   ├── video                   : utilities for video I/O and stream
│   │   │   ├── ffmpeg_reader.py        : video reader using `ffmpeg-python`
│   │   │   ├── http_screen_mirror.py   : custom camera reading frames from the `HttpScreenMirror` app
│   │   │   ├── filters.py              : box filtering
│   │   │   ├── streaming.py            : camera streaming utilities
│   │   │   └── writer.py               : video writers (`OpenCV` and `ffmpeg-python` are currently supported)
│   │   ├── image_io.py             : image loading / writing
│   │   ├── image_normalization.py  : normalization schema
│   │   └── image_processing.py     : image processing utilities
│   ├── keras                   : keras and hardware acceleration utilities
│   │   ├── ops                     : operation interfaces for different backends
│   │   │   ├── builder.py              : operation builder
│   │   │   ├── core.py                 : core operations
│   │   │   ├── execution_contexts.py   : execution context management
│   │   │   ├── image.py                : image operations
│   │   │   ├── linalg.py               : linear algebra operations
│   │   │   ├── math.py                 : mathematical operations
│   │   │   ├── nn.py                   : neural network operations
│   │   │   ├── numpy.py                : numpy-compatible operations
│   │   │   └── random.py               : random operations
│   │   ├── runtimes                : model runtime implementations
│   │   │   ├── onnx_runtime.py         : ONNX runtime
│   │   │   ├── runtime.py              : base runtime class
│   │   │   ├── saved_model_runtime.py  : saved model runtime
│   │   │   ├── tensorrt_llm_runtime.py : TensorRT LLM runtime
│   │   │   └── tensorrt_runtime.py     : TensorRT runtime
│   │   ├── compile.py              : graph compilation features
│   │   └── gpu.py                  : GPU utilities
│   ├── text                    : text-related features
│   │   ├── abreviations
│   │   ├── parsers                 : document parsers (new implementation)
│   │   │   ├── combination.py      : box combination for parsing
│   │   │   ├── docx_parser.py      : DOCX document parser
│   │   │   ├── java_parser.py      : Java code parser
│   │   │   ├── md_parser.py        : Markdown parser
│   │   │   ├── parser.py           : base parser implementation
│   │   │   ├── pdf_parser.py       : PDF parser
│   │   │   ├── py_parser.py        : Python code parser
│   │   │   └── txt_parser.py       : text file parser
│   │   ├── cleaners.py             : text cleaning methods
│   │   ├── ctc_decoder.py          : CTC-decoding
│   │   ├── metrics.py              : text evaluation metrics
│   │   ├── numbers.py              : numbers cleaning methods
│   │   ├── paragraphs_processing.py   : paragraphs processing functions
│   │   ├── sentencepiece_tokenizer.py : sentencepiece tokenizer interface
│   │   ├── text_processing.py      : text processing functions
│   │   ├── tokenizer.py            : tokenizer implementation
│   │   └── tokens_processing.py    : token-level processing
│   ├── threading               : threading utilities
│   │   ├── async_result.py        : asynchronous result handling
│   │   ├── priority_queue.py      : priority queue with order consistency
│   │   ├── process.py             : process management
│   │   └── stream.py              : data streaming implementation
│   ├── comparison_utils.py     : convenient comparison features for various data types
│   ├── distances.py            : distance and similarity metrics
│   ├── embeddings.py           : embeddings saving / loading
│   ├── file_utils.py           : data saving / loading
│   ├── generic_utils.py        : generic features 
│   ├── plot_utils.py           : plotting functions
│   ├── sequence_utils.py       : sequence manipulation
│   └── wrappers.py             : function wrappers and decorators
├── example_audio.ipynb
├── example_custom_operations.ipynb
├── example_generic.ipynb
├── example_image.ipynb
├── example_text.ipynb
├── LICENSE
├── Makefile
├── README.md
└── requirements.txt
```

The `loggers` module is independant from the `utils` one, making it easily reusable / extractable.

## Installation and usage

See [the installation guide](https://github.com/yui-mhcp/blob/master/INSTALLATION.md) for a step-by-step installation :smile:

Here is a summary of the installation procedure, if you have a working python environment :
1. Clone this repository : `git clone https://github.com/yui-mhcp/data_processing.git`
2. Go to the root of this repository : `cd data_processing`
3. Install requirements : `pip install -r requirements.txt`
4. Open an example notebook and follow the instructions !

**Important Notes** :
- The `utils/{audio / image / text}` modules are not loaded by default, meaning that it is not required to install the requirements for a given submodule if you do not want to use it. In this case, you can simply remove the submodule and run the `pipreqs` command to compute a new `requirements.txt` file !
- The `keras` module is not imported by default, and most of the features are available without ever importing it ! :smile:
- The `requirements.txt` file does not include any backend (i.e., `tensorflow`, `torch`, `jax`, etc.), so make sure to manually install it if necessary !

## TO-DO list

- [x] Make example for audio processing
- [x] Make example for image processing
- [x] Make example for text processing
- [x] Make example for plot utils
- [x] Make example for embeddings manipulation
- [x] Make the code keras-3 compatible
- [x] Remove `keras` from dependencies (i.e., features that do not require `keras` should work even if `keras` is not installed)
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
- [x] Enable the `read_audio` function in `tf.data` pipeline

### Image

- [x] Add image loading / writing support
- [x] Add video loading / writing support
- [x] Add support for rotated bounding boxes
- [x] Implement a keras 3 Non-Maximal Suppression (NMS)
- [x] Implement the [Locality-Aware NMS (LaNMS)](https://github.com/argman/EAST)

### Text

- [x] Support text tokenization/encoding in `tf.data` pipeline
- [x] Implement text cleaning
    - [x] Abreviation extensions
    - [x] Time / dollar / number extensions
    - [x] unicode convertion
- [x] Support token-splitting instead of word-splitting in `Tokenizer`
- [x] Support `transformers` tokenizers convertion 
- [x] Support `sentencepiece` tokenizers
- [x] Extract text from documents
    - [x] `.txt`
    - [x] `.md`
    - [x] `.pdf`
    - [x] `.docx`
    - [x] `.html`
    - [ ] `.epub`
- [x] Implement token-based logits masking
- [x] Implement batch text encoding
- [x] Add custom tokens to `Tokenizer`
- [x] Add `CTC`-decoding

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

## Notes and references 

- The text cleaning module (`text.cleaners`) is inspired from [NVIDIA tacotron2](https://github.com/NVIDIA/tacotron2) repository. Their implementation of `Short-Time Fourrier Transform (STFT)` is also available in `audio/stft.py`, adapted in `keras 3`.

- The provided embeddings in `example_data/embeddings/embeddings_256_voxforge.csv` has been generated based on samples of the [VoxForge](http://www.voxforge.org/) dataset, and embedded with an [AudioSiamese](https://github.com/yui-mhcp/siamese_networks) model (`audio_siamese_256_mel_lstm`).

Tutorials :
- The [Keras 3 API](https://keras.io/api/) which has been (*partially*) adapted in the `keras_utils/ops` module to enable `numpy` backend, and `tf.data` compatibility
- The [tf.function](https://www.tensorflow.org/guide/function?hl=fr) guide

## Contacts and licence

Contacts :
- **Mail** : `yui-mhcp@tutanota.com`
- **[Discord](https://discord.com)** : yui0732

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See the [LICENSE](LICENSE) file for details.

This license allows you to use, modify, and distribute the code, as long as you include the original copyright and license notice in any copy of the software/source. Additionally, if you modify the code and distribute it, or run it on a server as a service, you must make your modified version available under the same license.

For more information about the AGPL-3.0 license, please visit [the official website](https://www.gnu.org/licenses/agpl-3.0.html)

## Citation

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