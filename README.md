# :yum: Data processing utilities

This repository is a set of utilities for data processing in audio, image and text. It also provides generic functions for data vizualisation (cf `utils/plot_utils.py`). 

It is a part of the [main project](https://github.com/yui-mhcp/base_dl_project) on Deep Learning : it is a submodule used in all other projects for data processing / monitoring. 

## Project structure

```bash
├── example_data        : some data example / plots / ...
├── loggers             : special funny and useful loggers
│   ├── telegram_handler.py : allows to log messages via Telegram bot
│   ├── time_logger.py      : utilities to log functions' performances'
│   ├── tts_handler.py      : allow to log messages via the TTS models
├── unitest             : unitest framework (experimental)
│   ├── tests
│   └── test.py
├── utils               : main utilities directory
│   ├── audio           : audio processing utilities 
│   │   ├── audio_annotation.py     : allows to annotate an audio by adding information to frames
│   │   ├── audio_augmentation.py   : augmentation utilities
│   │   ├── audio_io.py             : loading / writing audio utilities
│   │   ├── audio_processing.py     : some processing functions such as silence trimming
│   │   ├── audio_search.py         : utility class used in the Speech-To-Text project \*
│   │   ├── mkv_utils.py            : processing functions for .mkv files
│   │   └── stft.py                 : multiple implementations of the STFT and MFCC in tensorflow
│   ├── distance        : distance utilities (clustering / comparison)
│   │   ├── clustering.py       : abstract clustering class
│   │   ├── distance_method.py  : some distance functions
│   │   ├── k_means.py          : K-means implementation in tensorflow
│   │   ├── k_propagation.py    : custom algorithm to cluster based on K-most similar data
│   │   └── knn.py              : K-NN implementation in tensorflow
│   ├── image           : image utilities
│   │   ├── box_utils.py            : utilities for Bounding Box drawing / ...
│   │   ├── image_augmentation.py   : some augmentation functions for images
│   │   ├── image_io.py             : image / video writing / loading functions
│   │   ├── image_utils.py          : some utilities
│   │   ├── mask_utils.py           : utility functions for masking
│   │   └── video_utils.py          : video functions (copy / extract audio, ...)
│   ├── text            : functions for text processing (encoding / decoding)
│   │   ├── document_parser         : functions to extract text from documents (experimental)
│   │   │   ├── docx_parser.py
│   │   │   ├── html_parser.py
│   │   │   ├── parser_utils.py
│   │   │   ├── pdf_parser.py
│   │   │   └── txt_parser.py
│   │   ├── bpe.py                  : Byte-Pair-Encoding (BPE) functions
│   │   ├── cleaners.py             : many functions to clean text
│   │   ├── cmudict.py
│   │   ├── f1.py                   : F1-metric function (inspired from SQUAD evaluation script)
│   │   ├── numbers.py              : functions to convert numbers to text
│   │   ├── sentencepiece_encoder.py    : TextEncoder subclass to support sentencepiece encoders
│   │   ├── text_augmentation.py    : some operations for text masking
│   │   ├── text_decoder.py         : functions to decode text (such as Beam Search support)
│   │   ├── text_encoder.py         : main TextEncoder class 
│   │   └── text_processing.py      : some text processing functions (such as splitting)
│   ├── thread_utils        : producer-consumer framework
│   │   ├── consumer.py         : main Consumer class that consumes a stream and produces another
│   │   ├── grouper.py          : special Consumer that groups multiple items
│   │   ├── pipeline.py         : special Consumer that executes multiple tasks with some features
│   │   ├── producer.py         : main Producer class that produces a stream based on a generator
│   │   ├── splitter.py         : special Consumer class that splits an item into multiple items
│   │   ├── threaded_dict.py    : special thread-safe dict-like class with blocking get
│   │   └── threaded_queue.py   : deprecated, prefer to use the Consumer class which is more stable
│   ├── comparison_utils.py : utility functions to compare data
│   ├── embeddings.py       : utility functions to manipulate embeddings (save / load / select)
│   ├── file_utils.py       : loading / saving data from multiple file's formats'
│   ├── generic_utils.py    : generic utility functions
│   ├── pandas_utils.py     : utilities to manipulated pandas.DataFrame
│   ├── plot_utils.py       : plot functions that groups multiple features from matplotlib.pyplot
│   └── sequence_utils.py   : utilities for sequence manipulation (such as pad_batch)
├── example_audio.ipynb
├── example_clustering.ipynb
├── example_generic.ipynb
├── example_image.ipynb
├── example_producer_consumer.ipynb
├── example_text.ipynb
└── example_unitest.ipynb
```

\* This feature is provided in the [Speech To Text project](https://github.com/yui-mhcp/speech_to_text)

## Available features

It is not an exhaustive list of all the available features / functions but an interesting overview of the most useful ones. You can check the example notebooks for more concrete examples. 

- **Audio** (module `utils.audio`)

| Feature   | Fuction / class   | Description |
| :-------- | :---------------- | :---------- |
| display   | `display_audio`   | display audio in jupyter notebook     |
| loading   | `load_audio`  | load audio from multiple formats such as wav, mp3, ...    |
| writing   | `write_audio` | save audio to wav or mp3  | 
| noise reduction   | `reduce_noise`    | use the `noisereduce` library to reduce noise |
| silence trimming  | `trim_silence`    | trim silence (start / end) or remove it       |
| STFT / MFCC       | `MelSTFT class`   | abstract class supporting multiple STFT implementation (compatible with different models), all supporting `tensorflow 2.x graph`|

- **Image** (module `utils.image`) :

| Feature   | Fuction / class   | Description |
| :-------- | :---------------- | :---------- |
| display   | `display_iamge`   | display image in jupyter notebook     |
| loading   | `load_image`  | load image from multiple formats in `tensorflow 2.x`  |
| writing   | `write_image` | facility to `cv2.imwrite` supporting multiple types   |
| writing video | `write_video` | save a list of images as a video  |
| camera streaming  | `stream_camera`   | apply a given function to each camera frame and show result   |
| image augmentation    | `augment_image`   | apply multiple transformation on an image |
| mask color | `create_color_mask`   | create a mask on a given color (with threshold)  |
| mask transform    | `apply_mask`  | apply a transformation on a part of the image     |
| box manipulation  | `box_utils.py`    | many utilities to create, show and transform boxes    |

- **Text** (module `utils.text`) :

| Feature   | Fuction / class   | Description |
| :-------- | :---------------- | :---------- |
| cleaning  | `cleaners.py`    | multiple cleaners for text normalization  |
| encoding / decoding   | `TextEncoder` | class to facilitate text encoding / decoding / cleaning   |
| splitting | `split_text`  | multi-phase text splitting for a more consistent split    |
| Byte-Pair Encoding (BPE)  | `bpe / TextEncoder`   | you can use the `TextEncoder` for token-based encoding and `BPE` based encoding   |

- **Distance** (module `utils.distance`) :

| Feature   | Fuction / class   | Description |
| :-------- | :---------------- | :---------- |
| distance  | `distance`    | allow computing distance supporting multiple metrics (`euclidian`, `manhattan`, `levenstein`) |
| K-NN      | `KNN`     | fully optimizerd KNN implementation in `tensorflow 2.x graph` |
| KMeans    | `KMeans`  | implementation of the `KMeans` clustering algorithm ein `tensorflow` (also supports methodology to determine best value for `k`)  |
| KPropagation  | `KPropagation`    | custom clustering algorithm to propagate label based on a `similarity_matrix` (useful for the [Siamese Networks](https://github.com/yui-mhcp/siamese_networks)) |

- **Generic** (module `utils`) :

| Feature   | Fuction / class   | Description |
| :-------- | :---------------- | :---------- |
| plot      | `plot`        | utility that combines multiple `matplotlib` functions in one bigger function |
| plot spectrogram  | `plot_spectrogram`    | utility that calls `plot` with some default parameters specific to spectrograms    |
| plot embeddings   | `plot_embedding`  | utility that calls `plot` after creating a 2D representation of N-D vectors (the embeddings)  |
| plot confusion matrix | `plot_confusion_matrix`  | creates a heatmap representing the confusion matrix (based on labels / predictions).  |
| load / dump data  | `{load / dump}_data`  | utilities for safe json file manipulation |
| DataFrame manipulation    | `pandas_utils.py` | utilities to aggregate / filter DataFrame |
| embeddings manipulation   | `embeddings.py`   | utilities to load / save / ... embeddings data    |

## Installation and usage

1. Clone this repository : `git clone https://github.com/yui-mhcp/data_processing.git`
2. Go to the root of this repository : `cd data_processing`
3. Install requirements : `pip install -r requirements.txt`
4. Open an example notebook and follow the instructions !

The `utils/{data_type}` modules are note loaded by default meaning that it is not required to install the requirements for a given submodule if you do not want to use it. In this case, you can simply remove the submodule and run the `pipreqs` command to compute a new `requirements.txt` file.

**For audio processing** : you should also install `ffmpeg` if you want to use some audio processing functions (especially the `.mp3` support).

## TO-DO list

- [x] Make example for audio processing
- [x] Make example for image processing
- [x] Make example for text processing
- [x] Make example for plot utils
- [x] Make example for embeddings
- [x] Make example for the `producer-consumer` utility
- [ ] Make example for audio annotation
- [x] Comment all functions to facilitate their comprehension / usage

## Future improvments 

- Audio processing :
    - [x] Improve the audio annotation procedure.
    - [ ] See how to improve the trimming / remove silence processing
- Image processing :
    - [x] Clean and optimize the code.
    - [x] Add image loading / writing support.
    - [x] Add video loading / writing support.
    - [ ] Improve the `stream_camera` to better synchronize the frames with audio (when `play_audio = True`)
- Text processing :
    - [x] Add support for token-splitting instead of word-splitting in the `TextEncoder`.
    - [x] Add better support for Transformers. 
    - [x] Add support for `sentencepiece` encoders.
    - [x] Add text extraction from documents (experimental)
        - [x] Add support for .txt
        - [x] Add support for .pdf
        - [x] Add support for .docx
        - [x] Add support for .html
        - [ ] Add support for .epub
- Thread utilities :
    - [x] Add the possibility to create a pipeline based on a list of functions.
    - [x] Allow to plot a `producer-consumer` pipeline with `graphviz`. 
    - [ ] Add `unitest` to ensure correctness. 
- Plot functions :
    - [x] Better support subplots.
    - [x] Allow to plot embeddings as subplot.
    - [ ] Add 3D plot support.

## Pipeline-based prediction

Some models (in other projects) support the `pipeline-based` prediction, meaning that tasks are executed within the `Producer Consumer` framework. You can disable the multi-threading feature by passing `max_workers = -1 (or -2)` to the prediction's functions but it will obviously slow down the global prediction time. 

It is important to notice that it does **not** reduce the processing time of a single data, which still has to pass through all the functions (at least for sequential tasks). However it can improve a lot the global performances as the total time is now `the number of data * longest_task_time` instead of `the number of data * pipeline time`, which makes a huge difference in this case. 

Another benefit is that you can add tasks in parallel of other tasks : this is useful for streaming where the video writing / frame showing can be performed in parallel of the pipeline (check [the example notebook](example_producer_consumer.ipynb) for the illustration). 

The framework also supports batching (`batch_size` argument) and result saving (`filename` argument of the `Pipeline` object) which are useful features in model prediction. 

## Contacts and licence

You can contact [me](https://github.com/yui-mhcp) at yui-mhcp@tutanota.com or on [discord](https://discord.com) at `yui#0732`

The objective of these projects is to facilitate the development and deployment of useful application using Deep Learning for solving real-world problems and helping people. 
For this purpose, all the code is under the [Affero GPL (AGPL) v3 licence](LICENCE)

All my projects are "free software", meaning that you can use, modify, deploy and distribute them on a free basis, in compliance with the Licence. They are not in the public domain and are copyrighted, there exist some conditions on the distribution but their objective is to make sure that everyone is able to use and share any modified version of these projects. 

Furthermore, if you want to use any project in a closed-source project, or in a commercial project, you will need to obtain another Licence. Please contact me for more information. 

For my protection, it is important to note that all projects are available on an "As Is" basis, without any warranties or conditions of any kind, either explicit or implied. However, do not hesitate to report issues on the repository's project or make a Pull Request to solve it :smile: 

If you use this project in your work, please add this citation to give it more visibility ! :yum:

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

- [1] The text processing part is highly inspired from [NVIDIA tacotron2](https://github.com/NVIDIA/tacotron2) repository which I modified to be used in tensorflow. Same for the `audio/stft.py` which I optimized in tensorflow and added support for other kind of STFT computation (inspired from Jasper, DeepSpeech, ...) projects.

- [2] The provided embeddings in `example_data/embeddings/embeddings_256_voxforge.csv` are samples from the [VoxForge](http://www.voxforge.org/) dataset and embedded with my [AudioSiamese](https://github.com/yui-mhcp/siamese_networks) `audio_siamese_256_mel_lstm` model.

