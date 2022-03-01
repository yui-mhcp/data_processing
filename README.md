# :yum: Data processing utilities

This repository is a set of utilities for data processing in audio, image and text and provide generic functions for data vizualisation. 

This is part of the [main project](https://github.com/yui-mhcp/base_dl_project) on Deep Learning : it is a submodule used in all other projects for data processing / monitoring. 

## Project structure

```bash
├── audio/          : audio processing utilities
│   ├── audio_annotation.py     : utility class to easily annotate audios
│   ├── audio_augmentation.py   : utilities for audio augmentation
│   ├── audio_io.py             : utilities for audio loading / writing
│   ├── audio_processing.py     : audio normalization / silence trimming / ...
│   ├── audio_search.py         : utility class to visualize text search on audio file \*
│   ├── mkv_utils.py            : utilities to parse .mkv file for audio annotation
│   └── stft.py                 : multiple implementations of mel-spectrogram / STFT
├── distance/       : clustering / distance metrics
│   ├── clustering.py           : will be improve to support clustering functions
│   ├── distance_method.py      : some distance functions
│   ├── k_means.py              : tensorflow implementation of the `KMeans` algorithm
│   ├── k_propagation.py        : custom clustering algorithm used in the Siamese Networks clustering
│   └── knn.py                  : tensorflow implementation of `K-NN` algorithm
├── image/          : image processing and bounding-box utilities
│   ├── box_utils.py            : bounding-box utilities
│   ├── image_augmentation.py   : image augmentation methods
│   ├── image_io.py             : loading / writing images (will be improved)
│   ├── image_utils.py          : general utilities
│   ├── mask_utils.py           : utilities for image masking / mask exctraction
│   └── video_utils.py          : utilities for video processing
├── text/           : text processing (encoding / decoding / normalization)
│   ├── cleaners.py             : general cleaning functions for text normalization
│   ├── cmudict.py              : CMUdict (from `NVIDIA`'s repository' used in tacotron2)
│   ├── numbers.py              : expand numbers to text / numbers normalization
│   ├── text_encoder.py         : utility class for text encoding / decoding efficiently
│   └── text_processing.py      : functions for text processing
├── embeddings.py           : utilities for embeddings loading / saving / processing
├── generic_utils.py        : generic functions such as loading / saving json, ...
├── pandas_utils.py         : utilities for pd.DataFrame processing
├── plot_utils.py           : custom functions to make fancy graphs !
└── thread_utils.py         : `ThreadPool` class to parallelize tasks

```

\* This feature is provided in the [Speech To Text project](https://github.com/yui-mhcp/speech_to_text)

## Available features

- **Audio** (module `utils.audio`)

| Feature   | Fuction / class   | Description |
| :-------- | :---------------- | :---------- |
| display   | `display_audio`   | display audio in jupyter notebook     |
| loading   | `load_audio`  | load audio from multiple formats such as wav, mp3, ...    |
| writing   | `write_audio` | save audio to wav or mp3  | 
| noise reduction   | `reduce_noise`    | use the `noisereduce` library to reduce noise |
| silence detection | `trim_silence`    | trim silence (start / end) or remove it       |
| STFT / FMCC       | `MelSTFT class`   | abstract class supporting multiple STFT implementation (compatible with different models), all supporting `tensorflow 2.x graph`|

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
| load / dump json  | `{load / dump}_json`  | utilities for safe json file manipulation |
| DataFrame manipulation    | `pandas_utils.py` | utilities to aggregate / filter DataFrame |
| embeddings manipulation   | `embeddings.py`   | utilities to load / save / ... embeddings data    |

## Installation and usage

1. Clone this repository : `git clone https://github.com/yui-mhcp/data_processing.git`
2. Go to the root of this repository : `cd data_processing`
3. Install requirements : `pip install -r requirements.txt`
4. Open an example notebook and follow the instructions !

**For audio processing** : you should also install `ffmpeg` if you want to use some audio processing functions.

## TO-DO list

- [x] Make example for audio processing
- [x] Make example for image processing
- [x] Make example for text processing
- [x] Make example for plot utils
- [x] Make example for embeddings
- [ ] Make example for audio annotation
- [x] Comment all functions to facilitate their comprehension / usage

## Future improvments 

- Audio processing
    - [x] Try to improve the audio annotation procedure.
    - [ ] See how to improve the trimming / remove silence processing
- Image processing.
    - [x] Clean and optimize the code.
    - [x] Add image loading / writing support.
- Text processing
    - [x] Add support for token-splitting instead of word-splitting in the `TextEncoder`.
    - [x] Add better support for Transformers. 
- Plot functions :
    - [x] Better support subplots.
    - [ ] Allow to plot embeddings as subplot.
    - [ ] Add 3D plot support.

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

