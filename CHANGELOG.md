# :yum: CHANGELOG :yum:

This file tracks the major updates of the [data processing](https://github.com/yui-mhcp/data_processing) project. For the global, cross-project overview of all the `yui-mhcp` repositories, see the [central CHANGELOG](https://github.com/yui-mhcp/yui-mhcp/blob/main/CHANGELOG.md) :smile:

## [v1.0.0] - 21/07/2026 - Major refactoring release ! :yum:

First tagged release (`v1.0.0`) following a large refactoring of the shared `utils` / `loggers` foundation !

### Major updates

- The installation now relies on `pyproject.toml` instead of `requirements.txt` : install with `pip install -e .` and pick the extras you need (`audio`, `image`, `text`, `datasets`, `plot`, an optional keras backend via `tf` / `torch` / `keras`, and `dev`)
- The test suite has been migrated from `unittest` to `pytest`, mirroring the `utils` / `loggers` tree, with auto-skipped markers (`tensorflow`, `torch`, `keras`, `cv2`, `gpu`, `slow`, ...)
- The minimum supported Python is now `3.10` (structural pattern matching), and the version is single-sourced in `__version__.py`
- **[BREAKING CHANGE]** `utils.embeddings` is now a package : the embeddings I/O and processing live in `utils.embeddings.{embeddings_io, embeddings_processing}`
- **[BREAKING CHANGE]** the vector indices have moved from `utils.databases.vectors` to `utils.embeddings.index` (the `faiss` index has been removed)
- **[BREAKING CHANGE]** `loggers.tts_handler` has been removed ; a new `loggers.routing` module now handles level-based routing / formatting
- **[BREAKING CHANGE]** `utils.text.parsers.combination` has been replaced by `utils.text.parsers.html_parser`, and a new (experimental) `utils.text.web` module has been added

### Minor updates

- New `utils.image.masking` module ; the experimental `utils.image.video.{filters, streaming}` have been removed
- New `keras` runtimes : `keras_runtime`, `hf_runtime`, `custom_model_runner_cpp` and `tensorrt_llm_bert_runtime`
- New threading utilities : `inflight_batcher` and `stream_request_manager`
- A new `example_plot.ipynb` notebook demonstrates the `plot_utils` features
