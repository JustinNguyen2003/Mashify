# Mashify

CSC 475 (Music Information Retrieval) Group Project

## Requirements

**NOTE:** Use a version of Python 3.11 (known to work with 3.11.11)

### SSM-Net

In your preferred environment (i.e. Anaconda, etc.)

``` bash
git clone https://github.com/geoffroypeeters/ssmnet_ISMIR2023.git
cd ssmnet_ISMIR2023/

pip install -e .
```

### Install as needed

With `pip install`, ensure the following are installed as well:

- numpy
- librosa
- scipy
- matplotlib
- demucs
- pandas
- soundfile

## Running the Program

1. Put in the paths to the mp3 files of the two songs you wish to mash in `song_path1` and `song_path2`.
2. Pick customization options based on your preferences in `main.py`.
    - Choose to use vocals for the song segmentation by setting `VOCAL_SEGMENT` to `True`.
    - Use the average tempo of the two songs for the mashup by setting `USE_AVG_TEMPO` to `True`.
    - Pick which song goes first by setting its file path in `song_path1`.
3. Run `python main.py`

Alternatively, you can also walk through the whole process in the `main.ipynb` notebook.
