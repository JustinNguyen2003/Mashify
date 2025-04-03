import numpy as np
import librosa

from typing import List, Tuple
import os
import IPython.display as ipd
import soundfile as sf
import math

import song_analysis as sa

def match_keys(song1_key: str, song2_key: str, song1_samples: np.ndarray, sr1: np.number, song2_samples: np.ndarray, sr2: np.number) -> Tuple[str, str]:
    """
    Takes the average of two song's keys and pitches each of them to match that average.
    Favours pitching up over pitching down (e.g. if the song 1 is in A, and song 2 is in A#,
    song 1 is pitched up to A#). Writes files of the pitched songs and returns a path to them.

    Args:
        song1_key (str): Key of song 1 in string format e.g. "A#".
        song2_key (str): Key of song 2 in string format e.g. "A#".
        song1_samples (np.ndarray): Song 1 in samples loaded through librosa with librosa.load().
        sr1 (np.number): Song 1's sample rate.
        song2_samples (np.ndarray): Song 2 in samples loaded through librosa with librosa.load().
        sr2 (np.number): Song 2's sample rate.

    Returns:
        Tuple[str, str]: Path to the two songs after their keys have been matched.
    """
    note_to_midi = {
            "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, 
            "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11
        }

    song1_key_val = note_to_midi[song1_key]
    song2_key_val = note_to_midi[song2_key]

    print(f"Song 1's key: {song1_key} ({song1_key_val} in MIDI)")
    print(f"Song 2's key: {song2_key} ({song2_key_val} in MIDI)")

    key_diff = song2_key_val - song1_key_val

    print(f"The difference between the keys is: {key_diff}")

    if(key_diff < 0):
        key_diff = key_diff + 12

    song1_shift = 0
    song2_shift = 0

    if(key_diff <=6):
        if((key_diff % 2) == 0):
            song1_shift = int(key_diff/2)
            song2_shift = -(song1_shift)
        else:
            song1_shift = math.ceil(key_diff/2)
            song2_shift = -(song1_shift - 1)
    else:
        key_diff = abs(key_diff - 12)
        if((key_diff % 2) == 0):
            song2_shift = int(key_diff/2)
            song1_shift = -(song2_shift)
        else:
            song2_shift = math.ceil(key_diff/2)
            song1_shift = -(song2_shift - 1)

    print("Song 1 key shift:", song1_shift)
    print("Song 2 key shift:", song2_shift)

    pitched_dir = "./songs_pitched/"

    os.makedirs(pitched_dir, exist_ok=True)

    song1_pitched_dir = pitched_dir + "song1_pitched.mp3"
    song2_pitched_dir = pitched_dir + "song2_pitched.mp3"

    song1_pitch_shifted = librosa.effects.pitch_shift(y=song1_samples, sr=sr1, n_steps=song1_shift)
    sf.write(song1_pitched_dir, song1_pitch_shifted, sr1)

    song2_pitch_shifted = librosa.effects.pitch_shift(y=song2_samples, sr=sr2, n_steps=song2_shift)
    sf.write(song2_pitched_dir, song2_pitch_shifted, sr2)

    return (song1_pitched_dir, song2_pitched_dir)

def match_bpm(song_path1: str, song_path2:str, song1_samples: np.ndarray, sr1: np.number, song2_samples: np.ndarray, sr2: np.number, use_avg=False) -> Tuple[str, str]:
    """
    Matches two songs' tempos, writes files of the stretched songs, and returns a path to them.
    Can choose between using the average BPM by setting "use_avg" to True, or matches the slower
    song to the faster song if left as False.

    Args:
        song1_path (str): Path to song 1's audio file.
        song2_path (str): Path to song 2's audio file.
        song1_samples (np.ndarray): Song 1 in samples loaded through librosa with librosa.load().
        sr1 (np.number): Song 1's sample rate.
        song2_samples (np.ndarray): Song 2 in samples loaded through librosa with librosa.load().
        sr2 (np.number): Song 2's sample rate.

    Returns:
        Tuple[str, str]: Path to the two songs after their tempos have been matched.
    """
    song1_bpm = sa.get_bpm(song_path1)
    print("Song 1's BPM:", song1_bpm)

    song2_bpm = sa.get_bpm(song_path2)
    print("Song 2's BPM:", song2_bpm)

    if use_avg:
        print("Target BPM set to be the average of the two songs.")
        target_bpm = (song1_bpm + song2_bpm)/2
    else:
        # make target be the bpm of the faster song
        print("Target BPM set to be the BPM of the faster song.")
        target_bpm = max(song1_bpm, song2_bpm)

    song1_factor = target_bpm/song1_bpm
    song2_factor = target_bpm/song2_bpm
    print(f"The target BPM is {target_bpm}. Stretching song 1 by {song1_factor}, and song 2 by {song2_factor}")

    stretched_dir = "./songs_stretched/"

    os.makedirs(stretched_dir, exist_ok=True)

    song1_stretched_dir = stretched_dir + "song1_stretched.mp3"
    song2_stretched_dir = stretched_dir + "song2_stretched.mp3"

    song1_stretched = librosa.effects.time_stretch(song1_samples, rate=song1_factor)
    sf.write(song1_stretched_dir, song1_stretched, sr1)
    song2_stretched = librosa.effects.time_stretch(song2_samples, rate=song2_factor)
    sf.write(song2_stretched_dir, song2_stretched, sr2)

    return (song1_stretched_dir, song2_stretched_dir)