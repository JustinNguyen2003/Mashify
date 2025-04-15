import numpy as np
import librosa

from typing import List, Tuple
import os
import IPython.display as ipd
import soundfile as sf
import math

import song_analysis as sa
import utils

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

def crossfade_segments(seg1: np.ndarray, seg2: np.ndarray, sr: np.number, fade_duration=0.5) -> np.ndarray:
    """
    Crossfades two audio segments by a certain given fade duration (seconds)

    Args:
        seg1 (np.ndrray): Segment 1 in samples.
        seg2 (np.ndarray): Segment 2 in samples.
        sr (np.number): Audio's sample rate.
        fade_duration (float): length of crossfade. Defaults to 0.5 if not specified.

    Returns:
        np.ndarray: Combined segments with the crossfaded.
    """
    fade_len = int(sr * fade_duration)
    
    if len(seg1) < fade_len or len(seg2) < fade_len:
        raise ValueError("Segment too short for crossfade")

    fade_out = np.linspace(1, 0, fade_len)
    fade_in = np.linspace(0, 1, fade_len)

    seg1_end = seg1[-fade_len:] * fade_out
    seg2_start = seg2[:fade_len] * fade_in

    overlapped = seg1_end + seg2_start
    combined = np.hstack((seg1[:-fade_len], overlapped, seg2[fade_len:]))
    return combined

def apply_fade_out(audio: np.ndarray, sr: np.number, fade_duration=2.0):
    """
    Applies a fade out by a certain given duration (seconds)

    Args:
        audio (np.ndrray): Audio to fade in samples.
        sr (np.number): Audio's sample rate.
        fade_duration (float): Length of fade. Defaults to 2.0 if not specified.

    Returns:
        np.ndarray: Segment with fade out applied.
    """
    fade_len = int(sr * fade_duration)
    if len(audio) < fade_len:
        fade_len = len(audio)
    fade_curve = np.linspace(1.0, 0.0, fade_len)
    audio[-fade_len:] *= fade_curve
    return audio

def generate_mashup(segment_files1, segment_files2):
    """
    Algorithmically produces a mashup which follows the following song structure:
        Intro of song A, Verse of song A, Verse of song B, Chorus of song B, Chorus of song A
    Writes an output file named "mash.mp3"

    Assumes that the two songs given have corresponding segments that match with a song's structure as folllows:
        Segment 1: Intro, Segments 2-3: Verse, Segments 4-5: Chorus

    Args:
        segment1_files (str): Filepath to the audio file segments of song 1.
        segment2_files (str): Filepath to the audio file segments of song 2.
    """
    segments1 = utils.get_all_filepaths_sorted(segment_files1)
    segments2 = utils.get_all_filepaths_sorted(segment_files2)

    # Assume the following structure:
    # segment 1: intro
    # segments 2-3: verse
    # segments 4-5: chorus
    sr = None

    introA, sr = librosa.load(segments1[0], sr=None)
    verseA1, _ = librosa.load(segments1[1], sr=sr)
    verseA2, _ = librosa.load(segments1[2], sr=sr)
    chorusA1, _ = librosa.load(segments1[3], sr=sr)
    chorusA2, _ = librosa.load(segments1[4], sr=sr)

    verseB1, _ = librosa.load(segments2[1], sr=sr)
    verseB2, _ = librosa.load(segments2[2], sr=sr)
    chorusB1, _ = librosa.load(segments2[3], sr=sr)
    chorusB2, _ = librosa.load(segments2[4], sr=sr)

    # Combine with crossfades between A and B sections
    result = introA
    result = np.hstack((result, verseA1)) # np.hstack((result, verseA1, verseA2))

    result = crossfade_segments(result, verseB2, sr) # crossfade_segments(result, verseB1, sr)
    result = np.hstack((result, chorusB1)) # np.hstack((result, verseB2, chorusB1, chorusB2))

    result = crossfade_segments(result, chorusA2, sr)
    # result = np.hstack((result, chorusA2))

    # Apply fade-out to the final section
    result = apply_fade_out(result, sr, fade_duration=2.0)

    sf.write("mash.mp3", result, sr)
    print("Mashup saved as mash.mp3")