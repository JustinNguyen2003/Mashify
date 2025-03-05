#Mashify
""" 
Have to pip install the following libraries...
 - demucs
 - librosa
"""
import demucs.separate
import librosa
import numpy as np
import os
import soundfile as sf



def isolate_song_parts(song_filepath: str):
    """
    Separates the given song into vocals and instrumental parts using Demucs.

    Args:
        song_filepath (str): Path to the input song file.

    Raises:
        FileNotFoundError: If the song file does not exist.
        ValueError: If the file extension is not supported.
        RuntimeError: For any errors during the separation process.
    """
    try:
        # Check if the file exists
        if not os.path.isfile(song_filepath):
            raise FileNotFoundError(f"File not found: {song_filepath}")
        
        # Check if the file is an MP3 (or other supported format)
        valid_extensions = {".mp3", ".wav", ".flac", ".ogg"}
        _, ext = os.path.splitext(song_filepath)
        if ext.lower() not in valid_extensions:
            raise ValueError(f"Unsupported file format '{ext}'. Supported formats are: {', '.join(valid_extensions)}")
        
        # Run Demucs separation
        print(f"Processing file: {song_filepath}")
        demucs.separate.main(["--mp3", "--two-stems", "vocals", "-n", "mdx_extra", song_filepath])
        print("Separation completed successfully.")

        song_name = song_filepath.replace(".mp3", "")
        return(f"separated/mdx_extra/{song_name}/vocals.mp3", f"separated/mdx_extra/{song_name}/no_vocals.mp3")

    except FileNotFoundError as fnf_error:
        print(f"Error: {fnf_error}")
    except ValueError as val_error:
        print(f"Error: {val_error}")
    except Exception as e:
        # Catch-all for unexpected errors
        print(f"An unexpected error occurred: {e}")
        raise RuntimeError("Demucs processing failed.") from e
    return (None, None)



def get_key_and_bpm(song_file: str):
    audio_file = librosa.load(song_file)

    y, sr = audio_file

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    # Compute the Chroma Short-Time Fourier Transform (chroma_stft)
    chromagram = librosa.feature.chroma_stft(y=y, sr=sr)

    # Calculate the mean chroma feature across time
    mean_chroma = np.mean(chromagram, axis=1)

    # Define the mapping of chroma features to keys
    chroma_to_key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    # Find the key by selecting the maximum chroma feature
    estimated_key_index = np.argmax(mean_chroma)
    estimated_key = chroma_to_key[estimated_key_index]

    # Print the detected key and BPM
    print("Detected Key:", estimated_key)
    print("Detected Tempo:", tempo)

    return tempo, estimated_key


def change_bpm(input_file: str, output_file: str, target_bpm: float):
    """
    Change the BPM of a song.

    Args:
        input_file (str): Path to the input audio file (e.g., .mp3, .wav).
        output_file (str): Path to save the output audio file.
        target_bpm (float): The desired BPM for the output song.
    """
    # Load the audio file
    y, sr = librosa.load(input_file, sr=None)
    
    # Analyze the current BPM
    current_bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
    print(f"Current BPM: {current_bpm}")
    
    # Compute the time-stretch factor
    bpm_ratio = target_bpm / current_bpm
    print(f"Stretching audio by a factor of {bpm_ratio:.2f} to match target BPM {target_bpm}.")
    
    # Adjust the tempo
    y_stretched = librosa.effects.time_stretch(y, bpm_ratio)
    
    # Save the output file
    sf.write(output_file, y_stretched, sr)
    print(f"Output saved to {output_file}.")



def main():
    song_file = "Really Dont Care.mp3"

    vocals_filepath, instrumental_filepath = isolate_song_parts(song_file)

    tempo, key = get_key_and_bpm(vocals_filepath)

    new_tempo = int(input("Enter new tempo (BPM) for song: "))
    change_bpm(vocals_filepath, "testoutput.mp3", new_tempo)



if __name__ == "__main__":
    main()

    