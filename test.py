import numpy as np
import librosa
import scipy.linalg
import matplotlib.pyplot as plt
from scipy.stats import zscore
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class KeyEstimator:
    """
    Estimates the musical key of a song based on chroma feature matching.
    """

    major_template = np.asarray(
        [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    )
    minor_template = np.asarray(
        [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    )

    def __post_init__(self):
        self.major_template = zscore(self.major_template)
        self.major_norm = scipy.linalg.norm(self.major_template)
        self.major_template = scipy.linalg.circulant(self.major_template)

        self.minor_template = zscore(self.minor_template)
        self.minor_norm = scipy.linalg.norm(self.minor_template)
        self.minor_template = scipy.linalg.circulant(self.minor_template)

    def estimate_key(self, chroma_vector: np.array) -> Tuple[str, float]:
        """
        Estimates the most likely key for a given chroma vector.
        """
        chroma_vector = zscore(chroma_vector)
        chroma_norm = scipy.linalg.norm(chroma_vector)

        major_scores = self.major_template.T.dot(chroma_vector) / self.major_norm / chroma_norm
        minor_scores = self.minor_template.T.dot(chroma_vector) / self.minor_norm / chroma_norm

        key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        # Find best matching major and minor key
        best_major = key_names[np.argmax(major_scores)] + " Major"
        best_minor = key_names[np.argmax(minor_scores)] + " Minor"

        best_key = best_major if np.max(major_scores) > np.max(minor_scores) else best_minor
        confidence = max(np.max(major_scores), np.max(minor_scores))

        return best_key, confidence


def extract_chroma_segments(song_path: str, segment_length: float = 10.0) -> List[np.array]:
    """
    Extracts chroma features for multiple time segments of a song.
    """
    y, sr = librosa.load(song_path, sr=None)
    harmonic, _ = librosa.effects.hpss(y)

    # Compute total duration
    duration = librosa.get_duration(y=y, sr=sr)
    num_segments = int(duration // segment_length)

    chroma_segments = []

    for i in range(num_segments):
        start_sample = int(i * segment_length * sr)
        end_sample = int((i + 1) * segment_length * sr)

        segment = harmonic[start_sample:end_sample]

        if len(segment) > 0:
            chroma = librosa.feature.chroma_cqt(y=segment, sr=sr)
            avg_chroma = np.mean(chroma, axis=1)
            chroma_segments.append(avg_chroma)

    return chroma_segments


def analyze_song_key(song_path: str, segment_length: float = 10.0):
    """
    Performs key detection on multiple segments and finds the most common key.
    """
    print(f"Processing {song_path}...")

    chroma_segments = extract_chroma_segments(song_path, segment_length=segment_length)
    estimator = KeyEstimator()

    key_counts = {}
    segment_results = []

    for i, chroma_vector in enumerate(chroma_segments):
        key, confidence = estimator.estimate_key(chroma_vector)
        print(f"Segment {i+1}: {key} (Confidence: {confidence:.2f})")

        segment_results.append(key)

        if key in key_counts:
            key_counts[key] += 1
        else:
            key_counts[key] = 1

    # Find the most frequently detected key
    most_common_key = max(key_counts, key=key_counts.get)

    print("\n🔹 Key Detection Summary:")
    print(f"Most Common Key: {most_common_key}")
    print(f"Key Frequencies: {key_counts}")

    # Plot key distribution
    plt.figure(figsize=(8, 5))
    plt.hist(segment_results, bins=len(set(segment_results)), color="blue", alpha=0.7, rwidth=0.8)
    plt.xticks(rotation=45)
    plt.title("Key Distribution Across Segments")
    plt.xlabel("Detected Key")
    plt.ylabel("Frequency")
    plt.show()

    return most_common_key


# ====== Example Usage ======
if __name__ == "__main__":
    song_path = "justthetwoofus.mp3"  # Replace with actual file

    # Run the analysis with segmented key detection
    analyze_song_key(song_path, segment_length=10.0)
