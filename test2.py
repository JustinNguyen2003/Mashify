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

    # Krumhansl-Schmuckler Key Profiles
    major_template = np.asarray(
        [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    )
    minor_template = np.asarray(
        [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    )

    def __post_init__(self):
        # Normalize and create circulant matrices for template matching
        self.major_template = zscore(self.major_template)
        self.major_norm = scipy.linalg.norm(self.major_template)
        self.major_template = scipy.linalg.circulant(self.major_template)

        self.minor_template = zscore(self.minor_template)
        self.minor_norm = scipy.linalg.norm(self.minor_template)
        self.minor_template = scipy.linalg.circulant(self.minor_template)

    def estimate_key(self, chroma_vector: np.array, top_n: int = 3) -> List[Tuple[str, float]]:
        """
        Estimates the most likely keys from the given chroma vector.
        Returns the top `top_n` best-matching keys with confidence scores.
        """
        chroma_vector = zscore(chroma_vector)
        chroma_norm = scipy.linalg.norm(chroma_vector)

        # Compute similarity scores
        major_scores = self.major_template.T.dot(chroma_vector) / self.major_norm / chroma_norm
        minor_scores = self.minor_template.T.dot(chroma_vector) / self.minor_norm / chroma_norm

        key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        # Create a sorted list of key matches with scores
        all_keys = []
        for i in range(12):
            all_keys.append((f"{key_names[i]} Major", major_scores[i]))
            all_keys.append((f"{key_names[i]} Minor", minor_scores[i]))

        # Sort by confidence score (highest first)
        all_keys = sorted(all_keys, key=lambda x: x[1], reverse=True)

        # Return top `top_n` possible keys
        return all_keys[:top_n]


def extract_chroma_features(song_path: str) -> np.array:
    """
    Extracts chroma features from an audio file.
    Uses Harmonic-Percussive Source Separation (HPSS) for better accuracy.
    """
    y, sr = librosa.load(song_path, sr=None)  # Load audio file

    # Apply Harmonic-Percussive Source Separation (HPSS)
    harmonic, _ = librosa.effects.hpss(y)

    # Extract chroma features using Constant-Q Transform (CQT)
    chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sr)

    # Compute the mean chroma vector over the entire song
    avg_chroma = np.mean(chroma, axis=1)

    return avg_chroma


def plot_chroma(chroma_vector: np.array):
    """
    Plots the chroma feature vector for visualization.
    """
    key_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    plt.figure(figsize=(10, 4))
    plt.bar(key_labels, chroma_vector, color='blue')
    plt.xlabel("Pitch Class")
    plt.ylabel("Intensity")
    plt.title("Chroma Feature Distribution")
    plt.show()


# ====== Example Usage ======
if __name__ == "__main__":
    song_path = "cheatingonyou.mp3"  # Replace with your actual file

    print("Extracting chroma features...")
    chroma_vector = extract_chroma_features(song_path)

    print("Estimating key...")
    estimator = KeyEstimator()
    top_keys = estimator.estimate_key(chroma_vector, top_n=3)

    print("Predicted Keys:")
    for i, (key, score) in enumerate(top_keys):
        print(f"{i+1}. {key} (Confidence: {score:.2f})")

    # Plot chroma feature distribution
    plot_chroma(chroma_vector)
