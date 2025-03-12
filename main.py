import song_analysis as sa

def main():
    song_path = "songs/Ladies_and_gentlemen_we_are_floating_in_space.mp3" # your file name here

    print("Extracting estimated BPM...")
    song_bpm = sa.get_bpm(song_path)
    print("The song's tempo in BPM is:", song_bpm)

    print("Extracting chroma features...")
    chroma_vector = sa.extract_chroma_features(song_path)

    print("Estimating key...")
    estimator = sa.KeyEstimator()
    top_keys = estimator.estimate_key(chroma_vector, top_n=3)

    print("Predicted Keys:")
    for i, (key, score) in enumerate(top_keys):
        print(f"{i+1}. {key} (Confidence: {score:.2f})")

    # Plot chroma feature distribution
    sa.plot_chroma(chroma_vector)

    print("Separating the tracks of the song...")
    separated_track_paths = sa.isolate_song_parts(song_path)
    print("The separated tracks are located at: ", separated_track_paths)

if __name__ == "__main__":
    main()