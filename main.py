import librosa
import song_analysis as sa
import mashing_utils as mu

VOCAL_SEGMENT = True

def main():
    config_file = "./config.yaml" # configuration file for the model used for song segmentation

    # file names here
    song_path1 = "songs/cherry_wine.mp3" 
    song_path2 = "songs/perfect_pair.mp3" 

    print("Extracting chroma features of the two songs...")
    chroma_vector1 = sa.extract_chroma_features(song_path1)
    chroma_vector2 = sa.extract_chroma_features(song_path2)

    print("Estimating keys...")
    estimator = sa.KeyEstimator()
    top_keys1 = estimator.estimate_key(chroma_vector1, top_n=3)
    top_keys2 = estimator.estimate_key(chroma_vector2, top_n=3)

    print("Matching keys...")
    song1_key = top_keys1[0][0].split(" ")[0]
    (song1_samples, sr1) = librosa.load(song_path1)

    song2_key = top_keys2[0][0].split(" ")[0]
    (song2_samples, sr2) = librosa.load(song_path2)

    (song1_pitched_dir, song2_pitched_dir) = mu.match_keys(song1_key, song2_key, song1_samples, sr1, song2_samples, sr2)
    print("The pitched songs with matched keys are located at: ", song1_pitched_dir, song2_pitched_dir)

    print("Matching tempos...")
    (song1_pitched_samples, sr1) = librosa.load(song1_pitched_dir)
    (song2_pitched_samples, sr2) = librosa.load(song2_pitched_dir)

    (song1_stretched_dir, song2_stretched_dir) = mu.match_bpm(song1_pitched_dir, song2_pitched_dir, song1_pitched_samples, sr1, song2_pitched_samples, sr2, use_avg=False)
    print("The stretched songs with matched bpm are located at: ", song1_stretched_dir, song2_stretched_dir)

    if(VOCAL_SEGMENT):
        # segment based on extracted vocals to not cut off the lyrics
        print("Segmenting based on vocals...")

        print("Separating the tracks of song 1...")
        separated_track1_paths = sa.isolate_song_parts(song1_stretched_dir)
        print("The separated tracks are located at: ", separated_track1_paths)

        song1_dir = separated_track1_paths[0] # vocals

        print("Separating the tracks of song 2...")
        separated_track2_paths = sa.isolate_song_parts(song2_stretched_dir)
        print("The separated tracks are located at: ", separated_track2_paths)

        song2_dir = separated_track2_paths[0]
    else:
        song1_dir = song1_stretched_dir
        song2_dir = song2_stretched_dir

    # print(song1_dir, song2_dir)

    print("Labeling the segment boundaries for song 1...")
    (segment_pdf1, segment_csv1) = sa.label_segment_boundaries(config_file, song1_dir)
    # print("The segmentation outputs for song 1 are located at: " + segment_pdf1 + ", " + segment_csv1)

    print("Writing audio files for each of song 1's segments...")
    segment_files1 = sa.write_segments(song1_stretched_dir, segment_csv1)
    print("The audio files of song 1's segments are located at: ", segment_files1)

    print("Labeling the segment boundaries for song 2...")
    (segment_pdf2, segment_csv2) = sa.label_segment_boundaries(config_file, song2_dir)
    # print("The segmentation outputs for song 2 are located at: " + segment_pdf2 + ", " + segment_csv2)

    print("Writing audio files for each of song 2's segments...")
    segment_files2 = sa.write_segments(song2_stretched_dir, segment_csv2)
    print("The audio files of song 2's segments are located at: ", segment_files2)

if __name__ == "__main__":
    main()