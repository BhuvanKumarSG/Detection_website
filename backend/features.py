import numpy as np
import librosa
from scipy.signal import hilbert


def extract_features(path, sr=22050, n_mfcc=13, feature_set="base"):
    """Load an audio file and compute a compact feature vector.

    feature_set: "base" (fast, small) or "bio" (richer, "bio-inspired" auditory features)

    Returns a 1D numpy array.
    """
    # librosa.load supports many formats via soundfile/audioread backends
    y, _ = librosa.load(path, sr=sr, mono=True)
    if y.size == 0:
        # conservative default size (base): n_mfcc*3 + 3
        if feature_set == "base":
            return np.zeros((n_mfcc * 3 + 3,), dtype=float)
        else:
            # estimate for bio set (mfcc + delta + std + contrast + chroma + tonnetz + 3 spectral)
            return np.zeros((n_mfcc * 3 + 12 + 3,), dtype=float)

    # Base features: MFCC mean/std/delta, plus centroid/rolloff/zcr
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta_mean = np.mean(mfcc_delta, axis=1)

    spec_centroid_mean = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spec_rolloff_mean = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zcr_mean = np.mean(librosa.feature.zero_crossing_rate(y))

    base_feat = np.concatenate([
        mfcc_mean,
        mfcc_std,
        mfcc_delta_mean,
        [spec_centroid_mean, spec_rolloff_mean, zcr_mean]
    ])

    if feature_set == "base":
        feat = base_feat
    else:
        # Bio-inspired feature set (improved cochleagram + modulation features)
        # Key changes vs earlier version:
        # - use more mel bands for finer spectral resolution
        # - apply mild compressive nonlinearity (log1p) before envelope extraction
        # - per-band normalization of envelopes to reduce channel imbalance
        # - additional modulation bands and modulation-centroid features
        # - include temporal delta/RMS of envelope as dynamic cue
        # Compute mel spectrogram (power)
        n_mels = 64
        hop_length = 512
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=sr // 2, hop_length=hop_length, power=2.0)

        # Convert to amplitude and apply mild compressive nonlinearity (approx cochlear compression)
        S_amp = np.sqrt(S)
        S_comp = np.log1p(S_amp)

        # For each mel band compute the temporal envelope via Hilbert transform on the compressed amplitude
        # S_comp shape: (n_mels, t_frames)
        # Ensure numeric dtype then apply hilbert per-band to satisfy static checkers and avoid axis ambiguity
        S_comp = np.asarray(S_comp, dtype=float)
        env = np.abs(np.apply_along_axis(hilbert, 1, S_comp))

        # Per-band normalization (zero mean, unit std) to reduce dominance of loud bands
        env_mean = np.mean(env, axis=1, keepdims=True)
        env_std = np.std(env, axis=1, keepdims=True) + 1e-8
        env_norm = (env - env_mean) / env_std

        # Compute modulation spectrum (FFT across time for each band)
        mod_spec = np.fft.rfft(env_norm, axis=1)
        mod_power = np.abs(mod_spec) ** 2

        # Build modulation frequency axis (frames sampling rate = sr / hop_length)
        n_frames = env.shape[1]
        frame_rate = float(sr) / float(hop_length)
        freqs = np.fft.rfftfreq(n_frames, d=1.0 / frame_rate)

        # Define modulation bands (Hz) (covering slower to faster modulations)
        mod_bands = [(0.5, 2.0), (2.0, 8.0), (8.0, 16.0), (16.0, 32.0)]
        mod_feats = []
        mod_centroids = []
        for lo, hi in mod_bands:
            idx = (freqs >= lo) & (freqs < hi)
            if not np.any(idx):
                band_energy = np.zeros((n_mels,))
                band_centroid = 0.0
            else:
                band_power = mod_power[:, idx]
                band_energy = np.mean(band_power, axis=1)
                # centroid per mel band then summarized
                band_centroid = np.sum(freqs[idx] * np.mean(band_power, axis=0)) / (np.sum(np.mean(band_power, axis=0)) + 1e-12)
            # summarize across mel bands (mean and std)
            mod_feats.append(np.mean(band_energy))
            mod_feats.append(np.std(band_energy))
            mod_centroids.append(band_centroid)

        # modulation-delta: compute simple temporal derivative energy across mel bands
        env_delta = np.diff(env_norm, axis=1)
        delta_rms = np.sqrt(np.mean(env_delta ** 2, axis=1))
        delta_mean = np.mean(delta_rms)
        delta_std = np.std(delta_rms)

        # spectral contrast, chroma and tonnetz as complements
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)
        spec_contrast_mean = np.mean(spec_contrast, axis=1)

        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        chroma_mean = np.mean(chroma, axis=1)

        y_harmonic = librosa.effects.harmonic(y)
        try:
            tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
            tonnetz_mean = np.mean(tonnetz, axis=1)
        except Exception:
            tonnetz_mean = np.zeros((6,))

        # Mel-band summary statistics (mean and std) from compressed mel spectrogram
        mel_mean = np.mean(S_comp, axis=1)
        mel_std = np.std(S_comp, axis=1)

        # concatenate into an extended feature vector
        feat = np.concatenate([
            base_feat,
            spec_contrast_mean,
            chroma_mean,
            tonnetz_mean,
            np.array(mod_feats),
            np.array(mod_centroids),
            np.array([delta_mean, delta_std]),
            mel_mean,
            mel_std
        ])

    # Ensure finite
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    return feat
