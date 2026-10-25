import os
import time
import numpy as np
import scipy.signal
import onnxruntime as ort

class NeuralSpeakerRecognizer:
    """
    Candidate B: SOTA Real-Time Deep Neural Speaker Recognizer (CAM++ ONNX).
    Extracts 80-dim log-Mel filterbanks (fbank) and maps to a 192-dimensional
    speaker embedding, scored via Cosine Similarity / Fast Vector indexing.
    Optimized for near real-time streaming inference (<10ms per window on CPU).
    """
    def __init__(self, model_path="models/campplus_cn_en_common_200k.onnx", sample_rate=16000, num_threads=2):
        self.sr = sample_rate
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found at {model_path}")
            
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        
        # Enrolled speaker profile registry: speaker_id -> normalized 192D embedding
        self.database = {}

    def extract_fbank(self, audio, n_mels=80, win_length=400, hop_length=160, n_fft=512):
        """
        Compute standard 80-dim Log Mel-Filterbank energy with pre-emphasis and CMVN.
        25ms window, 10ms hop @ 16kHz.
        """
        if len(audio) < win_length:
            # Pad short audio
            audio = np.pad(audio, (0, win_length - len(audio)), mode='constant')
            
        # Pre-emphasis filter
        audio_preemph = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
        
        # STFT
        _, _, Zxx = scipy.signal.stft(
            audio_preemph,
            fs=self.sr,
            window='hamming',
            nperseg=win_length,
            noverlap=win_length - hop_length,
            nfft=n_fft,
            boundary=None
        )
        mag = np.abs(Zxx)
        
        # Mel filterbank construction
        def hz_to_mel(hz):
            return 2595.0 * np.log10(1.0 + hz / 700.0)
        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
            
        mel_points = np.linspace(hz_to_mel(20), hz_to_mel(self.sr / 2), n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        bins = np.floor((n_fft + 1) * hz_points / self.sr).astype(int)
        
        fbanks = np.zeros((n_mels, int(n_fft // 2 + 1)), dtype=np.float32)
        for m in range(1, n_mels + 1):
            f_m_minus = bins[m - 1]
            f_m = bins[m]
            f_m_plus = bins[m + 1]
            for k in range(f_m_minus, f_m):
                fbanks[m - 1, k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1] + 1e-8)
            for k in range(f_m, f_m_plus):
                fbanks[m - 1, k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m] + 1e-8)
                
        mel_energy = np.dot(fbanks, mag)
        mel_energy = np.where(mel_energy == 0, np.finfo(float).eps, mel_energy)
        log_mel = np.log(mel_energy).T # [Time, 80]
        
        # Cepstral Mean Normalization (CMVN)
        log_mel = log_mel - np.mean(log_mel, axis=0, keepdims=True)
        return log_mel.astype(np.float32)

    def extract_embedding(self, audio):
        """Extract L2-normalized 192D speaker embedding."""
        fbank = self.extract_fbank(audio)
        inp = np.expand_dims(fbank, axis=0) # [1, Time, 80]
        
        start = time.perf_counter()
        raw_emb = self.session.run(None, {self.input_name: inp})[0][0] # [192]
        elapsed = time.perf_counter() - start
        
        norm = np.linalg.norm(raw_emb)
        norm_emb = raw_emb / (norm + 1e-10)
        return norm_emb, elapsed

    def enroll(self, speaker_id, audio):
        """Enroll or update speaker profile with running average centroid."""
        emb, _ = self.extract_embedding(audio)
        if speaker_id in self.database:
            centroid = self.database[speaker_id] + emb
            self.database[speaker_id] = centroid / np.linalg.norm(centroid)
        else:
            self.database[speaker_id] = emb
        return self.database[speaker_id]

    def match(self, audio):
        """
        Query input audio against all enrolled profiles.
        Returns dict of speaker_id -> cosine similarity score (-1.0 to 1.0)
        and computation time.
        """
        start = time.perf_counter()
        emb, _ = self.extract_embedding(audio)
        scores = {}
        for spk_id, spk_emb in self.database.items():
            # Cosine similarity of unit vectors is their dot product
            scores[spk_id] = float(np.dot(emb, spk_emb))
        elapsed = time.perf_counter() - start
        return scores, elapsed

    def verify(self, audio_ref, audio_query):
        """Direct 1:1 verification score between two audio clips."""
        start = time.perf_counter()
        e_ref, _ = self.extract_embedding(audio_ref)
        e_query, _ = self.extract_embedding(audio_query)
        elapsed = time.perf_counter() - start
        score = float(np.dot(e_ref, e_query))
        return score, elapsed
