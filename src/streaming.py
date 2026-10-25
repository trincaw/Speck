import time
import numpy as np

class RealTimeStreamingEngine:
    """
    Simulates a streaming audio pipeline (e.g. from a live microphone or WebSocket stream).
    Processes audio chunks incrementally (e.g. 500ms chunks) with a rolling context window (e.g. 1.5s).
    """
    def __init__(self, recognizer, sample_rate=16000, window_sec=1.5, chunk_sec=0.5, threshold=0.75):
        self.recognizer = recognizer
        self.sr = sample_rate
        self.window_size = int(window_sec * sample_rate)
        self.chunk_size = int(chunk_sec * sample_rate)
        self.threshold = threshold
        self.buffer = np.zeros(self.window_size, dtype=np.float32)

    def process_chunk(self, chunk):
        """
        Feed a new audio chunk into the sliding buffer and infer current active speaker.
        Returns (predicted_speaker, score, latency_seconds, rtf).
        """
        # Ensure float32 format
        chunk = np.asarray(chunk, dtype=np.float32)
        
        # Shift rolling buffer left and append new chunk
        self.buffer = np.roll(self.buffer, -len(chunk))
        self.buffer[-len(chunk):] = chunk
        
        # Run recognition
        start = time.perf_counter()
        scores, _ = self.recognizer.match(self.buffer)
        latency = time.perf_counter() - start
        
        rtf = latency / (len(chunk) / self.sr)
        
        best_spk, best_score = "Unknown", 0.0
        if scores:
            best_spk = max(scores, key=scores.get)
            best_score = scores[best_spk]
            if best_score < self.threshold:
                best_spk = "Unknown"
                
        return best_spk, best_score, latency, rtf
