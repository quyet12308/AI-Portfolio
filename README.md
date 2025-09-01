---

## 1. End-to-End Speech Recognition (ASR) System for Call Centers

### a. The Problem & Business Context

In a typical call center environment, managers and QA teams need to review call recordings to assess performance and gather customer insights. Listening to hours of audio is incredibly time-consuming and inefficient. The goal of this project was to build an automated system that transcribes these calls, providing a quick, searchable text-based summary to significantly speed up the review process.

### b. My Solution & Architecture

I designed and developed a multi-stage AI pipeline to process call recordings and enrich the transcribed text with additional metadata.

**Data Flow Diagram:**
*(Chèn ảnh biểu đồ bạn vừa tạo bằng Mermaid Chart vào đây)*

**Key Components:**
*   **Core Speech-to-Text Engine:** The system's backbone. I led the research and implementation through several iterations:
    *   **V1 (wav2vec2):** Initial version with decent results but struggled with Vietnamese dialects.
    *   **V2 (Whisper):** Improved accuracy but had high GPU resource consumption and latency.
    *   **V3 (Chunkformer):** The final, optimal solution. It delivered **90-95% accuracy** on our 8kHz audio data, while being significantly faster and more resource-efficient.
*   **Speaker Diarization Module:** After transcription, I implemented a K-Means clustering algorithm on the Chunkformer outputs to segment the text by speaker (e.g., "Agent:" vs. "Customer:"). This achieved ~60-70% accuracy.
*   **Sentiment Analysis:** The speaker-segmented text was then fed into a fine-tuned sentiment model (`5CD-AI/Vietnamese-Sentiment-visobert`) to classify each utterance as positive, neutral, or negative.
*   **Infrastructure & Optimization:**
    *   Developed a nightly batch processing system to handle audio downloads and conversions (`.wav` to `.flac`), preventing bandwidth congestion during business hours.
    *   Independently set up and maintained the GPU server environment, including troubleshooting NVIDIA driver conflicts and CUDA errors.

### c. Key Challenges & Learnings

This project was a journey of iterative improvement and practical problem-solving.

*   **Overcoming Low-Quality Audio:** A major challenge was that our source audio was only sampled at **8kHz**, half the standard 16kHz. While initial models struggled, I discovered that the **Chunkformer architecture was surprisingly robust** and performed exceptionally well even on this low-quality input without needing a full retraining cycle.
*   **Resource vs. Accuracy Trade-off:** Moving from Whisper to Chunkformer was a critical decision. It taught me that the "best" model isn't always the largest one, but the one that fits the specific constraints of the business (in this case, speed and resource efficiency).
*   **The Value of a Pipeline Approach:** I learned that a single model rarely solves a complex business problem. By building a multi-stage pipeline, we could add significant value (speaker labels, sentiment) even if the secondary components were not perfect. It's better to provide an 80% complete solution than no solution at all.

### d. Tech Stack
*   **Languages & Frameworks:** Python, FastAPI
*   **Key Libraries:** Pytorch, Hugging Face Transformers, Librosa, Scikit-learn
*   **Infrastructure:** Docker, NVIDIA GPU Server (self-managed)
