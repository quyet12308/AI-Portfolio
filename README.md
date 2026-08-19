# Backend & AI Integration Engineer Portfolio - Tran Dang Quyet

### Hello! I'm Quyet, a Software Engineer with nearly 4 years of hands-on experience bridging the gap between AI models and Production Systems.

My expertise lies in **Backend Development (Python/FastAPI)**, **System Troubleshooting**, and **AI Integration**. I specialize in maintaining legacy systems, diagnosing complex server-side bottlenecks, and deploying AI solutions (CV, NLP) into real-world applications using Docker and Linux environments.

Below is a collection of my real-world production incident reports and key AI integration projects.

---

## PART 1: PRODUCTION SYSTEM TROUBLESHOOTING & DEVOPS
*(Real-world case studies of diagnosing and fixing critical system issues)*

### 1. Zero-Downtime Surgery on Thumbor CDN
*   **The Problem:** The legacy CRM system experienced severe image loading delays (10-20s timeouts) due to a single-process Python 2 bottleneck and a double-call bug in PHP.
*   **The Solution:** Orchestrated a zero-downtime migration. Configured Nginx proxy-cache and deployed a multi-process Dockerized Thumbor architecture.
*   **The Result:** Achieved a **176x speedup** for warm cache and 4.3x for cold cache, resolving the timeout issue completely without interrupting live user traffic.

### 2. The "Ghost Connections" - Debugging 900% CPU Load
*   **The Problem:** A Sandbox Worker VPS experienced a massive CPU spike (900%) with no apparent traffic increase or service crashes.
*   **The Investigation:** Utilized OS-level debugging tools (`pidstat`, `/proc` filesystem) to trace the issue. Discovered that non-PTY SSH connections combined with Go's SIGPIPE suppression caused `docker logs` processes to become zombies.
*   **The Result:** Identified and killed 14 orphaned PIDs, dropping CPU usage from 900% to 0.2% instantly without restarting any services. Established new SSH automation protocols to prevent recurrence.

### 3. Distributed Auth Failure (The Clock Skew Traitor)
*   **The Problem:** Users randomly experienced 401 Unauthorized errors immediately after logging in, caused by a silent failure in the distributed architecture.
*   **The Investigation:** Traced the issue across PHP, Nginx, and .NET Gateway. Discovered a 5-minute and 1-second clock drift between servers due to a blocked NTP port (UDP 123), which exceeded the .NET JWT `ClockSkew` tolerance by exactly 1 second.
*   **The Result:** Forced a system clock sync and restored the authentication flow. Documented the critical importance of infrastructure time-sync in distributed microservices.

---

## PART 2: AI INTEGRATION PROJECTS
*(End-to-end development from model selection to API deployment)*

### 1. End-to-End Speech Recognition (ASR) System for Call Centers
*   **The Problem:** Automating the transcription of call center recordings to improve QA efficiency.
*   **My Solution:** Built a multi-stage AI pipeline. Transitioned from Wav2Vec to Whisper, and finally to **Chunkformer**, achieving **90-95% accuracy** on challenging 8kHz audio. Integrated K-Means for Speaker Diarization.
*   **Tech Stack:** Python, FastAPI, PyTorch, Hugging Face, Docker, RabbitMQ.

### 2. Computer Vision System for Crop Disease Diagnosis
*   **The Problem:** A mobile-friendly AI app for farmers to diagnose cassava diseases from field photos.
*   **My Solution:** Designed a two-stage pipeline: YOLOv8 as a pre-filtering gate (quality control) and CNNs (ResNet/ViT) for core classification. Applied **XAI (Explainable AI)** to debug model decisions and identify data mismatches.
*   **Tech Stack:** Python, PyTorch, YOLOv8, OpenCV, FastAPI.

### 3. Multi-Document OCR Solution
*   **The Problem:** Automating data entry from unstructured documents (ID cards, business cards).
*   **My Solution:** Fine-tuned YOLOv8 for text region detection and integrated EasyOCR for text recognition. Diagnosed and resolved hardware bottlenecks (CPU vs. GPU inference) during deployment.
*   **Tech Stack:** Python, YOLOv8, EasyOCR, Docker.
