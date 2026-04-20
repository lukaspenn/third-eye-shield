# Third Eye Shield - AI-Powered Dementia Care Solution (by Ocean 3)

> **NAISC Healthcare Challenge Track** - Privacy-preserving activity monitoring for elderly individuals with dementia using Graph Neural Networks and depth-only skeleton data.

## 👥 Team & Impact

### Our Team
- **Students**: 3 Junior College students collaborated with other students from Malaysia under another team called Teh Tarik Tech. From then, 3 of us formed another team called Ocean 3 to further develop the system and participate in NAISC
- **Background**: Passionate about using AI and computing for social good
- **Focus**: Combining technical innovation with compassionate care, especially in smart monitoring devices

### Impact on Singapore
- **Early Detection**: Helps identify dementia symptoms before crises via pattern recognition
- **Family Support**: Reduces caregiver burden through smart monitoring
- **Dignity**: Allows elderly to maintain independence longer, enabling them more
- **Healthcare**: Integrates with Singapore's healthcare ecosystem
- **Community**: Fosters dementia-friendly neighborhoods by providing piece of mind to patients' family members

### Challenges Faced
- **Technical**: Optimizing GCN for edge devices while maintaining accuracy
- **Ethical**: Balancing monitoring needs with privacy rights
- **Cultural**: Adapting to Singapore's family caregiving culture
- **Dataset**: Finding appropriate training data for dementia behaviors

### Reflections
This project taught us that AI in healthcare must prioritize humanity before business and finance. While technical challenges were significant, the most important lessons were about empathy, cultural sensitivity, and the balance between care and dignity. We learned that successful AI solutions require deep understanding of the human context they're meant to serve.

---

## 🎯 Competition Overview

This project is our submission for the **NAISC Healthcare Challenge Track** (2026), addressing the growing challenge of dementia care in Singapore's ageing population. By 2030, Singapore could have over 150,000 people living with dementia, creating urgent needs for innovative AI solutions that support dignity, safety, and family caregiving.

### Our Solution: Third Eye Shield

Third Eye Shield is a compassionate AI system that monitors daily activities of elderly individuals with dementia using depth cameras and edge AI. It detects patterns of confusion, wandering, falls, and other care needs while preserving privacy and dignity.

**Key Innovation:**
- **Depth-Only Processing**: No RGB video unless absolutely necessary for care
- **Real-Time GCN Analysis**: Graph Neural Networks for accurate activity recognition
- **5-Level Care Classification**: From normal activities to critical medical emergencies
- **Singapore-Focused**: Multilingual support, family-centric alerts, cultural sensitivity
- **Edge Deployment**: Runs on affordable Raspberry Pi + Coral TPU hardware

---



## 🏗️ System Architecture

### 3-Stage Pipeline

```
Stage 1: Depth Processing
         Intel RealSense D455 depth camera
         Skeleton extraction (25 joints)
              |
              v
Stage 2: GCN Activity Recognition
         EfficientGCN-Lite model (500K parameters)
         60-class action recognition on NTU RGB+D
              |
              v
Stage 3: Care Level Classification
         Maps actions to 5 care levels
         Confidence-based alerting
              |
              v
Stage 4: Caregiver Integration
         Family alerts, care logging, emergency response
```

### Care Levels

| Level | Name      | Examples | Response |
|-------|-----------|----------|----------|
| 0     | Normal    | Eating, reading, walking | No action |
| 1     | Monitor   | Slight confusion, inactivity | Log & monitor |
| 2     | Concern   | Wandering, agitation | Family alert |
| 3     | Support   | Distress, poor posture | Immediate support |
| 4     | Emergency | Falls, medical distress | Emergency services |

---

## 🚀 Quick Start

### For Competition Judges

```bash
# 1. Clone repository
git clone https://github.com/lukaspenn/third-eye-shield.git
cd third-eye-shield

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run demo
python demo.py
```

### For Caregivers

```bash
# Deploy on Raspberry Pi
python deployment/setup_pi.py

# Start monitoring
python deployment/care_monitor.py
```

---

## 📊 Technical Implementation

### Model Details
- **Architecture**: EfficientGCN-Lite with spatial-temporal graph convolutions
- **Dataset**: NTU RGB+D (56,880 samples, 60 action classes)
- **Accuracy**: 83% on cross-subject validation
- **Inference**: <50ms on Coral Edge TPU
- **Model Size**: <1MB (INT8 quantized)

### Hardware Requirements
- **Training**: NVIDIA RTX 3060+ GPU, 16GB RAM
- **Deployment**: Raspberry Pi 4 + Coral TPU, Intel RealSense D455
- **Power**: ~7W total system consumption

### Privacy & Ethics
- **Depth-Only**: 95% of processing uses non-identifiable depth data
- **RGB Triggered**: Visual recording only for verified care needs
- **Data Local**: No cloud upload, all processing on-device
- **Opt-In**: Family consent required for all monitoring

---

## 📁 Project Structure

```
third-eye-shield/
├── config.yaml              # Model configuration
├── requirements.txt         # Python dependencies
├── README.md               # This file (competition submission)
├── CARE_CLASSIFICATION_GUIDE.md  # 5-level care system
├── PROJECT_SUMMARY.md       # Technical deliverables
├── QUICKSTART.md           # Quick reference guide
├── models/                 # GCN model implementations
├── src/                    # Core implementation
├── scripts/                # Utility scripts
└── docs/                   # Additional documentation
```

---

## 🔧 Installation & Usage

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- Coral Edge TPU (for deployment)

### Training
```bash
# Prepare dataset
python scripts/prepare_ntu.py

# Train model
python src/train.py --config config.yaml

# Evaluate
python src/evaluate.py --checkpoint checkpoints/best.pth
```

### Deployment
```bash
# Export to Edge TPU
python scripts/export_model.py --format edgetpu

# Deploy on Pi
python deployment/deploy_pi.py
```

---

## 📈 Performance Metrics

- **Accuracy**: 91.35% on NTU RGB+D xsub
- **Inference Speed**: <50ms per frame on Coral TPU
- **Model Size**: 608KB (quantized)
- **Power Efficiency**: 7W system consumption
- **Privacy**: 95% depth-only processing

---

## 🤝 Acknowledgments

- **NTU RGB+D Dataset**: For providing comprehensive action recognition data
- **EfficientGCN Authors**: For the lightweight GCN architecture
- **Singapore AI Community**: For inspiration and guidance
- **Competition Organizers**: For creating this meaningful challenge

---

## 📄 License

MIT License - See LICENSE file for details

---

*This project is our submission for the 2026 Singapore Dementia Care AI Competition. We hope it contributes to a more caring and dementia-friendly Singapore.*

## Hardware

- Raspberry Pi 4 (8GB RAM)
- Intel RealSense D455 depth camera
- Google Coral Edge TPU

## Setup

```bash
# 1. Clone and enter project
cd DepthAutoencoder-AnomalyDetection

# 2. Create virtual environment
python3.11 -m venv venv311
source venv311/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up emotion model
python3 scripts/setup_emotion.py
```

## Usage

### Basic Wellness Monitoring (depth + skeleton + action)

```bash
python3 scripts/wellness_monitor.py
```

### With Emotion Detection (opt-in)

```bash
# 1. Register user (few-shot, ~2 min per user)
python3 scripts/calibrate_emotion.py --user-id UNCLE_TAN

# 2. Run with personalised emotion
python3 scripts/wellness_monitor.py --enable-emotion --emotion-profile UNCLE_TAN
```

### With LLM Companion (on laptop)

```bash
# On laptop:
python3 src/llm_companion.py --model aisingapore/llama-3-8b-cpt-sea-lionv3-instruct --port 5000

# On RPi:
python3 scripts/wellness_monitor.py --llm-endpoint http://<laptop-ip>:5000/chat
```

### With Telegram Alerts

```bash
python3 scripts/wellness_monitor.py --telegram-token "BOT_TOKEN" --telegram-chat-ids "CHAT_ID_1"
```

### Daily Summary Dashboard

```bash
python3 scripts/wellness_dashboard.py
python3 scripts/wellness_dashboard.py --date 2026-03-17 --render summary.png
```

### SSH Keyboard Controls

| Key | Action |
|-----|--------|
| R   | Start/stop video recording |
| S   | Save screenshot |
| E   | Toggle emotion on/off |
| L   | Lock/unlock person tracking |
| Q   | Quit |

---

## Project Structure

```
scripts/
  wellness_monitor.py            # Main wellness monitoring loop
  wellness_dashboard.py          # Daily wellness summary
  calibrate_emotion.py           # Per-user few-shot emotion registration
  setup_emotion.py               # Download + convert Mini-Xception to TFLite
  collect_action_data.py         # Collect training data for action classifier
  train_action_classifier.py     # Train RandomForest action model
  demo_showcase.py               # Guided demo for competition showcase
  touchscreen_ui.py              # Touchscreen UI launcher (800x480 display)
  demo_depth.py                  # Depth + skeleton demo with rule overlay
  capture_sequence.py            # Quick skeleton capture utility

models/
  emotion_classifier.py          # Emotion classifier with few-shot personalisation
  movenet_pose_extractor.py      # MoveNet skeleton extraction
  *.tflite / *.pkl               # Binary models (generated, not tracked in git)

src/
  llm_companion.py               # LLM health companion (Flask API)
  telegram_notifier.py           # Telegram family alerts
  audio_interface.py             # Voice I/O abstraction (future-ready)
  utils/
    skeleton.py                  # Shared skeleton constants and drawing utilities
    autoencoder.py               # Shared TFLite depth autoencoder wrapper
    process.py                   # Process management (camera contention)
    wellness_features.py         # Posture, sedentary, wellness level computation
    kinematics.py                # Kinematic feature extraction

config.yaml                      # Full system configuration
```

---

## Privacy Design

1. **Depth-first:** 95% of processing uses non-identifiable depth maps.
2. **Skeleton only:** RGB is used transiently for pose extraction and never stored or transferred.
3. **Emotion opt-in:** Off by default. Face crops exist in memory only (48x48 greyscale), never saved to disk.
4. **Data minimisation:** Only derived metrics logged (scores, labels, timestamps). No raw sensor data.
5. **Edge processing:** All core monitoring runs on-device. No cloud dependency.
6. **PDPA-aligned:** Informed consent, purpose limitation, right to delete.

---

## Emotion Few-Shot System

Third Eye Shield uses an open-source Mini-Xception model (~60K parameters, MIT license) pre-trained on FER2013, with per-user prototypical few-shot learning:

1. **Setup:** `setup_emotion.py` downloads the model and converts to two TFLite models (classifier + 128-dim feature extractor).
2. **Registration:** `calibrate_emotion.py` captures face samples per emotion, extracts 128-dim embeddings, stores as user profiles (.npz).
3. **Inference:** Classify by cosine similarity to user prototypes (60% few-shot + 40% base model blending).

---

## License

MIT License
