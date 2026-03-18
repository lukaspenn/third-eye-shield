# Third Eye Shield - Privacy-Preserving AI for Multimodal Remote Elderly Wellness Monitoring

> Depth-first, edge-deployed, multilingual wellness monitor for elderly individuals living alone in Singapore.

**Competition:** AI for Multimodal Remote Health and Wellness Monitoring (Problem Statement 2)

---

## Overview

Third Eye Shield is a privacy-preserving multimodal AI system that continuously monitors elderly wellness using a depth camera and edge AI, without capturing identifiable imagery. It detects falls, tracks posture and activity, monitors sedentary behaviour, provides an opt-in emotion sensing module with per-user few-shot personalisation, and offers an LLM health companion for empathetic check-ins.

### 3-Stage Depth-First Pipeline

```
Stage 1: Depth Autoencoder (INT8 TFLite, 7KB)
         95% of frames end here -- no RGB processing for empty rooms
              |
              v  (person detected)
Stage 2: MoveNet Lightning (17-joint skeleton, INT8)
         Non-identifiable pose extraction for activity/posture analysis
              |
              v  (skeleton available)
Stage 3: Action Classification + Wellness Computation
         10 actions (RandomForest), posture scoring, sedentary tracking
         + Optional: Emotion (opt-in) + LLM Companion (remote)
```

### Wellness Levels

| Level | Name      | Trigger                       | Colour   |
|-------|-----------|-------------------------------|----------|
| 0     | Active    | Exercising                    | Green    |
| 1     | Normal    | Daily activity                | Cyan     |
| 2     | Sedentary | Inactive >30 min              | Orange   |
| 3     | Concern   | Poor posture / negative emotion | Dark orange |
| 4     | Alert     | Fall detected                 | Red      |

### Key Features

- **Privacy by Design:** 95% of frames use non-identifiable depth data. Emotion detection opt-in only.
- **Edge-Deployed:** Runs on Raspberry Pi 4 + Intel RealSense D455 (~S$300 hardware). No cloud dependency.
- **Few-Shot Emotion Personalisation:** Mini-Xception (FER2013, MIT license) with per-user prototypical learning.
- **LLM Health Companion:** SEA-LION / MERaLiON for multilingual elderly care conversations (EN/ZH/MS/TA).
- **Telegram Alerts:** Automated family/caregiver notifications on falls and concerns.
- **Posture Scoring:** Real-time 0-100 score from shoulder alignment, head position, and spine angle.

---

## Hardware

- Raspberry Pi 4 (8GB RAM)
- Intel RealSense D455 depth camera
- Optional: Google Coral Edge TPU

## Setup

```bash
# 1. Clone and enter project
cd DepthAutoencoder-AnomalyDetection

# 2. Create virtual environment
python3.11 -m venv venv311
source venv311/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up emotion model (downloads Mini-Xception, converts to TFLite)
python3 scripts/setup_emotion_model.py
```

## Usage

### Basic Wellness Monitoring (depth + skeleton + action)

```bash
python3 scripts/wellness_monitor.py
```

### With Emotion Detection (opt-in)

```bash
# 1. Register user (few-shot, ~2 min per user)
python3 scripts/calibrate_emotions.py --user-id UNCLE_TAN

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
  calibrate_emotions.py          # Per-user few-shot emotion registration
  setup_emotion_model.py         # Download + convert Mini-Xception to TFLite
  collect_action_data.py         # Collect training data for action classifier
  train_action_classifier.py     # Train RandomForest action model
  science_fair_showcase.py       # Guided demo for competition showcase
  touchscreen_launcher.py        # Touchscreen UI launcher
  rule_based_demo_depth_overlay.py  # Simple depth + skeleton demo
  capture_single_sequence.py     # Quick skeleton capture utility

models/
  emotion_classifier.py          # Emotion classifier with few-shot personalisation
  movenet_pose_extractor.py      # MoveNet skeleton extraction
  *.tflite / *.pkl               # Binary models (generated, not tracked in git)

src/
  llm_companion.py               # LLM health companion (Flask API)
  telegram_notifier.py           # Telegram family alerts
  audio_interface.py             # Voice I/O abstraction (future-ready)
  utils/
    wellness_features.py         # Posture, sedentary, wellness level computation
    kinematics.py                # Kinematic feature extraction

config.yaml                      # Full system configuration
SUBMISSION.md                    # Competition deliverable (summary + privacy plan)
```

---

## Privacy Design

1. **Depth-first:** 95% of processing uses non-identifiable depth maps.
2. **Skeleton only:** RGB is used transiently for pose extraction -- never stored.
3. **Emotion opt-in:** Off by default. Face crops exist in memory only (48x48 greyscale), never saved to disk.
4. **Data minimisation:** Only derived metrics logged (scores, labels, timestamps). No raw sensor data.
5. **Edge processing:** All core monitoring runs on-device. No cloud dependency.
6. **PDPA-aligned:** Informed consent, purpose limitation, right to delete.

See [SUBMISSION.md](SUBMISSION.md) for the full data handling and privacy plan.

---

## Emotion Few-Shot System

Third Eye Shield uses an open-source Mini-Xception model (~60K parameters, MIT license) pre-trained on FER2013, with per-user prototypical few-shot learning:

1. **Setup:** `setup_emotion_model.py` downloads the model and converts to two TFLite models (classifier + 128-dim feature extractor).
2. **Registration:** `calibrate_emotions.py` captures face samples per emotion, extracts 128-dim embeddings, stores as user profiles (.npz).
3. **Inference:** Classify by cosine similarity to user prototypes (60% few-shot + 40% base model blending).

---

## Competition Submission

See [SUBMISSION.md](SUBMISSION.md) for the full competition deliverable including executive summary, presentation outline, and data handling & privacy plan.

---

## License

MIT License
