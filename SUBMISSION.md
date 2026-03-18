# Third Eye Shield -- Competition Submission

## AI for Multimodal Remote Health and Wellness Monitoring (Problem Statement 2)

---

# 1. Executive Summary

## Privacy-Preserving AI for Multimodal Remote Elderly Wellness Monitoring

### The Problem

Singapore faces a rapidly ageing population -- by 2030, 1 in 4 residents will be aged 65 and above. Falls are the leading cause of injury-related deaths among the elderly, with over 40% of falls among seniors occurring at home. Many elderly Singaporeans live alone, meaning falls and health deterioration can go undetected for hours. Traditional solutions -- clinic visits, wearable devices, panic buttons -- suffer from poor compliance, delayed response, and require active user participation that frail seniors may forget or be unable to perform.

### Our Solution: Third Eye Shield

Third Eye Shield is a **privacy-preserving, multimodal AI wellness monitoring system** designed for elderly individuals living alone. It uses a depth camera (Intel RealSense) and edge AI (Raspberry Pi) to continuously monitor wellness indicators **without capturing identifiable imagery**.

**How it works -- 3-Stage Depth-First Pipeline:**
1. **Stage 1 (95% of frames):** A quantised depth autoencoder screens for presence -- no RGB processing needed for empty rooms.
2. **Stage 2 (5% of frames):** When a person is detected, MoveNet extracts a non-identifiable skeletal pose for activity and posture analysis.
3. **Stage 3 (on events):** Action classification identifies falls, exercises, and daily activities. An opt-in emotion module and LLM companion provide holistic care.

**Wellness Metrics Monitored:** fall detection, physical activity level, posture quality, sedentary behaviour, daily activity patterns, and (opt-in) emotional wellbeing.

### Why Multimodal AI

Conventional methods are insufficient: clinic visits are periodic and miss daily patterns; wearables require compliance and charging; panic buttons require conscious activation -- impossible during a fall or stroke. Third Eye Shield provides **continuous, passive, non-contact monitoring** using depth (activity/falls), selective RGB (pose), opt-in facial emotion sensing, audio-ready voice interaction, and an LLM health companion -- 4 complementary modalities that together capture a holistic wellness picture no single sensor can provide.

### Key Differentiators

- **Privacy by Design:** Depth-first architecture ensures 95% of processing uses non-identifiable depth data. Emotion detection is off by default and requires explicit user consent.
- **Edge-Deployable:** Runs entirely on Raspberry Pi 4 + RealSense depth camera (~S$300 hardware). No cloud dependency for core monitoring.
- **Multilingual LLM Companion:** Powered by SEA-LION/MERaLiON, supports English, Mandarin, Malay, Tamil, and Singlish for empathetic health conversations with Singapore's diverse elderly population.
- **Personalised:** Per-user few-shot emotion learning adapts to individual expression patterns, improving accuracy for elderly faces.

### Expected Impact

- **Early fall detection** reduces response time from hours to seconds for elderly living alone
- **Sedentary alerts** encourage movement, reducing complications from prolonged inactivity
- **LLM companion** reduces social isolation -- a key risk factor for elderly cognitive decline
- **Scalable:** Low hardware cost enables community-wide deployment (senior living facilities, HDB flats)

### Roadmap

Current prototype -> Pilot with 10 elderly volunteers (Q3 2026) -> Clinical validation with AIC/NUHS (Q1 2027) -> Community deployment via AAG/IMDA sensor network (Q3 2027)

---

# 2. Presentation Outline

## Slide Deck (~10-12 slides, 10-minute pitch)

### Slide 1: Title
- **Third Eye Shield** -- Privacy-Preserving AI for Multimodal Remote Elderly Wellness Monitoring

### Slide 2: The Problem
- Singapore: 1 in 4 residents will be >=65 by 2030
- Falls are #1 cause of injury death in elderly; 40% occur at home
- Many elderly live alone -- falls/deterioration go undetected for hours
- Current solutions fail: clinic visits (periodic), wearables (low compliance), panic buttons (require conscious activation)

### Slide 3: Why AI? Why Multimodal?
- No single sensor captures full wellness picture
- Depth camera -> activity recognition, fall detection, presence (non-identifiable)
- Pose estimation -> posture quality, sedentary behaviour
- Facial emotion -> mental wellness indicator (opt-in only)
- Voice + LLM -> companionship, cognitive engagement, alert escalation

### Slide 4: Third Eye Shield Architecture
- 3-Stage Depth-First Pipeline diagram
  - Stage 1: Depth autoencoder (TFLite INT8, 7KB) -- 95% of frames stop here
  - Stage 2: MoveNet Lightning (17 joints, skeletal only)
  - Stage 3: Action classification + wellness computation
  - Optional: Emotion (opt-in) + LLM companion (remote)
- Hardware: Raspberry Pi 4 + Intel RealSense D455 (~S$300)

### Slide 5: Wellness Metrics
- 5 continuous indicators: activity level, posture score, sedentary tracking, fall detection, emotional wellbeing
- 5-level wellness state: Active -> Normal -> Sedentary -> Concern -> Alert
- Demo screenshot: HUD overlay with all metrics in real-time

### Slide 6: Privacy & Ethics
- Depth-first: 95% of processing uses non-identifiable depth data
- RGB only for skeleton extraction -- face crops in memory only
- Emotion OFF by default -- explicit opt-in required
- No video recording, no cloud upload. PDPA-aligned.

### Slide 7: Emotion Sensing (Opt-In)
- Mini-Xception on FER2013 (TFLite, ~60K params)
- Per-user few-shot prototypical learning for personalisation
- Face crops processed in-memory, never stored
- Clinical value: persistent negative emotions correlate with depression risk

### Slide 8: LLM Health Companion
- SEA-LION (AI Singapore) -- English, Mandarin, Malay, Tamil, Singlish
- Contextual conversations: fall guidance, sedentary encouragement, mood check-ins
- Rule-based fallback for offline resilience
- Audio-ready: Whisper STT + pyttsx3 TTS interface planned

### Slide 9: Live Demo
- Depth overlay with person detection, skeleton, action recognition
- Posture score + sedentary timer updating in real-time
- Fall detection -> alert escalation -> Telegram notification
- Side-by-side: depth (what system sees) vs RGB (what system does NOT store)

### Slide 10: Data Strategy & Evaluation
- Custom dataset: 10 actions, 25 clips per action
- Metrics: fall sensitivity/specificity, action F1 (~89%), posture correlation, emotion accuracy
- Planned user study: N=10 elderly, 4-week pilot, SUS questionnaire

### Slide 11: Roadmap & Scalability
- Current: Working prototype on RPi4 + RealSense
- Q3 2026: Pilot with 10 elderly volunteers
- Q1 2027: Clinical validation (AIC/NUHS)
- Q3 2027: Community deployment -- senior living, HDB sensor network
- S$300/unit, zero recurring cloud cost

### Slide 12: Summary & Call to Action
- Privacy-first, edge-deployed, multilingual AI wellness monitor
- <S$300 hardware, 4 modalities, holistic wellness picture
- Ask: Pilot partnership, clinical validation, community deployment pathway

---

# 3. Data Handling & Privacy Plan

## 3.1 Privacy Architecture

| Stage | Data Type | Identifiable? | Processing |
|-------|-----------|---------------|------------|
| Stage 1 | Depth map (424x240) | **No** | Autoencoder anomaly detection |
| Stage 2 | Skeleton (17 joints) | **No** | Pose extraction via MoveNet |
| Stage 3 | Action label + kinematics | **No** | Classification + wellness |
| Opt-in | Face crop (48x48 greyscale) | **Partially** -- in-memory only | Emotion classification |

95% of frames are processed using depth data only. RGB is accessed only when a person is detected, exclusively for skeleton extraction -- not for facial recognition, recording, or storage.

### Emotion Module -- Opt-In by Design

- Disabled by default (`wellness.emotion.enabled: false`)
- Requires explicit user opt-in (config toggle, runtime `E` key, or `--enable-emotion` flag)
- Face crops exist in volatile memory only -- never written to disk
- Crops are 48x48 greyscale (below facial recognition threshold)
- Only the emotion label is logged -- never the image

## 3.2 Data Collection & Storage

| Data Type | Stored? | Location | Retention | Content |
|-----------|---------|----------|-----------|---------|
| Wellness events | Yes | `logs/wellness/` CSV | 90 days rolling | timestamp, action, posture, wellness level, emotion label |
| Emotion profile | Yes (opt-in) | `data/emotion_profiles/` .npz | Until deleted | Feature embeddings only (no images) |
| Depth frames | No | -- | -- | Real-time, discarded |
| RGB frames | No | -- | -- | Transient skeleton use, discarded |
| Face crops | No | -- | -- | In-memory only |
| Video/audio | No | -- | -- | Not recorded |

**Data minimisation:** Only derived metrics are logged (scores, labels, timestamps). No PII stored. User referenced by opaque ID only.

## 3.3 PDPA Compliance (Singapore)

- **Consent:** Setup includes consent briefing. Emotion requires separate opt-in.
- **Purpose limitation:** Data used solely for wellness monitoring. Never for marketing, profiling, or surveillance.
- **Notification:** Users informed of what is collected, how it is processed, and how to opt out.
- **Access & correction:** Users view logs via dashboard, can request full data deletion.
- **Protection:** All data on local device. No cloud transmission. File permissions restrict access.

## 3.4 Technical Security

- Edge-first: all core processing on-device, no internet required
- LLM communication over local network only (LAN, not internet-exposed)
- No external API calls, no telemetry
- Future: AES-256 encryption at rest, TLS for LLM communication, full-disk encryption

## 3.5 Ethical Considerations

- **Elderly autonomy:** Wellness tool, not surveillance. Emotion is opt-in to respect dignity.
- **Caregiver relationship:** Summarised reports, not live feeds. Supplements human care.
- **Bias:** Depth-based monitoring unaffected by skin tone. Per-user calibration mitigates emotion classifier bias.
- **Transparency:** On-device, auditable, interpretable metrics. No black-box decisions.

## 3.6 Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| False fall detection | Medium | Medium | Multi-stage confirmation, cooldown, caregiver verification |
| Missed fall | Low | High | Continuous monitoring, threshold tuning, optional panic button |
| Emotion inaccuracy | Medium | Low | Per-user few-shot learning, opt-in design, advisory only |
| Unauthorised access | Low | Medium | Edge processing, file permissions, future encryption |
| Camera failure | Low | High | Health check alerts, status reporting |
| User resistance | Medium | Medium | Depth-first demo, consent briefing, easy opt-out |

## 3.7 Incident Response

1. **Contain:** Power off device
2. **Assess:** Determine exposure scope (wellness logs only -- no images stored)
3. **Notify:** User and caregiver within 72 hours
4. **Remediate:** Reset device, regenerate credentials
5. **Report:** PDPC notification if breach threshold met

---

## Summary

Third Eye Shield's design rests on three pillars:
1. **Architectural privacy** -- Depth-first means identifiable data is rarely needed and never stored
2. **User control** -- Emotion opt-in, all monitoring can be disabled, data deletion on request
3. **Data minimisation** -- Only derived metrics logged; no images, video, or audio recordings
