# Care-Level Classification System for Dementia Monitoring

## 🎯 Overview

This guide explains how to use the **5-level care classification** system for your compassionate dementia monitoring application. Instead of recognizing 60/120 individual actions, the system classifies behavior into care-relevant levels to support elderly individuals and their families.

---

## 📊 5-Level Care Taxonomy

### Level 0: Normal Daily Activities ✅
**Examples**: brushing teeth, drinking water, eating, reading, writing, using phone/tablet, sitting, standing, walking, waving, clapping, exercising

**Interpretation**: Routine personal or social behavior; baseline "normal" activities

**Response**: No action required - privacy maintained

**RGB Trigger**: ❌ No

---

### Level 1: Mild Changes / Monitoring ⚠️
**Examples**: bending to pick object, carrying object, moving chair, cleaning floor, lying down, running, moving quickly

**Interpretation**: Slight deviations from routine (e.g., office running = potential confusion)

**Response**: Log activity - Gentle monitoring for patterns

**RGB Trigger**: ❌ No (monitoring only)

---

### Level 2: Moderate Care Needs 🟡
**Examples**: reaching into pocket/bag, sneaking, peeking, entering restricted area, hiding object, taking something from another person, suspicious handling

**Interpretation**: May indicate early confusion, wandering, or safety concerns

**Response**: Alert family caregiver - Record depth + skeleton for review

**RGB Trigger**: ✅ Yes (Low Priority) - if confidence > 77%

---

### Level 3: Significant Care Support 🔴
**Examples**: pushing, pulling, hitting with object, slapping, kicking someone, throwing object at person, fighting, aggressive pointing/yelling, shoving, dragging

**Interpretation**: Strong indicators of agitation, distress, or interpersonal issues

**Response**: Immediate family alert - High priority support needed

**RGB Trigger**: ✅✅ Yes (High Priority) - if confidence > 70%

---

### Level 4: Critical Medical Emergency 🚨
**Examples**: wielding knife/gun, stabbing, shooting, choking, strangling, collapsing/falling suddenly, fainting, chest-pain gesture, medical distress

**Interpretation**: Life-threatening or emergency situations requiring immediate attention

**Response**: IMMEDIATE alert + RGB + emergency services + family notification

**RGB Trigger**: ✅✅✅ YES (CRITICAL) - if confidence > 56%

---

## 🔧 Implementation Options

### Option A: Action → Care Level Mapping (Recommended for EfficientGCN)

Train model on full NTU RGB+D (60 or 120 classes), then map predictions to care levels at inference.

**Advantages**:
- Leverages full NTU dataset (56K+ samples)
- Better feature learning from diverse activities
- Can map multiple actions to same care level
- Easier to update care mapping without retraining

**Usage**:
```bash
# Train on NTU RGB+D 60/120 classes
python train.py --config config.yaml

# Inference with care mapping
python inference_care.py --model checkpoints/best_model.pth --care-mode action
```

**Code**:
```python
from utils.care_mapping import map_action_to_care, get_care_distribution

# Method 1: Map predicted action to care level
action_class = model_output.argmax()
care_level = map_action_to_care(action_class, num_classes=60)

# Method 2: Get care probability distribution
action_probs = softmax(model_output)  # (60,)
care_probs = get_care_distribution(action_probs, num_classes=60)  # (5,)
care_level = care_probs.argmax()
```

---

### Option B: Direct Threat Classification (Recommended for Edge Deployment)

Re-label training data with threat levels and train 5-class classifier.

**Advantages**:
- Smaller output layer (5 vs 60/120 neurons)
- Faster inference (~20% speedup)
- Smaller model size
- Handles class imbalance with weighted loss
- Directly optimized for security task

**Usage**:
```bash
# Train threat classifier
python train_threat.py --config config.yaml

# Inference
python inference_threat.py --model checkpoints/threat_model_best.pth --threat-mode direct
```

**Code**:
```python
from utils.threat_mapping import create_threat_labels

# Re-label dataset
action_labels = np.array([0, 1, 42, 49, 102])  # drink, eat, fall, punch, knife
threat_labels = create_threat_labels(action_labels, num_classes=60)
# Output: [0, 0, 4, 3, 4]  # neutral, neutral, critical, aggressive, critical
```

---

## 📁 Complete NTU 120 Action → Threat Mapping

The mapping is defined in `utils/threat_mapping.py`:

```python
NTU_120_THREAT_MAP = {
    # Level 0 - Non-Threat (60+ actions)
    0: 0,   # drink water
    1: 0,   # eat meal
    2: 0,   # brushing teeth
    # ... (see file for complete mapping)
    
    # Level 1 - Low-Threat (20+ actions)
    4: 1,   # drop
    5: 1,   # pickup
    # ...
    
    # Level 2 - Suspicious (5 actions)
    56: 2,  # touch other person's pocket
    103: 2, # grab other person's stuff
    # ...
    
    # Level 3 - Aggressive (10 actions)
    49: 3,  # punching/slapping
    50: 3,  # kicking other person
    # ...
    
    # Level 4 - Critical (8 actions)
    42: 4,  # falling
    102: 4, # wield knife
    104: 4, # shoot with gun
    # ...
}
```

View complete mapping:
```bash
python utils/threat_mapping.py
```

---

## 🚀 Quick Start

### 1. Test Threat Mapping

```bash
# View threat mapping for NTU 60
python -c "from utils.threat_mapping import print_threat_mapping_summary; print_threat_mapping_summary(60)"

# View threat mapping for NTU 120
python -c "from utils.threat_mapping import print_threat_mapping_summary; print_threat_mapping_summary(120)"

# Export mapping to JSON
python -c "from utils.threat_mapping import export_threat_mapping_json; export_threat_mapping_json()"
```

### 2. Train Action Model (Option A)

```bash
# Standard training on NTU RGB+D 60
python train.py --config config.yaml

# Evaluate with threat metrics
python evaluate.py --checkpoint checkpoints/best_model.pth --threat-aware
```

### 3. Train Threat Model (Option B)

```bash
# Direct threat classification training
python train_threat.py --config config.yaml

# Output: checkpoints/threat_model_best.pth (5-class model)
```

### 4. Run Inference

```bash
# Using action model with threat mapping
python inference_threat.py \
    --model checkpoints/best_model.pth \
    --input test_skeleton.npy \
    --threat-mode action \
    --confidence 0.7

# Using direct threat model
python inference_threat.py \
    --model checkpoints/threat_model_best.pth \
    --input test_skeleton.npy \
    --threat-mode direct \
    --confidence 0.7
```

---

## 🔒 Privacy-Preserving RGB Trigger Logic

The system triggers RGB recording based on threat level and confidence:

```python
def should_trigger_rgb(threat_level: int, confidence: float, threshold: float = 0.7):
    """
    RGB trigger logic for privacy preservation.
    """
    if threat_level >= 4 and confidence >= threshold * 0.8:
        # Critical: Lower threshold (56%) for emergencies
        return True, "CRITICAL"
    
    elif threat_level >= 3 and confidence >= threshold:
        # High threat: Standard threshold (70%)
        return True, "HIGH"
    
    elif threat_level >= 2 and confidence >= threshold * 1.1:
        # Suspicious: Higher threshold (77%) to reduce false positives
        return True, "MEDIUM"
    
    else:
        # Normal or low-threat: Depth-only monitoring
        return False, "NONE"
```

**Integration in your Raspberry Pi code**:

```python
# On Raspberry Pi
from inference_threat import ThreatRecognizer

recognizer = ThreatRecognizer(
    model_path='model_edgetpu.tflite',
    model_type='edgetpu',
    threat_mode=True,  # Use direct threat model
    confidence_threshold=0.7
)

# Process depth frame
skeleton = extract_skeleton_from_depth(depth_frame)  # Your RealSense processing
result = recognizer.predict(skeleton, skeleton_format='movenet')

# Security response logic
if result['trigger_rgb']:
    if result['priority'] == 'CRITICAL':
        # Emergency: Trigger RGB + alert + call emergency services
        trigger_rgb_camera(high_priority=True)
        send_alert(priority='CRITICAL', message=result['threat_name'])
        call_emergency_services()
    
    elif result['priority'] == 'HIGH':
        # Aggression: Trigger RGB + alert security
        trigger_rgb_camera(high_priority=True)
        send_alert(priority='HIGH', message=result['threat_name'])
    
    elif result['priority'] == 'MEDIUM':
        # Suspicious: Trigger RGB + log
        trigger_rgb_camera(high_priority=False)
        log_event(result)

else:
    # Normal or low-threat: Depth-only monitoring
    if result['threat_level'] == 1:
        log_anomaly(result)  # Context-dependent behavior
    # No RGB data transmitted → Privacy preserved
```

---

## 📈 Expected Performance

### Action Model (60 classes) → Threat Mapping
- **Action Accuracy**: 85-89% (on NTU 60 test set)
- **Threat Accuracy**: 88-92% (when mapped to 5 levels)
- **Inference Time**: 35-50ms (Coral TPU)
- **Model Size**: 1MB (INT8 quantized)

### Direct Threat Model (5 classes)
- **Threat Accuracy**: 86-90% (depends on class balance)
- **Inference Time**: 30-45ms (Coral TPU, ~15% faster)
- **Model Size**: 0.8MB (INT8 quantized, smaller output layer)

### Class Distribution in NTU 60
When mapped to threat levels:
- **Level 0 (Neutral)**: ~65% of samples
- **Level 1 (Low)**: ~20% of samples
- **Level 2 (Suspicious)**: ~5% of samples
- **Level 3 (Aggressive)**: ~7% of samples
- **Level 4 (Critical)**: ~3% of samples

⚠️ **Note**: Class imbalance! Use weighted loss during training:

```python
from utils.threat_mapping import get_threat_weights

class_weights = get_threat_weights(num_classes=60)
criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights))
```

---

## 🔄 Updating Threat Mappings

You can customize the threat mapping for your specific security scenarios:

```python
# Edit utils/threat_mapping.py

# Example: Reclassify "running" as suspicious in office context
NTU_120_THREAT_MAP[95] = 2  # running on the spot: Low → Suspicious

# Example: Add custom mapping
CUSTOM_THREAT_MAP = NTU_120_THREAT_MAP.copy()
CUSTOM_THREAT_MAP[23] = 2  # kicking something: Low → Suspicious (vandalism)
```

No retraining needed! Just update the mapping and re-run inference.

---

## 🧪 Testing and Validation

### Test Threat Mapping

```python
from utils.threat_mapping import map_action_to_threat, NTU_120_ACTIONS

# Test specific actions
test_actions = [
    (0, "drink water"),
    (42, "falling"),
    (49, "punching/slapping"),
    (102, "wield knife"),
    (104, "shoot with gun")
]

for action_id, action_name in test_actions:
    threat = map_action_to_threat(action_id, num_classes=120)
    print(f"Action {action_id} ({action_name}): Threat Level {threat}")

# Output:
# Action 0 (drink water): Threat Level 0
# Action 42 (falling): Threat Level 4
# Action 49 (punching/slapping): Threat Level 3
# Action 102 (wield knife): Threat Level 4
# Action 104 (shoot with gun): Threat Level 4
```

### Validate RGB Trigger Logic

```python
from utils.threat_mapping import should_trigger_rgb

# Test different scenarios
scenarios = [
    (0, 0.95, "Normal behavior, high confidence"),
    (1, 0.80, "Low threat, good confidence"),
    (2, 0.80, "Suspicious, good confidence"),
    (3, 0.75, "Aggressive, good confidence"),
    (4, 0.60, "Critical, moderate confidence"),
]

for threat, conf, desc in scenarios:
    trigger, priority = should_trigger_rgb(threat, conf, threshold=0.7)
    print(f"{desc}")
    print(f"  Threat {threat}, Confidence {conf:.0%}")
    print(f"  → RGB Trigger: {trigger}, Priority: {priority}\n")
```

---

## 📦 Deployment Checklist

### For Raspberry Pi 4 + Coral TPU:

- [ ] **Model Selection**: Choose Option A (action model) or Option B (threat model)
- [ ] **Model Training**: Train and validate model (≥85% accuracy)
- [ ] **Model Export**: Export to Edge TPU format
  ```bash
  python export_model.py --checkpoint checkpoints/best_model.pth --quantize int8 --compile-edgetpu
  ```
- [ ] **Threat Mapping**: Copy `utils/threat_mapping.py` and `threat_mapping.json` to Pi
- [ ] **Inference Script**: Deploy `inference_threat.py` to Pi
- [ ] **RGB Trigger**: Implement camera trigger logic based on threat level
- [ ] **Testing**: Validate on real RealSense D455 depth data
- [ ] **Performance**: Verify ≤50ms inference time on Coral TPU
- [ ] **Privacy**: Ensure depth-only processing until RGB trigger

---

## 🎓 Training Tips

### For Option A (Action Model):

1. **Train on full NTU dataset** for maximum action diversity
2. **Use data augmentation** to improve generalization
3. **Focus on action accuracy**, threat mapping is post-processing
4. **Target ≥85% action accuracy** on test set

### For Option B (Direct Threat Model):

1. **Use weighted loss** to handle class imbalance
   ```python
   class_weights = get_threat_weights(num_classes=60)
   criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights))
   ```

2. **Oversample minority classes** (threat levels 2-4)

3. **Monitor per-threat-level metrics**:
   ```bash
   python train_threat.py --config config.yaml
   # Output includes precision/recall/F1 for each threat level
   ```

4. **Target ≥80% for critical threats** (Level 3-4), accuracy most important for high-threat scenarios

---

## 🆚 Option A vs Option B Comparison

| Feature | Option A (Action → Threat) | Option B (Direct Threat) |
|---------|---------------------------|-------------------------|
| **Training Data** | 60/120 action classes | 5 threat classes |
| **Model Size** | 1.0 MB (INT8) | 0.8 MB (INT8) |
| **Inference Speed** | 35-50 ms | 30-45 ms |
| **Action Accuracy** | 85-89% | N/A |
| **Threat Accuracy** | 88-92% | 86-90% |
| **Flexibility** | Update mapping anytime | Requires retraining |
| **Interpretability** | High (see individual actions) | Medium (only threat level) |
| **Edge Deployment** | Good | Excellent |
| **Recommended For** | General purpose, research | Production deployment |

**Recommendation**: 
- Use **Option A** for development and testing with pretrained models
- Use **Option B** for final production deployment on Raspberry Pi

---

## 📚 Related Files

- `utils/threat_mapping.py` - Core threat mapping functions
- `train_threat.py` - Direct threat classification training
- `inference_threat.py` - Threat-aware inference engine
- `threat_mapping.json` - Exported mapping for deployment
- `PRETRAINED_MODELS_GUIDE.md` - Pretrained model options

---

## 🤝 Integration with Your System

Your complete privacy-preserving pipeline:

```
Intel RealSense D455 (Depth Only)
    ↓
Skeleton Extraction (MoveNet/BlazePose)
    ↓
Joint Mapping to NTU 25-joint format
    ↓
EfficientGCN-Lite Inference (Coral TPU)
    ↓
Threat Classification (0-4)
    ↓
Decision Logic:
  - Level 0-1: Depth-only monitoring (privacy preserved)
  - Level 2: RGB trigger (medium priority)
  - Level 3: RGB trigger (high priority alert)
  - Level 4: RGB trigger (CRITICAL + emergency services)
    ↓
Cloud Dashboard (encrypted, threat level + timestamp only)
RGB Recording (only if triggered, encrypted, short clips)
```

---

## ✅ Summary

You now have a complete threat-level classification system that:

✅ Maps 120 NTU actions to 5 security-relevant threat levels  
✅ Supports both action-based and direct threat classification  
✅ Implements privacy-preserving RGB trigger logic  
✅ Optimized for edge deployment (Raspberry Pi + Coral TPU)  
✅ Handles class imbalance with weighted training  
✅ Provides clear security responses for each threat level  
✅ Integrates seamlessly with your EfficientGCN-Lite model  

**Next Steps**:
1. Test threat mapping: `python utils/threat_mapping.py`
2. Choose Option A or B based on your requirements
3. Train model with your dataset
4. Deploy to Raspberry Pi and validate with real depth data
5. Fine-tune threat thresholds based on false positive/negative rates

Good luck with your privacy-preserving security system! 🔒🎯
