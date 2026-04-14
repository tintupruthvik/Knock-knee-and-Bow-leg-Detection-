# 🦵 Leg Alignment Analysis System

Detects **Knock Knees (genu valgum)**, **Bow Legs (genu varum)**, or **Normal** leg alignment from images using AI-powered pose estimation and a rule-based classifier.

---

## How It Works

```
Input (image)
        │
        ▼
  PoseDetector  ─── MediaPipe Pose (BlazePose DNN)
        │             Extracts 33 body landmarks
        │
        ▼
  Stance Validator  ─── Rejects: tilted, bent, partial-body frames
        │
        ▼
  AngleCalculator  ─── Vector maths on hip/knee/ankle
        │               • Knee angles (arccos formula)
        │               • Q-angle proxies (signed)
        │               • Normalised knee & ankle gap
        │
        ▼
  RuleBasedClassifier  ─── 3-signal voting system
        │
        ▼
  TemporalSmoother  ─── Rolling-window vote (stability)
        │
        ▼
  Visualiser  ─── OpenCV overlay: skeleton, angles, HUD
```

---

## Project Structure

```
leg_analysis/
├── main.py               Entry point — CLI
├── pose_detection.py     MediaPipe wrapper + stance validation
├── angle_calculation.py  Vector-math metrics (angles, gaps)
├── classification.py     Rule-based classifier + temporal smoother
├── visualisation.py      OpenCV drawing helpers
├── requirements.txt      Python dependencies
└── README.md             This file
```

---

## Setup

### 1. Python version
Python **3.10–3.13** recommended.

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Running

### Single image
```bash
python main.py --input path/to/image.jpg
```

### Save annotated output
```bash
python main.py --input photo.jpg --output annotated.jpg
```

### Hide the metrics panel
```bash
python main.py --input photo.jpg --no-metrics
```

### All options
```
--input FILE          Path to image file (required)
--complexity {0,1,2}  MediaPipe model: 0=lite, 1=full, 2=heavy (default: 1)
--no-metrics          Hide the metrics panel in HUD
--output FILE         Save annotated image
```

---

## Classification Logic — Rule-Based (3-signal voting)

| Signal | Knock Knees | Bow Legs | Normal |
|--------|------------|----------|--------|
| Knee–ankle ratio | > +0.03 | < −0.03 | in between |
| Avg Q-angle | > +2° | < −2° | −2° to +2° |
| Gap comparison | ankle >> knee | knee >> ankle | roughly equal |

All three signals vote; the majority wins. Confidence scales with agreement and is penalised if knee angles are asymmetric (>15° difference).

---

## Angle Formula

All angles use the **vector dot-product** formula:

```
angle = arccos( (A · B) / (|A| × |B|) )
```

where **A** and **B** are vectors from the vertex joint toward each neighbour.

---

## Metrics Explained

| Metric | Description |
|--------|-------------|
| Left/Right knee angle | Angle at the knee (hip–knee–ankle). 180° = straight leg |
| Left/Right Q-angle | Lateral knee displacement. Positive = valgus (inward) |
| Knee gap (norm) | Horizontal knee distance normalised by leg length |
| Ankle gap (norm) | Horizontal ankle distance normalised by leg length |
| Knee-ankle ratio | (ankle_gap − knee_gap) / leg_length. Positive = knock knees |

---

## Edge Cases Handled

| Situation | Handling |
|-----------|----------|
| No person detected | "No person detected" banner |
| Only partial body visible | "Full legs not visible — move camera back" |
| Person leaning / tilted | "Person appears to be leaning" |
| Knees heavily bent | Warning shown; result flagged as less reliable |
| Asymmetric knee angles | Warning shown; confidence penalised |
| Low joint visibility | Frame rejected (MIN_VISIBILITY = 0.55) |

---

## Performance Tips

- Use `--complexity 0` for fastest inference (lite model)
- Use `--complexity 1` (default) for best accuracy/speed balance
- Best results with a **full-body view** (head-to-toe) in good lighting
- Camera should be at roughly hip height, ~2–3 m away

---

## Limitations & Disclaimer

> **This tool is for educational and research purposes only.**
> It is NOT a medical device and should NOT be used to diagnose any condition.
> Always consult a qualified orthopaedic specialist or physiotherapist for clinical assessment.

- Works best with a **full-body view** in good lighting
- Loose/baggy clothing may obscure joint positions
- A single-camera 2-D projection cannot fully replicate a 3-D clinical assessment
- Results depend on camera angle

---

## License

MIT — free for research and educational use.
