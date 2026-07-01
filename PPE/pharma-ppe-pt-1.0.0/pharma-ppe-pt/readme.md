# PPE Detection Model
# JAN 31 2026


## Version 1.2.6


### Model Overview
- **Architecture**: YOLOv11
- **Model Size**: 25,849,024 parameters
- **FLOPs**: 78.7 GFLOPs
- **Layers**: 218 layers (fused)
- **Input Size**: 640x640
- **Format**: ONNX (FP16)
- **Opset**: 17

### Training Dataset
- **Training Images**: 
- **Validation Images**: 
- **Background Images**: 
- **Total Instances**: 

### Classes (18 total)
1. **mask** - Face mask
2. **gown** - Protective gown
3. **goggles** - Safety goggles
4. **hairnet** - Hair net
5. **cap** - Cap
6. **boots** - Safety boots
7. **gloves** - Protective gloves
8. **vest** - Safety vest
9. **helmet** - Safety helmet
10. **faceshield** - Face shield
11. **coverallsuit** - Coverall suit
12. **earmuff** - Ear muff/protection
13. **safety-harness** - Safety harness
14. **sleeve** - Protective sleeve
15. **mobile-phone** - Mobile phone (for detection in restricted areas)
16. **jacket** - Safety jacket
17. **safety-cap** - Safety cap
18. **turban** - Turban

### Model Performance (mAP Metrics)

| Class          | Images | Instances | Box(P) | R     | mAP50 | mAP50-95 |
|----------------|--------|-----------|--------|-------|-------|----------|
| all            | 8561   | 29459     | 0.825  | 0.795 | 0.839 | 0.573    |
| mask           | 919    | 1013      | 0.836  | 0.834 | 0.879 | 0.53     |
| gown           | 1451   | 1799      | 0.886  | 0.897 | 0.929 | 0.621    |
| goggles        | 506    | 544       | 0.712  | 0.738 | 0.774 | 0.396    |
| hairnet        | 1867   | 2182      | 0.941  | 0.971 | 0.981 | 0.766    |
| cap            | 163    | 184       | 0.856  | 0.853 | 0.881 | 0.636    |
| boots          | 4415   | 7841      | 0.86   | 0.896 | 0.925 | 0.705    |
| gloves         | 2107   | 3125      | 0.806  | 0.767 | 0.833 | 0.568    |
| vest           | 3005   | 3637      | 0.927  | 0.96  | 0.975 | 0.791    |
| helmet         | 4170   | 5034      | 0.945  | 0.972 | 0.982 | 0.824    |
| faceshield     | 70     | 72        | 0.906  | 0.75  | 0.849 | 0.591    |
| coverallsuit   | 1637   | 1788      | 0.868  | 0.882 | 0.909 | 0.642    |
| earmuff        | 114    | 167       | 0.811  | 0.721 | 0.797 | 0.391    |
| safety-harness | 83     | 86        | 0.713  | 0.347 | 0.52  | 0.248    |
| sleeve         | 970    | 1268      | 0.45   | 0.438 | 0.407 | 0.206    |
| mobile-phone   | 672    | 701       | 0.833  | 0.812 | 0.861 | 0.507    |
| turban         | 17     | 18        | 0.842  | 0.889 | 0.92  | 0.751    |
|-------------------------------------------------------------------------|

### Top Performing Classes
- **Helmet**: mAP50-95: 0.848 (Precision: 0.963, Recall: 0.975)
- **Vest**: mAP50-95: 0.824 (Precision: 0.946, Recall: 0.95/home/visionify/Desktop/visionify/model-zoo/pt_models/ppe-detection/dec22/ppe-detection-pt/ppe-detection.pt8)
- **Hairnet**: mAP50-95: 0.787 (Precision: 0.957, Recall: 0.978)
- **Boots**: mAP50-95: 0.719 (Precision: 0.869, Recall: 0.898)

### Export Command
```bash
python convert_onnx.py --model /home/visionify/Desktop/visionify/model-zoo/pt_models/ppe-detection/jan31/ppe-detection-pt/ppe-detection.pt --half  --device=0 --dynamic
```

### Model Configuration
- **Batch Size**: Dynamic batching enabled
- **Max Batch Size**: 1
- **Platform**: ONNX Runtime
- **Precision**: FP16 (half precision)
- **Dynamic Shapes**: Enabled
- **Simplified**: Yes

### Deployment Notes
- Model is optimized for GPU inference (FP16)
- Dynamic batching configured with 1ms max queue delay
- Suitable for real-time PPE compliance monitoring
- Output format: `[batch, 20, detections]` where 20 = 4 bbox coords + 16 class scores

### Use Cases
- Construction site safety monitoring
- Manufacturing facility compliance
- Healthcare PPE verification
- Industrial safety audits
- Restricted area access control (mobile phone detection)



## Updates
- Added Agitha data 
- Added false positives

# classes
 0: mask
 1: gown
 2: goggles
 3: hairnet
 4: cap
 5: boots
 6: gloves
 7: vest
 8: helmet
 9: faceshield
10: coverallsuit
11: earmuff
12: safety-harness
13: sleeve
14: mobile-phone
15: jacket
16: safety-cap
17: turban