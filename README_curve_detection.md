# Sharp Curve Detection in Visual Odometry

This enhanced version of the MonoVideoOdometry class includes functionality to detect sharp curves in vehicle trajectories using rotation analysis.

## Features Added

### 1. Rotation Tracking
- Tracks yaw angles (rotation around vertical axis) over time
- Maintains a history of rotation data for analysis
- Converts rotation matrices to Euler angles for easier interpretation

### 2. Curve Detection Algorithm
- Analyzes rotation rate over a sliding window of frames
- Detects sharp curves when rotation rate exceeds a threshold
- Configurable sensitivity parameters

### 3. Real-time Curve Indication
- Provides real-time feedback on current curvature
- Indicates when the vehicle is currently in a curve
- Stores detected curve locations and statistics

## New Methods

### Configuration Methods
```python
# Set curve detection sensitivity
vo.set_curve_sensitivity(threshold=0.08, window=5)
```

### Real-time Detection Methods
```python
# Check if currently in a curve
is_curve = vo.is_currently_in_curve()

# Get current curvature value
curvature = vo.get_current_curvature()
```

### Analysis Methods
```python
# Get comprehensive curve statistics
stats = vo.get_curve_statistics()
```

## Parameters

### Curve Detection Parameters
- **threshold**: Rotation threshold in radians per frame (default: 0.1)
  - Lower values = more sensitive (detects gentler curves)
  - Higher values = less sensitive (only sharp curves)
  - Recommended range: 0.05 - 0.15

- **window**: Number of frames to analyze for curve detection (default: 5)
  - Smaller window = more responsive to sudden changes
  - Larger window = smoother detection, less noise
  - Recommended range: 3 - 10

## Usage Examples

### Basic Usage
```python
from monovideoodometery import MonoVideoOdometery

# Initialize with curve detection
vo = MonoVideoOdometery(img_path, pose_path)
vo.set_curve_sensitivity(threshold=0.08, window=5)

# Process frames
while vo.hasNextFrame():
    vo.process_frame()
    
    if vo.is_currently_in_curve():
        print(f"Sharp curve detected at frame {vo.id}")
        print(f"Curvature: {vo.get_current_curvature():.4f} rad/frame")

# Get final statistics
stats = vo.get_curve_statistics()
print(f"Total curves detected: {stats['total_curves']}")
```

### Advanced Analysis
```python
# For detailed analysis, see curve_detection_example.py
python curve_detection_example.py
```

## Output Information

### Real-time Output
- Frame-by-frame curve detection notifications
- Current curvature values
- Visual indicators in trajectory display

### Final Statistics
- Total number of sharp curves detected
- Average and maximum rotation rates
- Frame locations of detected curves
- Spatial coordinates of curve locations

## Visualization

### In main.py
- **Red dots**: Actual trajectory (ground truth)
- **Green dots**: Estimated trajectory
- **Magenta circles**: Detected sharp curves (larger circles)

### In curve_detection_example.py
- Trajectory plot with curve markers
- Curvature analysis over time
- Threshold visualization

## Tuning Guidelines

### For Different Scenarios

1. **Highway Driving** (gentle curves):
   - threshold = 0.03 - 0.05
   - window = 7 - 10

2. **City Driving** (moderate curves):
   - threshold = 0.05 - 0.08
   - window = 5 - 7

3. **Racing/Aggressive Driving** (sharp curves only):
   - threshold = 0.1 - 0.15
   - window = 3 - 5

### Troubleshooting

**Too many false positives**: Increase threshold or window size
**Missing obvious curves**: Decrease threshold or window size
**Noisy detection**: Increase window size
**Delayed detection**: Decrease window size

## Technical Details

### Rotation Analysis
The system extracts yaw angles from rotation matrices and analyzes the rate of change:

```
rotation_rate = |yaw(t) - yaw(t-window)| / window
```

### Curve Classification
A curve is classified as "sharp" when:
```
rotation_rate > threshold
```

### Coordinate System
- Uses camera coordinate system (right-handed)
- Yaw rotation around Y-axis (vertical)
- Positive yaw = left turn, Negative yaw = right turn

## Integration Notes

The curve detection functionality is designed to:
- Have minimal impact on existing odometry performance
- Provide both real-time and post-processing capabilities
- Be easily configurable for different scenarios
- Maintain compatibility with existing code

## File Structure

- `monovideoodometery.py`: Enhanced main class with curve detection
- `main.py`: Updated example with curve visualization
- `curve_detection_example.py`: Comprehensive analysis example
- `README_curve_detection.md`: This documentation file
