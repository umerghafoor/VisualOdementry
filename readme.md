## 1. `main.py`

1. **Imports Required Libraries**
	- Uses **NumPy** for maths calculations.        
    - Uses **OpenCV** for displaying images and drawing results.
    - Uses **MonoVideoOdometery** to process images and track motion.
2. **Takes Input Paths for Images & Pose Data**
    - Allows the user to specify **custom paths** for images and pose data, or it uses default values (`./images` and `./pose`).
3. **Sets Camera Parameters**
    - Defines camera settings like **focal length** and **principal point** (centre of the camera lens).
    - Initialises **rotation** and **translation** matrices to track movement.
4. **Inside the Main Loop**
    - The program **processes each frame** from the image sequence, continuing until no more frames are available.
    - **Display the current image frame** to the user.
    - **Listens for key presses**:
        - `Esc` → Stops the loop and ends the program.
        - `y` → Toggles the flow lines on/off for visualising the movement.
5. **Track Movement Using Optical Flow**
    - **Optical Flow** ( [**Lucas-Kanade**](https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html)) is used to track how specific points move between frames.
    - If **'y' is pressed**, it turns the flow lines on/off to visualise the movement of tracked points.
6. **Process Frame for Odometry**
    - Calls **`vo.process_frame()`** to update the odometry calculation, which tracks the camera's position based on the movement detected in the current frame.
7. **Estimate Position and Compare**
    - **Estimated position** (`mono_coord`) is obtained from odometry.
    - **Actual position** (`true_coord`) is compared to the estimated position.
    - **Error calculation**: Finds the difference between the estimated and true position using **MSE (Mean Squared Error)**.
8. **Draw the Motion Path**
    - The program **draws the position** as dots on a trajectory image:
        - **Red dot** → Actual position.
        - **Green dot** → Estimated position.
    - **Text labels** are added to explain the colours: "Red" for true position and "Green" for estimated position.
9. **Show and Save the Results**
    - Displays the trajectory image with the red and green dots representing the actual and estimated positions.
    - **Saves the trajectory image** (`trajectory.png`) after processing all frames.
### Main Loop Summary (Key Steps)

1. **Displays frames** from the image sequence.
2. **Processes each frame** to estimate the position using odometry.
3. **Compares estimated vs. true positions** and calculates error.
4. **Draws a trajectory** with red (actual) and green (estimated) dots.
5. **Saves the trajectory** as an image and exits when the frames end or the user presses `Esc`.
Flow diagram of the main loop can be seen [[01_experimant.excalidraw| here]]
## 2. `monovideoodometery.py`

The `MonoVideoOdometery` class performs monocular visual odometry by processing a sequence of images and corresponding true poses. It tracks key points across frames and estimates the camera's motion based on optical flow.
### **Constructor (`__init__`):**
- **Arguments:**
  - `img_file_path`: Path to the image sequence.
  - `pose_file_path`: Path to a text file with true pose data.
- **Keyword Arguments:**
  - `focal_length`: Camera focal length (default: 718.8560).
  - `pp`: Principal point (default: (607.1928, 185.2157)).
  - `lk_params`: Parameters for  [**Lucas-Kanade**](https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html) optical flow.
  - `detector`: Feature detector (default: FastFeatureDetector).
- **File Validations:**
  - Checks if all files in `img_file_path` are PNG images.
  - Validates that the pose file exists and can be opened.
---
### **Key Methods:**

#### 1. `hasNextFrame()`
- **Purpose**: Checks if there are more frames to process.
- **Returns**: `True` if there are more frames, otherwise `False`.
#### 2. `detect(img)`
- **Purpose**: Detects key points in the provided image.
- **Arguments**: `img` - Input image to detect features.
- The FAST algorithm uses intensity comparisons between pixels surrounding a central pixel to determine corners. The algorithm checks if for each pixel, a circle of radius **r=16** pixels is drawn around it. The intensity of each pixel in the circle is compared to the intensity of the centre pixel. where δ is a predefined intensity threshold. If enough pixels on the circle meet this condition, the point is considered a corner.
	![[Pasted image 20250402021735.png]]
- Details https://www.edwardrosten.com/work/fast.html
- **Returns**: An array of detected feature coordinates in (x, y) format.
#### 3. `visual_odometery()`
- **Purpose**: Performs visual odometry by computing optical flow between consecutive frames.
  - If fewer than 2000 features remain, it triggers a new feature detection.
  - Uses [**Lucas-Kanade**](https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html) optical flow to track key points from the old frame to the current frame.
  - Estimates the camera's relative motion (rotation and translation) by computing the essential matrix and recovering pose.
  - Updates the camera's rotation (`R`) and translation (`t`) based on the optical flow.
  - Ensures proper initialisation during the first few frames, then updates the motion estimates for subsequent frames.
#### 4. `get_mono_coordinates()`
- **Purpose**: Returns the adjusted monocular coordinates after applying a transformation matrix to the translation vector.
$$
t_{new} = T.t
$$
 - **Returns**: Flattened translation vector (x, y, z).
#### 5. `get_true_coordinates()`
- **Purpose**: Returns the true coordinates from the pose file.
- **Returns**: Flattened array of true coordinates (x, y, z).

#### 6. `get_absolute_scale()`
- **Purpose**: Estimates the scale factor between consecutive frames based on true pose data.
- The absolute scale is computed using the true coordinates of the previous and current frames. where $t_{current​}$ and $t_{previous}$​ are the true translation vectors for the current and previous frames.
$$
scale = |t_{currnet}-t_{previous}|
$$
- **Returns**: Scalar value representing the distance between the current and previous frame's true positions.
#### 7. `process_frame()`
- **Purpose**: Processes a frame from the image sequence, performs visual odometry, and updates the state (`old_frame`, `current_frame`, `id`).
  - For the first two frames, initialises the visual odometry process.
  - For subsequent frames, updates the old and current frames and processes them.

---

### Usage Flow

1. **Initialisation** 
   - The class is initialised with image and pose file paths, and other optional parameters like focal length and detector.
1. **Frame Processing**:
   - The class iterates through each frame in the image sequence using the `process_frame()` method.
   - For each frame:
     - It detects features and tracks their movement using optical flow.
     - It computes the essential matrix and recovers the relative motion between frames.
     - It updates the camera’s translation and rotation based on the optical flow.
1. **Coordinate Retrieval**:
   - The class provides methods to retrieve monocular and true coordinates, allowing for comparison of estimated and true poses.

4. **Scale Estimation**:
   - The scale of the motion is estimated based on the true poses, which can be used to adjust the monocular odometry for more accurate results.