import numpy as np
import cv2
import os


class MonoVideoOdometery(object):
    def __init__(self, 
                img_file_path,
                pose_file_path,
                focal_length = 718.8560,
                pp = (607.1928, 185.2157), 
                lk_params=dict(winSize  = (21,21), criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)), 
                detector=cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)):
        '''
        Arguments:
            img_file_path {str} -- File path that leads to image sequences
            pose_file_path {str} -- File path that leads to true poses from image sequence
        
        Keyword Arguments:
            focal_length {float} -- Focal length of camera used in image sequence (default: {718.8560})
            pp {tuple} -- Principal point of camera in image sequence (default: {(607.1928, 185.2157)})
            lk_params {dict} -- Parameters for Lucas Kanade optical flow (default: {dict(winSize  = (21,21), criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))})
            detector {cv2.FeatureDetector} -- Most types of OpenCV feature detectors (default: {cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)})
        
        Raises:
            ValueError -- Raised when file either file paths are not correct, or img_file_path is not configured correctly
        '''

        self.file_path = img_file_path
        self.detector = detector
        self.lk_params = lk_params
        self.focal = focal_length
        self.pp = pp
        self.R = np.zeros(shape=(3, 3))
        self.t = np.zeros(shape=(3, 3))
        self.id = 0
        self.n_features = 0
        
        # Curve detection parameters
        self.rotation_history = []  # Store rotation angles for curve detection
        self.yaw_angles = []  # Store yaw angles over time
        self.curve_threshold = 0.02  # Threshold for detecting sharp curves (radians) - very sensitive for real data
        self.curve_window = 8  # Longer window for real-world data analysis
        self.analysis_window = 15  # Longer analysis window for trend detection
        self.sharp_curves = []  # List to store detected sharp curve locations
        self.in_curve = False  # Flag to track if currently in a curve (prevents multiple prints)
        self.curve_start_frame = None  # Track when a curve started
        self.cumulative_rotation = 0  # Track total rotation over longer period
        self.current_turn_direction = None  # Track current turn direction: 'left', 'right', or None

        try:
            if not all([".png" in x for x in os.listdir(img_file_path)]):
                raise ValueError("img_file_path is not correct and does not exclusively png files")
        except Exception as e:
            print(e)
            raise ValueError("The designated img_file_path does not exist, please check the path and try again")

        try:
            with open(pose_file_path) as f:
                self.pose = f.readlines()
        except Exception as e:
            print(e)
            raise ValueError("The pose_file_path is not valid or did not lead to a txt file")

        self.process_frame()


    def hasNextFrame(self):
        '''Used to determine whether there are remaining frames
           in the folder to process
        
        Returns:
            bool -- Boolean value denoting whether there are still 
            frames in the folder to process
        '''

        return self.id < len(os.listdir(self.file_path)) 


    def detect(self, img):
        '''Used to detect features and parse into useable format

        
        Arguments:
            img {np.ndarray} -- Image for which to detect keypoints on
        
        Returns:
            np.array -- A sequence of points in (x, y) coordinate format
            denoting location of detected keypoint
        '''

        p0 = self.detector.detect(img)
        
        return np.array([x.pt for x in p0], dtype=np.float32).reshape(-1, 1, 2)


    def visual_odometery(self):
        '''
        Used to perform visual odometery. If features fall out of frame
        such that there are less than 2000 features remaining, a new feature
        detection is triggered. 
        '''

        if self.n_features < 2000:
            self.p0 = self.detect(self.old_frame)


        # Calculate optical flow between frames, st holds status
        # of points from frame to frame
        self.p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_frame, self.current_frame, self.p0, None, **self.lk_params)
        

        # Save the good points from the optical flow
        self.good_old = self.p0[st == 1]
        self.good_new = self.p1[st == 1]

        # print("Good points",self.good_new[0])

        E, _ = cv2.findEssentialMat(self.good_new, self.good_old, self.focal, self.pp, cv2.RANSAC, 0.999, 1.0, None)
        _, R, t, _ = cv2.recoverPose(E, self.good_old, self.good_new, focal=self.focal, pp=self.pp, mask=None)
        
        # Calculate rotation angle for curve detection
        self.update_rotation_tracking(R)
        
        # If the frame is one of first two, we need to initalize
        # our t and R vectors so behavior is different
        if self.id < 2:
            self.R = R
            self.t = t
        else:
            absolute_scale = self.get_absolute_scale()
            if (absolute_scale > 0.1 and abs(t[2][0]) > abs(t[0][0]) and abs(t[2][0]) > abs(t[1][0])):
                self.t = self.t + absolute_scale*self.R.dot(t)
                self.R = R.dot(self.R)

        # Check for sharp curves
        self.detect_sharp_curves()
        
        # Print debug information if enabled
        self.print_rotation_debug()

        # Save the total number of good features
        self.n_features = self.good_new.shape[0]


    def get_mono_coordinates(self):
        # We multiply by the diagonal matrix to fix our vector
        # onto same coordinate axis as true values
        diag = np.array([[-1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]])
        adj_coord = np.matmul(diag, self.t)

        return adj_coord.flatten()


    def get_true_coordinates(self):
        '''Returns true coordinates of vehicle
        
        Returns:
            np.array -- Array in format [x, y, z]
        '''
        return self.true_coord.flatten()


    def get_absolute_scale(self):
        '''Used to provide scale estimation for mutliplying
           translation vectors
        
        Returns:
            float -- Scalar value allowing for scale estimation
        '''
        pose = self.pose[self.id - 1].strip().split()
        # print(pose)
        # input("Press Enter to continue...")
        x_prev = float(pose[3])
        y_prev = float(pose[7])
        z_prev = float(pose[11])
        pose = self.pose[self.id].strip().split()
        x = float(pose[3])
        y = float(pose[7])
        z = float(pose[11])

        true_vect = np.array([[x], [y], [z]])
        self.true_coord = true_vect
        prev_vect = np.array([[x_prev], [y_prev], [z_prev]])

        # print("True coordinates: ", true_vect)
        # print("Previous coordinates: ", prev_vect)
        # print("Translation vector: ", self.t)
        # print("Rotation vector: ", self.R)
        
        
        return np.linalg.norm(true_vect - prev_vect)


    def process_frame(self):
        '''Processes images in sequence frame by frame
        '''

        if self.id < 2:
            self.old_frame = cv2.imread(self.file_path +str().zfill(6)+'.png', 0)
            self.current_frame = cv2.imread(self.file_path + str(1).zfill(6)+'.png', 0)
            self.visual_odometery()
            self.id = 2
        else:
            self.old_frame = self.current_frame
            self.current_frame = cv2.imread(self.file_path + str(self.id).zfill(6)+'.png', 0)
            self.visual_odometery()
            self.id += 1

    def get_selected_points(self):
        '''Returns the (x, y) coordinates of the points being tracked after optical flow'''
        points = self.good_new
        return points

    def rotation_matrix_to_euler_angles(self, R):
        '''Convert rotation matrix to Euler angles (roll, pitch, yaw)
        
        Arguments:
            R {np.array} -- 3x3 rotation matrix
            
        Returns:
            tuple -- (roll, pitch, yaw) angles in radians
        '''
        sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
        
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2,1], R[2,2])
            y = np.arctan2(-R[2,0], sy)
            z = np.arctan2(R[1,0], R[0,0])
        else:
            x = np.arctan2(-R[1,2], R[1,1])
            y = np.arctan2(-R[2,0], sy)
            z = 0
            
        return x, y, z

    def update_rotation_tracking(self, R_current):
        '''Update rotation tracking for curve detection
        
        Arguments:
            R_current {np.array} -- Current rotation matrix
        '''
        # Extract yaw angle (rotation around vertical axis)
        _, _, yaw = self.rotation_matrix_to_euler_angles(R_current)
        
        # Store yaw angle
        self.yaw_angles.append(yaw)
        
        # Keep only recent history for curve detection
        if len(self.yaw_angles) > self.curve_window * 2:
            self.yaw_angles = self.yaw_angles[-self.curve_window * 2:]

    def determine_turn_direction(self):
        '''Determine the direction of turn based on yaw angle changes
        
        Returns:
            str -- 'left', 'right', or 'straight' based on rotation direction
        '''
        if len(self.yaw_angles) < 3:
            return 'straight'
        
        # Use multiple frames to determine consistent direction
        window_size = min(5, len(self.yaw_angles))
        recent_yaws = self.yaw_angles[-window_size:]
        
        # Calculate differences to determine direction
        differences = []
        for i in range(1, len(recent_yaws)):
            diff = recent_yaws[i] - recent_yaws[i-1]
            
            # Handle angle wrapping (crossing from -pi to pi or vice versa)
            if diff > np.pi:
                diff -= 2 * np.pi
            elif diff < -np.pi:
                diff += 2 * np.pi
                
            differences.append(diff)
        
        # Count positive vs negative changes (with noise threshold)
        noise_threshold = 0.003  # Small threshold to ignore noise
        left_count = sum(1 for d in differences if d > noise_threshold)
        right_count = sum(1 for d in differences if d < -noise_threshold)
        
        # Determine direction based on majority
        if left_count > right_count and left_count >= 2:
            return 'left'
        elif right_count > left_count and right_count >= 2:
            return 'right'
        else:
            return 'straight'
    
    def get_turn_angle_magnitude(self):
        '''Calculate the magnitude of the current turn
        
        Returns:
            float -- Absolute turn angle magnitude over the analysis window
        '''
        if len(self.yaw_angles) < self.curve_window:
            return 0.0
            
        recent_yaws = self.yaw_angles[-self.curve_window:]
        total_rotation = recent_yaws[-1] - recent_yaws[0]
        
        # Handle angle wrapping
        if total_rotation > np.pi:
            total_rotation -= 2 * np.pi
        elif total_rotation < -np.pi:
            total_rotation += 2 * np.pi
            
        return abs(total_rotation)

    def detect_sharp_curves(self):
        '''Detect sharp curves based on multiple analysis methods for real-world data
        
        Returns:
            bool -- True if a sharp curve is detected
        '''
        if len(self.yaw_angles) < self.curve_window:
            return False
            
        # Method 1: Short-term rotation rate analysis
        recent_yaws = self.yaw_angles[-self.curve_window:]
        total_rotation = abs(recent_yaws[-1] - recent_yaws[0])
        avg_rotation_rate = total_rotation / (self.curve_window - 1)
        
        # Method 2: Longer-term cumulative rotation analysis
        longer_window = min(self.analysis_window, len(self.yaw_angles))
        if longer_window > 5:
            longer_yaws = self.yaw_angles[-longer_window:]
            cumulative_rotation = abs(longer_yaws[-1] - longer_yaws[0])
            long_term_rate = cumulative_rotation / (longer_window - 1)
            
            # Check for sustained turning over longer period
            sustained_curve = long_term_rate > (self.curve_threshold * 0.5)
        else:
            sustained_curve = False
            long_term_rate = 0
        
        # Method 3: Direction change analysis (detect consistent turning)
        direction_change = False
        if len(self.yaw_angles) >= 5:
            recent_5 = self.yaw_angles[-5:]
            # Check if there's a consistent direction of change
            differences = [recent_5[i+1] - recent_5[i] for i in range(len(recent_5)-1)]
            same_direction_count = sum(1 for d in differences if abs(d) > 0.005)  # Small threshold for noise
            direction_change = same_direction_count >= 3  # At least 3 out of 4 changes in same direction
        
        # Combined detection: either short-term sharp turn OR sustained turning OR consistent direction change
        is_sharp_curve = (avg_rotation_rate > self.curve_threshold) or sustained_curve or direction_change
        
        # Handle curve state transitions
        if is_sharp_curve and not self.in_curve:
            # Entering a new curve - print only once
            self.in_curve = True
            self.curve_start_frame = self.id
            curve_info = {
                'frame_id': self.id,
                'rotation_rate': avg_rotation_rate,
                'total_rotation': total_rotation,
                'long_term_rate': long_term_rate,
                'detection_method': self._get_detection_method(avg_rotation_rate, sustained_curve, direction_change),
                'coordinates': self.get_mono_coordinates().copy() if hasattr(self, 't') else None
            }
            self.sharp_curves.append(curve_info)
            method = curve_info['detection_method']
            print(f"🔄 Sharp curve detected at frame {self.id} ({method}): rotation rate = {avg_rotation_rate:.4f} rad/frame")
            
        elif not is_sharp_curve and self.in_curve:
            # Exiting a curve
            self.in_curve = False
            if self.curve_start_frame is not None:
                duration = self.id - self.curve_start_frame
                print(f"✅ Curve completed. Duration: {duration} frames")
            
        return is_sharp_curve
    
    def _get_detection_method(self, short_rate, sustained, direction):
        '''Helper method to identify which detection method triggered'''
        methods = []
        if short_rate > self.curve_threshold:
            methods.append("Sharp Turn")
        if sustained:
            methods.append("Sustained Curve")
        if direction:
            methods.append("Direction Change")
        return " + ".join(methods) if methods else "Unknown"

    def get_curve_statistics(self):
        '''Get statistics about detected curves
        
        Returns:
            dict -- Dictionary containing curve statistics
        '''
        if not self.sharp_curves:
            return {'total_curves': 0, 'average_rotation_rate': 0, 'max_rotation_rate': 0}
            
        rotation_rates = [curve['rotation_rate'] for curve in self.sharp_curves]
        
        return {
            'total_curves': len(self.sharp_curves),
            'average_rotation_rate': np.mean(rotation_rates),
            'max_rotation_rate': np.max(rotation_rates),
            'curve_locations': [(curve['frame_id'], curve['coordinates']) for curve in self.sharp_curves]
        }

    def set_curve_sensitivity(self, threshold=0.02, window=8, analysis_window=15):
        '''Set the sensitivity parameters for curve detection optimized for real-world data
        
        Arguments:
            threshold {float} -- Rotation threshold in radians per frame for detecting curves (default: 0.02 - very sensitive)
            window {int} -- Number of frames to consider for immediate curve detection (default: 8 - longer for stability)
            analysis_window {int} -- Number of frames for long-term trend analysis (default: 15)
        '''
        self.curve_threshold = threshold
        self.curve_window = window
        self.analysis_window = analysis_window
        # Reset curve state when changing sensitivity
        self.in_curve = False
        self.curve_start_frame = None
        print(f"Curve sensitivity updated: threshold={threshold:.3f}, window={window}, analysis_window={analysis_window}")

    def is_currently_in_curve(self):
        '''Check if the vehicle is currently in a curve
        
        Returns:
            bool -- True if currently in a curve
        '''
        return self.in_curve

    def get_current_curvature(self):
        '''Calculate current curvature based on recent rotation
        
        Returns:
            float -- Current curvature value
        '''
        if len(self.yaw_angles) < 2:
            return 0.0
            
        return abs(self.yaw_angles[-1] - self.yaw_angles[-2])

    def enable_debug_mode(self, enable=True):
        '''Enable debug mode to print rotation analysis information'''
        self.debug_mode = enable
        if enable:
            print("Debug mode enabled - will print rotation analysis every 10 frames")
    
    def print_rotation_debug(self):
        '''Print debug information about rotation analysis'''
        if not hasattr(self, 'debug_mode') or not self.debug_mode:
            return
            
        if self.id % 10 == 0 and len(self.yaw_angles) >= 5:  # Print every 10 frames
            recent_yaw = self.yaw_angles[-1] if self.yaw_angles else 0
            curvature = self.get_current_curvature()
            
            if len(self.yaw_angles) >= self.curve_window:
                recent_yaws = self.yaw_angles[-self.curve_window:]
                total_rotation = abs(recent_yaws[-1] - recent_yaws[0])
                avg_rotation_rate = total_rotation / (self.curve_window - 1)
                
                print(f"Frame {self.id}: yaw={recent_yaw:.4f}, curvature={curvature:.4f}, "
                      f"avg_rate={avg_rotation_rate:.4f}, threshold={self.curve_threshold:.4f}")
    
    def get_rotation_statistics(self):
        '''Get detailed rotation statistics for analysis'''
        if len(self.yaw_angles) < 2:
            return None
            
        yaw_array = np.array(self.yaw_angles)
        return {
            'current_yaw': self.yaw_angles[-1] if self.yaw_angles else 0,
            'yaw_range': np.max(yaw_array) - np.min(yaw_array),
            'yaw_std': np.std(yaw_array),
            'total_frames': len(self.yaw_angles),
            'current_curvature': self.get_current_curvature(),
            'average_curvature': np.mean(np.abs(np.diff(yaw_array))) if len(yaw_array) > 1 else 0
        }

