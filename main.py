import numpy as np
import cv2 as cv
from monovideoodometery import MonoVideoOdometery
import argparse

def main():
    # Argument parser for image and pose paths
    # This allows the user to specify custom paths for images and pose files
    # when running the script from the command line
    # If no paths are specified, default paths './images' and './pose' are used and a warning message is printed
    parser = argparse.ArgumentParser(description='Process paths for image and pose data.')
    parser.add_argument('--img_path', type=str, default='./images', help='Path to the image directory')
    parser.add_argument('--pose_path', type=str, default='./pose', help='Path to the pose file')
    args = parser.parse_args()

    # Warning message if the default paths are used
    if args.img_path == './images' or args.pose_path == './pose':
        print("Warning: Using default paths './images' and './pose'. Specify --img_path and --pose_path to use custom paths.")

    img_path = args.img_path
    pose_path = args.pose_path

    # Camera parameters 
    # focal length and principal point can be obtained from camera calibration
    focal = 718.8560
    pp = (607.1928, 185.2157)
    R_total = np.zeros((3, 3)) # Rotation matrix for the poseof the camera
    t_total = np.empty(shape=(3, 1)) # Translation vector for the pose of the camera

    # Parameters for lucas kanade optical flow
    # winSize: size of the window for optical flow
    # maxLevel: maximum level of the pyramid for optical flow
    # criteria: termination criteria for the optical flow algorithm
    # This will determine when the algorithm should stop iterating 
    lk_params = dict(winSize=(21, 21),
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))

    # Create some random colors
    color = np.random.randint(0, 255, (5000, 3))

    vo = MonoVideoOdometery(img_path, pose_path, focal, pp, lk_params)
    traj = np.zeros(shape=(600, 800, 3))

    flag = True

    # Loop for each frame in the video
    while vo.hasNextFrame():
        frame = vo.current_frame
        cv.imshow('frame', frame)
        k = cv.waitKey(1)
        if k == 27:  # Escape key to stop
            break

        if k == 121:  # 'y' key to toggle flow lines
            flag = not flag
            toggle_out = lambda flag: "On" if flag else "Off"
            print("Flow lines turned ", toggle_out(flag))
            mask = np.zeros_like(vo.old_frame)
            mask = np.zeros_like(vo.current_frame)

        vo.process_frame()

        good_points = vo.get_selected_points()

        # show good points on the current frame
        for i in range(len(good_points)):
            a, b = good_points[i]
            # print("Good points: ", a, b)
            frame = cv.circle(frame, (int(a), int(b)), 3, color[i].tolist(), -1)

        # show updated frame
        # cv.imshow('frame', frame)

        # delay
        # cv.waitKey(1)
        
        # print(vo.get_mono_coordinates())

        mono_coord = vo.get_mono_coordinates()
        true_coord = vo.get_true_coordinates()

        # print("MSE Error: ", np.linalg.norm(mono_coord - true_coord))
        # print("x: {}, y: {}, z: {}".format(*[str(pt) for pt in mono_coord]))
        # print("true_x: {}, true_y: {}, true_z: {}".format(*[str(pt) for pt in true_coord]))

        draw_x, draw_y, draw_z = [int(round(x)) for x in mono_coord]
        true_x, true_y, true_z = [int(round(x)) for x in true_coord]

        traj = cv.circle(traj, (true_x + 400, true_z + 100), 1, list((0, 0, 255)), 4)
        traj = cv.circle(traj, (draw_x + 400, draw_z + 100), 1, list((0, 255, 0)), 4)

        # cv.putText(traj, 'Actual Position:', (140, 90), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # cv.putText(traj, 'Red', (270, 90), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        # cv.putText(traj, 'Estimated Odometry Position:', (30, 120), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # cv.putText(traj, 'Green', (270, 120), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv.imshow('trajectory', traj)

    cv.imwrite("./images/trajectory.png", traj)

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
