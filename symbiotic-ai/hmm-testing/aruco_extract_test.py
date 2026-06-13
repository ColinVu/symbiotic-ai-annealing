import cv2
import cv2.aruco as aruco

def detect_all_8_markers(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not find image at {image_path}")
        return

    # 1. Pre-processing: Grayscale and a slight contrast boost
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. Define 5x5 Dictionary
    # Note: Using 5X5_1000 ensures we don't miss higher ID numbers
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)
    
    # 3. Tweak parameters for "Difficult" detections
    parameters = aruco.DetectorParameters()
    
    # These three lines are the "secret sauce" for blurry/small markers:
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 23
    parameters.adaptiveThreshConstant = 7
    # Helps detect markers that are tilted or at an angle
    parameters.perspectiveRemoveIgnoredMarginPerCell = 0.13

    # 4. Initialize and Run Detector
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        # Flatten and sort by X position (left to right)
        found = sorted(zip(ids.flatten(), corners), key=lambda x: x[1][0][0][0])
        
        print(f"Success! Found {len(ids)} markers:")
        for marker_id, corner in found:
            print(f" -> Marker ID: {marker_id}")

        # Draw results
        aruco.drawDetectedMarkers(frame, corners, ids)
        cv2.imshow('Detection', frame)
        cv2.waitKey(0)
    else:
        print("Still only seeing 1 (or zero). Check if the image path is correct!")

if __name__ == "__main__":
    # Change this to your filename
    detect_all_8_markers('aruco_reference.png')