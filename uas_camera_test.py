import cv2
import time
import os

# --- Configuration ---
OUTPUT_DIR = "uas_risk_map_test"
CAPTURE_INTERVAL = 2.0  
CAMERA_INDEX = 1        

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Connecting to camera...")
    # MSMF is generally more stable for focus commands on modern Windows
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_MSMF) 

    if not cap.isOpened():
        print("Error: Could not open the camera. Trying DSHOW fallback...")
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not cap.isOpened():
            return

    # Set High Resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Initial Hardware Setup
    focus_val = 0        
    exposure_val = -6    
    
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) 
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) 
    
    # Send initial commands
    cap.set(cv2.CAP_PROP_FOCUS, focus_val)
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure_val)

    print("\n--- CONTROLS ---")
    print("W / S : FOCUS Up/Down (Infinity / Macro)")
    print("A / D : EXPOSURE Down/Up (Fix Jiggle)")
    print("Q     : Quit and Save Final Settings\n")

    image_count = 0
    last_capture_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Display info on screen
        status = f"FOCUS: {focus_val} | EXP: {exposure_val} | Files: {image_count}"
        cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("SUAS Camera Tuner", frame)

        key = cv2.waitKey(1) & 0xFF
        
        # Only sends commands on keypress to prevent USB saturation
        if key == ord('w'):
            focus_val = min(255, focus_val + 10)
            cap.set(cv2.CAP_PROP_FOCUS, focus_val)
        elif key == ord('s'):
            focus_val = max(0, focus_val - 10)
            cap.set(cv2.CAP_PROP_FOCUS, focus_val)
        elif key == ord('a'):
            exposure_val = max(-13, exposure_val - 1)
            cap.set(cv2.CAP_PROP_EXPOSURE, exposure_val)
        elif key == ord('d'):
            exposure_val = min(-1, exposure_val + 1)
            cap.set(cv2.CAP_PROP_EXPOSURE, exposure_val)
        elif key == ord('q'):
            break

        # Auto-capture logic
        current_time = time.time()
        if (current_time - last_capture_time) >= CAPTURE_INTERVAL:
            filename = os.path.join(OUTPUT_DIR, f"map_img_{image_count:04d}.jpg")
            cv2.imwrite(filename, frame)
            print(f"[SAVED] {filename} | F:{focus_val} E:{exposure_val}")
            last_capture_time = current_time
            image_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Final Flight Settings for Config: Focus={focus_val}, Exposure={exposure_val}")

if __name__ == "__main__":
    main()