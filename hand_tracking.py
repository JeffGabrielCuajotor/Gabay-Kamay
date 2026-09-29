import cv2
import time
import mediapipe as mp


# ---------------------------------------------------
# Gabay-Kamay
# Basic webcam hand landmark tracking
# ---------------------------------------------------

MODEL_PATH = "models/hand_landmarker.task"


# MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


# Connections between MediaPipe's 21 hand landmarks
HAND_CONNECTIONS = [
    # Thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    # Index finger
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    # Middle finger
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    # Ring finger
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    # Pinky
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    # Palm
    (0, 17),
]


def draw_hand_landmarks(frame, landmarks):
    """
    Draw the 21 hand landmarks and the connections
    between them onto the OpenCV frame.
    """

    height, width, _ = frame.shape

    points = []

    # Convert normalized coordinates into pixel coordinates
    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

    # Draw connections first
    for start_index, end_index in HAND_CONNECTIONS:
        start_point = points[start_index]
        end_point = points[end_index]

        cv2.line(
            frame,
            start_point,
            end_point,
            (0, 255, 0),
            2
        )

    # Draw landmark points
    for point in points:
        cv2.circle(
            frame,
            point,
            5,
            (0, 0, 255),
            -1
        )


def main():

    # Configure MediaPipe
    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=MODEL_PATH
        ),

        running_mode=RunningMode.VIDEO,

        # FSL can use both hands
        num_hands=2,

        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Open webcam
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print("Gabay-Kamay Hand Tracking")
    print("-------------------------")
    print("Camera started.")
    print("Press Q to quit.")

    # Create MediaPipe Hand Landmarker
    with HandLandmarker.create_from_options(options) as landmarker:

        while True:

            success, frame = camera.read()

            if not success:
                print("ERROR: Could not read camera frame.")
                break

            # Mirror webcam
            frame = cv2.flip(frame, 1)

            # OpenCV uses BGR.
            # MediaPipe expects RGB.
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Convert OpenCV image to MediaPipe Image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # VIDEO mode requires a timestamp
            timestamp_ms = int(
                time.monotonic() * 1000
            )

            # Detect hands
            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            # Draw detected hands
            if result.hand_landmarks:

                for hand_index, hand_landmarks in enumerate(
                    result.hand_landmarks
                ):

                    draw_hand_landmarks(
                        frame,
                        hand_landmarks
                    )

                    # Determine Left / Right hand
                    if result.handedness:

                        handedness = (
                            result.handedness[hand_index][0]
                        )

                        hand_name = handedness.category_name
                        confidence = handedness.score

                        # Position text near wrist
                        wrist = hand_landmarks[0]

                        height, width, _ = frame.shape

                        text_x = int(wrist.x * width)
                        text_y = int(wrist.y * height) - 20

                        label = (
                            f"{hand_name} "
                            f"{confidence:.2f}"
                        )

                        cv2.putText(
                            frame,
                            label,
                            (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2
                        )

                # Show hand count
                cv2.putText(
                    frame,
                    f"Hands detected: {len(result.hand_landmarks)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

            else:

                cv2.putText(
                    frame,
                    "No hands detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

            # Display webcam
            cv2.imshow(
                "Gabay-Kamay - Hand Tracking",
                frame
            )

            # Q quits the application
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    camera.release()
    cv2.destroyAllWindows()

    print("Camera closed.")


if __name__ == "__main__":
    main()