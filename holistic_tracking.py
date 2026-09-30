import cv2
import time
import mediapipe as mp

from feature_extraction import extract_features, TOTAL_FEATURES


# ---------------------------------------------------------
# Gabay-Kamay
# Holistic landmark tracking
#
# Detects:
# - Left hand
# - Right hand
# - Upper body pose
# - Face
# ---------------------------------------------------------

MODEL_PATH = "models/holistic_landmarker.task"


# MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


# ---------------------------------------------------------
# Hand connections
# ---------------------------------------------------------

HAND_CONNECTIONS = [
    # Thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    # Index
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    # Middle
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    # Ring
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


# ---------------------------------------------------------
# Upper-body pose
#
# 11 = left shoulder
# 12 = right shoulder
# 13 = left elbow
# 14 = right elbow
# 15 = left wrist
# 16 = right wrist
# 23 = left hip
# 24 = right hip
# ---------------------------------------------------------

POSE_CONNECTIONS = [
    (11, 12),

    (11, 13),
    (13, 15),

    (12, 14),
    (14, 16),

    (11, 23),
    (12, 24),

    (23, 24),
]


POSE_POINTS = [
    0,
    11,
    12,
    13,
    14,
    15,
    16,
    23,
    24,
]


# ---------------------------------------------------------
# Selected face landmarks
#
# We are NOT throwing away the other face landmarks.
# MediaPipe still detects all 478.
#
# We only draw a small selection so the screen does not
# become covered by hundreds of dots.
# ---------------------------------------------------------

FACE_POINTS = [
    # Eyes
    33,
    133,
    362,
    263,

    # Eyebrows
    70,
    105,
    334,
    300,

    # Mouth
    61,
    291,
    13,
    14,
]


def landmark_to_pixel(landmark, frame):
    """
    Convert a normalized MediaPipe landmark
    into OpenCV pixel coordinates.
    """

    height, width, _ = frame.shape

    x = int(landmark.x * width)
    y = int(landmark.y * height)

    return x, y


def draw_connections(
    frame,
    landmarks,
    connections,
    line_color,
    point_color
):
    """
    Draw landmark connections.
    """

    if not landmarks:
        return

    # Draw lines
    for start, end in connections:

        if start >= len(landmarks) or end >= len(landmarks):
            continue

        start_point = landmark_to_pixel(
            landmarks[start],
            frame
        )

        end_point = landmark_to_pixel(
            landmarks[end],
            frame
        )

        cv2.line(
            frame,
            start_point,
            end_point,
            line_color,
            2
        )

    # Draw points involved in the connections
    used_points = set()

    for start, end in connections:
        used_points.add(start)
        used_points.add(end)

    for index in used_points:

        if index >= len(landmarks):
            continue

        point = landmark_to_pixel(
            landmarks[index],
            frame
        )

        cv2.circle(
            frame,
            point,
            4,
            point_color,
            -1
        )


def draw_selected_points(
    frame,
    landmarks,
    indices,
    color
):
    """
    Draw only selected landmarks.
    """

    if not landmarks:
        return

    for index in indices:

        if index >= len(landmarks):
            continue

        point = landmark_to_pixel(
            landmarks[index],
            frame
        )

        cv2.circle(
            frame,
            point,
            3,
            color,
            -1
        )


def main():

    # -----------------------------------------------------
    # Configure MediaPipe Holistic Landmarker
    # -----------------------------------------------------

    options = HolisticLandmarkerOptions(

        base_options=BaseOptions(
            model_asset_path=MODEL_PATH
        ),

        running_mode=RunningMode.VIDEO,

        min_face_detection_confidence=0.5,
        min_face_landmarks_confidence=0.5,

        min_pose_detection_confidence=0.5,
        min_pose_landmarks_confidence=0.5,

        min_hand_landmarks_confidence=0.5,

        # Useful later for FSL facial/non-manual features
        output_face_blendshapes=True,

        output_segmentation_mask=False
    )


    # -----------------------------------------------------
    # Open webcam
    # -----------------------------------------------------

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return


    print()
    print("Gabay-Kamay Holistic Tracking")
    print("-----------------------------")
    print("Camera started.")
    print("Tracking:")
    print("- Face")
    print("- Upper body")
    print("- Left hand")
    print("- Right hand")
    print()
    print("Press Q to quit.")


    last_timestamp = 0


    # -----------------------------------------------------
    # Start MediaPipe
    # -----------------------------------------------------

    with HolisticLandmarker.create_from_options(
        options
    ) as landmarker:

        while True:

            success, frame = camera.read()

            if not success:
                print("ERROR: Could not read camera.")
                break


            # -------------------------------------------------
            # IMPORTANT
            #
            # We do not flip the input before detection.
            #
            # This helps keep anatomical left/right consistent,
            # which will matter when collecting training data.
            # -------------------------------------------------


            # Convert BGR -> RGB
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )


            # Convert to MediaPipe image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )


            # MediaPipe VIDEO mode requires timestamps
            timestamp_ms = int(
                time.monotonic() * 1000
            )

            # Guarantee increasing timestamps
            if timestamp_ms <= last_timestamp:
                timestamp_ms = last_timestamp + 1

            last_timestamp = timestamp_ms


            # -------------------------------------------------
            # Detect holistic landmarks
            # -------------------------------------------------

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            features = extract_features(result)

            print(
                f"Feature count: {len(features)}",
                end="\r"
)


            # -------------------------------------------------
            # LEFT HAND
            # -------------------------------------------------

            if result.left_hand_landmarks:

                draw_connections(
                    frame,
                    result.left_hand_landmarks,
                    HAND_CONNECTIONS,
                    (0, 255, 0),
                    (0, 0, 255)
                )


            # -------------------------------------------------
            # RIGHT HAND
            # -------------------------------------------------

            if result.right_hand_landmarks:

                draw_connections(
                    frame,
                    result.right_hand_landmarks,
                    HAND_CONNECTIONS,
                    (255, 255, 0),
                    (255, 0, 255)
                )


            # -------------------------------------------------
            # POSE
            # -------------------------------------------------

            if result.pose_landmarks:

                draw_connections(
                    frame,
                    result.pose_landmarks,
                    POSE_CONNECTIONS,
                    (255, 255, 255),
                    (0, 255, 255)
                )

                draw_selected_points(
                    frame,
                    result.pose_landmarks,
                    POSE_POINTS,
                    (0, 255, 255)
                )


            # -------------------------------------------------
            # FACE
            # -------------------------------------------------

            if result.face_landmarks:

                draw_selected_points(
                    frame,
                    result.face_landmarks,
                    FACE_POINTS,
                    (255, 0, 0)
                )


            # -------------------------------------------------
            # Status display
            # -------------------------------------------------

            face_status = (
                "YES"
                if result.face_landmarks
                else "NO"
            )

            pose_status = (
                "YES"
                if result.pose_landmarks
                else "NO"
            )

            left_status = (
                "YES"
                if result.left_hand_landmarks
                else "NO"
            )

            right_status = (
                "YES"
                if result.right_hand_landmarks
                else "NO"
            )


            cv2.putText(
                frame,
                f"Face: {face_status}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Pose: {pose_status}",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Left hand: {left_status}",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Right hand: {right_status}",
                (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )


            # -------------------------------------------------
            # Display
            # -------------------------------------------------

            cv2.imshow(
                "Gabay-Kamay Holistic Tracking",
                frame
            )


            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


    camera.release()
    cv2.destroyAllWindows()

    print("Camera closed.")


if __name__ == "__main__":
    main()