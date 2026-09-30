import numpy as np


# ---------------------------------------------------------
# Gabay-Kamay Feature Extraction
#
# Produces one fixed-length feature vector per video frame.
#
# Features:
# - 21 left-hand landmarks
# - 21 right-hand landmarks
# - 9 selected upper-body landmarks
# - 12 selected facial landmarks
#
# Each landmark contains:
# x, y, z
#
# Total:
# (21 + 21 + 9 + 12) * 3 = 189 features
# ---------------------------------------------------------


# Selected upper-body pose landmarks
POSE_INDICES = [
    0,      # Nose
    11,     # Left shoulder
    12,     # Right shoulder
    13,     # Left elbow
    14,     # Right elbow
    15,     # Left wrist
    16,     # Right wrist
    23,     # Left hip
    24      # Right hip
]


# Selected facial landmarks
FACE_INDICES = [
    # Left eye
    33,
    133,

    # Right eye
    362,
    263,

    # Left eyebrow
    70,
    105,

    # Right eyebrow
    334,
    300,

    # Mouth
    61,
    291,
    13,
    14
]


LEFT_HAND_FEATURES = 21 * 3
RIGHT_HAND_FEATURES = 21 * 3
POSE_FEATURES = len(POSE_INDICES) * 3
FACE_FEATURES = len(FACE_INDICES) * 3

TOTAL_FEATURES = (
    LEFT_HAND_FEATURES
    + RIGHT_HAND_FEATURES
    + POSE_FEATURES
    + FACE_FEATURES
)


def extract_landmark_xyz(landmarks, expected_count):
    """
    Convert a landmark list into:

    [x1, y1, z1, x2, y2, z2, ...]

    If the landmarks are missing, return zeros so that
    every frame always has the same number of features.
    """

    if not landmarks:
        return np.zeros(expected_count * 3, dtype=np.float32)

    values = []

    for landmark in landmarks[:expected_count]:
        values.extend([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    # Safety padding
    missing = expected_count - len(landmarks)

    if missing > 0:
        values.extend([0.0] * missing * 3)

    return np.array(values, dtype=np.float32)


def extract_selected_landmarks(
    landmarks,
    selected_indices
):
    """
    Extract only specific landmark indices.

    Missing landmarks are represented by zeros.
    """

    if not landmarks:
        return np.zeros(
            len(selected_indices) * 3,
            dtype=np.float32
        )

    values = []

    for index in selected_indices:

        if index < len(landmarks):

            landmark = landmarks[index]

            values.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        else:
            values.extend([
                0.0,
                0.0,
                0.0
            ])

    return np.array(values, dtype=np.float32)


def extract_features(result):
    """
    Convert one MediaPipe Holistic result into
    a single fixed-length NumPy feature vector.
    """

    # ----------------------------
    # Left hand
    # ----------------------------

    left_hand = extract_landmark_xyz(
        result.left_hand_landmarks,
        21
    )


    # ----------------------------
    # Right hand
    # ----------------------------

    right_hand = extract_landmark_xyz(
        result.right_hand_landmarks,
        21
    )


    # ----------------------------
    # Upper body
    # ----------------------------

    pose = extract_selected_landmarks(
        result.pose_landmarks,
        POSE_INDICES
    )


    # ----------------------------
    # Face
    # ----------------------------

    face = extract_selected_landmarks(
        result.face_landmarks,
        FACE_INDICES
    )


    # Combine everything
    features = np.concatenate([
        left_hand,
        right_hand,
        pose,
        face
    ])


    # Safety check
    if len(features) != TOTAL_FEATURES:
        raise ValueError(
            f"Expected {TOTAL_FEATURES} features, "
            f"but got {len(features)}."
        )


    return features