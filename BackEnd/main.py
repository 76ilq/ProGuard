from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import cv2
import mediapipe as mp
import tempfile

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
landmark_names = mp_pose.PoseLandmark.__members__  # Dict of valid keypoint names

@app.get("/")
async def root():
    return {"message": "Hello ProGuard 👋"}

# Endpoint: POST /highlight_keypoint/{keypoint_name}
# 
# This endpoint takes a video and a single body keypoint name (e.g., LEFT_KNEE),
# and returns the video with only that keypoint highlighted in green.
# 
# Example usage:
#   /highlight_keypoint/LEFT_KNEE
#   /highlight_keypoint/NOSE
#   /highlight_keypoint/RIGHT_ANKLE
#
# Valid keypoint names (from MediaPipe PoseLandmark):
#   NOSE
#   LEFT_EYE_INNER, LEFT_EYE, LEFT_EYE_OUTER
#   RIGHT_EYE_INNER, RIGHT_EYE, RIGHT_EYE_OUTER
#   LEFT_EAR, RIGHT_EAR
#   MOUTH_LEFT, MOUTH_RIGHT
#   LEFT_SHOULDER, RIGHT_SHOULDER
#   LEFT_ELBOW, RIGHT_ELBOW
#   LEFT_WRIST, RIGHT_WRIST
#   LEFT_PINKY, RIGHT_PINKY
#   LEFT_INDEX, RIGHT_INDEX
#   LEFT_THUMB, RIGHT_THUMB
#   LEFT_HIP, RIGHT_HIP
#   LEFT_KNEE, RIGHT_KNEE
#   LEFT_ANKLE, RIGHT_ANKLE
#   LEFT_HEEL, RIGHT_HEEL
#   LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX
#
# NOTE: keypoint_name is case-insensitive but must match exactly (e.g., LEFT_KNEE not leftknee)


@app.post("/highlight_keypoint/{keypoint_name}")
async def highlight_keypoint(keypoint_name: str, file: UploadFile = File(...)):
    keypoint_name = keypoint_name.upper()
    
    if keypoint_name not in landmark_names:
        raise HTTPException(status_code=400, detail=f"Invalid keypoint: {keypoint_name}")

    keypoint_id = mp_pose.PoseLandmark[keypoint_name].value

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(await file.read())
        temp_path = temp_file.name

    cap = cv2.VideoCapture(temp_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = temp_path.replace(".mp4", f"_{keypoint_name.lower()}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frame_rgb)

        if result.pose_landmarks:
            landmark = result.pose_landmarks.landmark[keypoint_id]
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)  # green dot

        out.write(frame)

    cap.release()
    out.release()

    return FileResponse(output_path, media_type="video/mp4", filename=f"{keypoint_name.lower()}.mp4")
