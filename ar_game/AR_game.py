import pyglet
import cv2
import cv2.aruco as aruco
import numpy as np
import random
import sys

WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 768

video_id = 0
if len(sys.argv) > 1:
    video_id = int(sys.argv[1])

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
aruco_params = cv2.aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, aruco_params)

cap = cv2.VideoCapture(video_id)
if not cap.isOpened():
    print(f"Fehler: Kamera mit ID {video_id} konnte nicht geoeffnet werden.")
    sys.exit()

window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT, "AR Game - Adapted Playfield")

finger_x, finger_y = -1, -1
score = 0
camera_texture = None
show_info = True

class Target:
    def __init__(self):
        self.x = random.randint(50, WINDOW_WIDTH - 50)
        self.y = WINDOW_HEIGHT + 20
        self.radius = random.randint(15, 30)
        self.speed = random.randint(3, 6)
        self.color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        self.shape = pyglet.shapes.Circle(self.x, self.y, self.radius, color=self.color)

    def update(self):
        self.y -= self.speed
        self.shape.y = self.y

targets = [Target() for _ in range(4)]

score_label = pyglet.text.Label(
    f'Score: {score}', font_name='Arial', font_size=18,
    x=20, y=WINDOW_HEIGHT - 35, color=(255, 255, 255, 255)
)

info_label = pyglet.text.Label(
    'Suche Board (Bringe alle 4 Marker in die Kamera)...', font_name='Arial', font_size=14,
    x=WINDOW_WIDTH // 2, y=WINDOW_HEIGHT - 35, anchor_x='center', color=(255, 0, 0, 255)
)

def order_points(pts):
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def process_cv_logic():
    global finger_x, finger_y, camera_texture

    ret, frame = cap.read()
    if not ret or frame is None:
        return False

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    warped_display = frame.copy()
    markers_found = False

    if ids is not None and len(ids) == 4:
        detected_centers = []
        for marker_corners in corners:
            c = marker_corners[0]
            centroid = (int(c[:, 0].mean()), int(c[:, 1].mean()))
            detected_centers.append(centroid)

        if len(detected_centers) == 4:
            markers_found = True
            src_pts = order_points(detected_centers)
            
            dst_pts = np.array([
                [0, 0], 
                [WINDOW_WIDTH - 1, 0], 
                [WINDOW_WIDTH - 1, WINDOW_HEIGHT - 1], 
                [0, WINDOW_HEIGHT - 1]
            ], dtype="float32")

            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped_display = cv2.warpPerspective(frame, M, (WINDOW_WIDTH, WINDOW_HEIGHT))

            ycrcb = cv2.cvtColor(warped_display, cv2.COLOR_BGR2YCrCb)
            lower_skin = np.array([0, 133, 77], dtype=np.uint8)
            upper_skin = np.array([255, 173, 127], dtype=np.uint8)
            
            mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.erode(mask, kernel, iterations=2)
            mask = cv2.dilate(mask, kernel, iterations=2)
            mask = cv2.GaussianBlur(mask, (3, 3), 0)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > 400:
                    fingertip = min(largest_contour, key=lambda p: p[0][1])[0]
                    
                    finger_x = int(fingertip[0])
                    finger_y = WINDOW_HEIGHT - int(fingertip[1])
                    
                    cv2.circle(warped_display, (int(fingertip[0]), int(fingertip[1])), 8, (0, 255, 0), -1)
            else:
                finger_x, finger_y = -1, -1
    else:
        finger_x, finger_y = -1, -1

    if not markers_found:
        if ids is not None:
            aruco.drawDetectedMarkers(warped_display, corners, ids)
        if rejected is not None and len(rejected) > 0:
            aruco.drawDetectedMarkers(warped_display, rejected, borderColor=(0, 0, 255))
        
        warped_display = cv2.resize(warped_display, (WINDOW_WIDTH, WINDOW_HEIGHT))

    warped_display = cv2.flip(warped_display, 0)
    rgb_frame = cv2.cvtColor(warped_display, cv2.COLOR_BGR2RGB)
    
    image_data = rgb_frame.tobytes()
    camera_texture = pyglet.image.ImageData(
        WINDOW_WIDTH, WINDOW_HEIGHT, 'RGB', image_data, pitch=WINDOW_WIDTH * 3
    ).get_texture()

    return markers_found

def game_update(dt):
    global score, finger_x, finger_y, show_info

    markers_found = process_cv_logic()
    show_info = not markers_found

    if markers_found:
        for target in targets:
            target.update()

            if finger_x != -1 and finger_y != -1:
                distance = np.sqrt((target.x - finger_x)**2 + (target.y - finger_y)**2)
                if distance < target.radius:
                    score += 1
                    score_label.text = f'Score: {score}'
                    target.__init__()

            if target.y < -50:
                target.__init__()

@window.event
def on_draw():
    window.clear()

    if camera_texture:
        camera_texture.blit(0, 0, width=WINDOW_WIDTH, height=WINDOW_HEIGHT)

    if not show_info:
        for target in targets:
            target.shape.draw()

        if finger_x != -1 and finger_y != -1:
            cursor = pyglet.shapes.Circle(finger_x, finger_y, 10, color=(255, 0, 0))
            cursor.draw()
            
        score_label.draw()
    else:
        info_label.draw()

pyglet.clock.schedule_interval(game_update, 1/60.0)

if __name__ == '__main__':
    pyglet.app.run()
    cap.release()