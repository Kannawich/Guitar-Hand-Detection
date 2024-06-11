import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageTk
import joblib
import pandas as pd
import random
import tkinter as tk
from tkinter import Canvas, Button
import os
import sys

sift = cv2.SIFT_create()

def select_fretboard(event):
    global x1, y1, x2, y2, selected, move, paused, down

    if event.type == tk.EventType.ButtonPress:
        move = False
        selected = False
        down = True
        x1, y1 = event.x, event.y

    elif event.type == tk.EventType.Motion:
        if selected:
            pass
        else:
            if selected == False and paused == True and down == True:
                move = True
                x2, y2 = event.x, event.y

    elif event.type == tk.EventType.ButtonRelease:
        x2, y2 = event.x, event.y
        selected = True
        move = False
        down = False

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

def get_hand_landmarks(image):
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            return hand_landmarks.landmark
        
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def preprocess_single_image(image_path, transform):
    input_image = Image.open(image_path).convert('RGB')
    input_tensor = transform(input_image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch

index_pointdraw = [6,7,8]
middle_pointdraw = [10,11,12]
ring_pointdraw = [14,15,16]
pinky_pointdraw = [18,19,20]
angle_open_pointdraw = [0,1,5,17]
angle_barre_pointdraw = [5,6,7,8]
chord_pointdraw = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
barre_f = ['F#', 'G', 'G#', 'A', 'A#', 'B']
barre_fm = ['F#m', 'Gm', 'G#m', 'Am', 'A#m', 'Bm']
barre_bb = ['B', 'C', 'C#', 'D', 'D#', 'E']
barre_bbm = ['Bm', 'Cm', 'C#m', 'Dm', 'D#m', 'Em']

def get_cropped_frame(landmarks, frame, point_draw):
    image_size=(224,224)
    img_y, img_x, _ = frame.shape
    min_x = min(landmarks[i].x for i in point_draw)
    min_y = min(landmarks[i].y for i in point_draw)
    max_x = max(landmarks[i].x for i in point_draw)
    max_y = max(landmarks[i].y for i in point_draw)
    real_x_min, real_y_min = int(min_x * img_x), int(min_y * img_y)
    real_x_max, real_y_max = int(max_x * img_x), int(max_y * img_y)
    if real_x_min >= 0 and real_x_max <= img_x and real_y_min >= 0 and real_y_max <= img_y:
        hand_cropped = frame[int(real_y_min)-20:int(real_y_max)+20, int(real_x_min)-20:int(real_x_max)+20]
        if not hand_cropped.size == 0:
            hand_image = cv2.resize(hand_cropped, image_size)
            return hand_image

def get_cropped_frame_chord(landmarks, frame, point_draw):
    image_size=(224,224)
    img_y, img_x, _ = frame.shape
    min_x = min(landmarks[i].x for i in point_draw)
    min_y = min(landmarks[i].y for i in point_draw)
    max_x = max(landmarks[i].x for i in point_draw)
    max_y = max(landmarks[i].y for i in point_draw)
    real_x_min, real_y_min = int(min_x * img_x), int(min_y * img_y)
    real_x_max, real_y_max = int(max_x * img_x), int(max_y * img_y)
    if real_x_min >= 0 and real_x_max <= img_x and real_y_min >= 0 and real_y_max <= img_y:
        hand_cropped = frame[int(real_y_min)-15:int(real_y_max)+15, int(real_x_min)-15:int(real_x_max)+15]
        if not hand_cropped.size == 0:
            hand_image = cv2.resize(hand_cropped, image_size)
            return hand_image
    else:
        return frame[0:1, 0:1]
        
def process_realtime_landmark(landmarks, frame, point_draw):
    if landmarks:
        landmarks[0].y += 0.00
        decoded_landmarks = []
        for l in landmarks:
            x, y, z = l.x * frame.shape[1] * 3/4 * frame.shape[1]/frame.shape[0], l.y * frame.shape[0], l.z
            decoded_landmarks.append({'x': x, 'y': y, 'z': z})

        x0, y0, z0 = decoded_landmarks[0]['x'], decoded_landmarks[0]['y'], decoded_landmarks[0]['z']
        for landmark in decoded_landmarks:
            landmark['x'] -= x0
            landmark['y'] -= y0
            landmark['z'] -= z0

        x5, x17 = decoded_landmarks[5]['x'], decoded_landmarks[17]['x']
        scale_factor = abs(x17 - x5)

        for landmark in decoded_landmarks:
            landmark['x'] /= scale_factor
            landmark['y'] /= scale_factor
            landmark['z'] /= scale_factor

        index_finger_landmarks = [decoded_landmarks[i] for i in point_draw]
        features = []
        features.extend([landmark['x'] for landmark in index_finger_landmarks])
        features.extend([landmark['y'] for landmark in index_finger_landmarks])
        features.extend([landmark['z'] for landmark in index_finger_landmarks])
        feature_names = [f'Landmark_{1}_x', f'Landmark_{2}_x', f'Landmark_{3}_x', f'Landmark_{4}_x', f'Landmark_{1}_y', f'Landmark_{2}_y', f'Landmark_{3}_y', f'Landmark_{4}_y', f'Landmark_{1}_z', f'Landmark_{2}_z', f'Landmark_{3}_z', f'Landmark_{4}_z']
        feature_df = pd.DataFrame([features], columns=feature_names)
    return feature_df

def draw_black_contours(image):
    if image is None or image.size == 0:
        return image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 0, 0), 2)
    return image

def only_edge(image):
    if image is None or image.size == 0:
        return image
    rgb_planes = cv2.split(image)
    diff_plane = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((11, 11), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 31)
        diff_img = cv2.absdiff(plane, bg_img)
        inv_diff_img = 255 - diff_img
        norm_img = cv2.normalize(inv_diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        diff_plane.append(norm_img)
    edge_image = cv2.merge(diff_plane)
    return edge_image

def get_fret_num(frame):
    landmarks_barre = get_hand_landmarks(frame)
    if landmarks_barre:
        indexpos = (landmarks_barre[6].x + landmarks_barre[7].x + landmarks_barre[8].x)/3
        if indexpos > 0.82:
            fretnumber = 1
        elif indexpos > 0.69:
            fretnumber = 2
        elif indexpos > 0.56:
            fretnumber = 3
        elif indexpos > 0.45:
            fretnumber = 4
        elif indexpos > 0.35:
            fretnumber = 5
        elif indexpos > 0.25:
            fretnumber = 6
        else:
            fretnumber = 7
    else:
        fretnumber = 0
    return fretnumber

def predict_cnn_index(model, image, transform):
    if image is None:
        return 0
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = model(input_batch)
    show = torch.round(torch.sigmoid(output-13)).cpu().numpy()
    result = int(show[0])
    return result 

def predict_cnn_middle(model, image, transform):
    if image is None:
        return 0
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = model(input_batch)
    show = torch.round(torch.sigmoid(output-4)).cpu().numpy()
    result = int(show[0])
    return result 

def predict_cnn_ring(model, image, transform):
    if image is None:
        return 0
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = model(input_batch)
    show = torch.round(torch.sigmoid(output)).cpu().numpy()
    result = int(show[0])
    return result 

def predict_cnn_barre(model, image, transform):
    if image is None:
        return 0
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = model(input_batch)
    show = torch.round(torch.sigmoid(output+7)).cpu().numpy()
    result = int(show[0])
    return result 

def contains_classes(predictions, classes_to_check):
    predicted_classes = set(predictions)
    return set(classes_to_check).issubset(predicted_classes)

def check_order(top4_predictions, class1, class2):
    index1 = top4_predictions.index(class1)
    index2 = top4_predictions.index(class2)
    if index1 < index2:
        return True
    else:
        return False

def predict_chord(model, model_inside, image, transform):
    pred_text = None
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = model(input_batch)
    top4_probs, top4_classes = torch.topk(output, 4, dim=1)
    top4_classes = top4_classes.cpu().numpy().flatten()
    top4_predictions = list(top4_classes)
    if top4_predictions[0] == 0:
        pred_text = 'C'
    elif contains_classes(top4_predictions, [1,3]) and top4_predictions[0] == 1:
        pred_text = 'D'
    elif contains_classes(top4_predictions, [4,9]):
        pred_text = 'E'
    elif top4_predictions[0] == 5 or contains_classes(top4_predictions, [6,4,5]):
        pred_text = 'G'
    elif contains_classes(top4_predictions, [6,8,1]) and check_order(top4_predictions, 6, 8):
        pred_text = 'A'
    elif contains_classes(top4_predictions, [7,1]):
        pred_text = 'Dm'
    elif contains_classes(top4_predictions, [8,1]):
        pred_text = 'Em'
    elif contains_classes(top4_predictions, [9,7]) or contains_classes(top4_predictions, [9,6]):
        pred_text = 'Am'
    elif contains_classes(top4_predictions, [3,10]):
        with torch.no_grad():
            output_inside = model_inside(input_batch)
        show = torch.round(torch.sigmoid(output_inside + 5)).cpu().numpy()
        result = int(show[0])
        if int(result) == 0:
            pred_text = 'F'
        else:
            pred_text = 'Fm'
    elif top4_predictions[0] == 3:
        pred_text = 'A#m'
    elif top4_predictions[0] == 11:
        pred_text = 'A#'
    return pred_text

def predict_rf(model, feature_df):
    output = model.predict(feature_df)
    return int(output[0])

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class Net12(nn.Module):
    def __init__(self):
        super(Net12, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 12)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class MyLitModel(nn.Module):
    def __init__(self):
        super(MyLitModel, self).__init__()
        self.model = Net()
    def forward(self, x):
        return self.model(x)

class MyLitModelChord(nn.Module):
    def __init__(self):
        super(MyLitModelChord, self).__init__()
        self.model = Net12()
    def forward(self, x):
        return self.model(x)
    
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

model_index = MyLitModel()
checkpoint_index = torch.load(resource_path('models/index.ckpt'), map_location=torch.device('cpu'))
model_index.load_state_dict(checkpoint_index['state_dict'])
model_index.eval()

model_middle = MyLitModel()
checkpoint_middle = torch.load(resource_path('models/middle.ckpt'), map_location=torch.device('cpu'))
model_middle.load_state_dict(checkpoint_middle['state_dict'])
model_middle.eval()

model_ring = MyLitModel()
checkpoint_ring = torch.load(resource_path('models/ring.ckpt'), map_location=torch.device('cpu'))
model_ring.load_state_dict(checkpoint_ring['state_dict'])
model_ring.eval()

model_angle_open = joblib.load(resource_path('models/angle_open.pkl'))

model_angle_barre = MyLitModel()
checkpoint_angle_barre = torch.load(resource_path('models/angle-barre.ckpt'), map_location=torch.device('cpu'))
model_angle_barre.load_state_dict(checkpoint_angle_barre['state_dict'])
model_angle_barre.eval()

model_chord = MyLitModelChord()
checkpoint_chord = torch.load(resource_path('models/chord.ckpt'), map_location=torch.device('cpu'))
model_chord.load_state_dict(checkpoint_chord['state_dict'])
model_chord.eval()

model_f = MyLitModel()
checkpoint_f = torch.load(resource_path('models/E_Barre.ckpt'), map_location=torch.device('cpu'))
model_f.load_state_dict(checkpoint_f['state_dict'])
model_f.eval()

output_text = ["Correct!", "Don't bend your finger inward!"]
output_text_barre = ["Correct!", "Use side of your index finger to barre"]
output_text_angle = ["Correct!", "Lift up your wrist!"]

def on_closing():
    global cap
    cap.release()
    root.destroy()

def on_key(event):
    if event.keysym == 'Escape':
        on_closing()

def toggle_pause():
    global paused
    paused = not paused

def toggle_enter():
    global paused, confirm, width, height, selected
    if selected:
        confirm = True
        paused = False
        width = 280
        height = 160

root = tk.Tk()
root.title("Guitar Chord Recognition")
root.protocol("WM_DELETE_WINDOW", on_closing)
root.bind('<Key>', on_key)

main_canvas = Canvas(root, width=640, height=480)
main_canvas.pack()

pause_button = Button(root, text="Pause", command=toggle_pause)
pause_button.pack()

enter_button = Button(root, text="Enter", command=toggle_enter)
enter_button.pack()

main_canvas.bind('<ButtonPress-1>', select_fretboard)
main_canvas.bind('<B1-Motion>', select_fretboard)
main_canvas.bind('<ButtonRelease-1>', select_fretboard)

x1, y1, x2, y2 = -1, -1, -1, -1
landmarks = None
selected = False
paused = False
move = False
down = False
confirm = False
text_index = None
text_middle = None
text_ring = None
text_angle_open = None
text_angle_barre = None
output_chord = None
chords = ['F#', 'G', 'G#', 'A', 'A#', 'B', 'F#m', 'Gm', 'G#m', 'Am', 'A#m', 'Bm', 'B', 'C', 'C#', 'D', 'D#', 'E', 'Bm', 'Cm', 'C#m', 'Dm', 'D#m', 'Em']
random_chord = random.choice(chords)

def update_frame():
    global cap, paused, frame, frameraw, confirm, landmarks, object_roi, random_chord, output_chord, text_index, text_middle, text_ring, text_angle_open, text_angle_barre, selected, x1, x2, y1, y2, freeze, move
    sift = cv2.SIFT_create()
    if not paused:
        move = False
        x1 = -1
        y1 = -1
        x2 = -1
        y2 = -1
        selected = False
        ret, frameraw = cap.read()
        frame = frameraw.copy()
        if not ret:
            return
        if confirm == False:
            cv2.putText(frameraw, f"Move your camera to see your entire fretboard", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3) 
            cv2.putText(frameraw, f"Click 'Pause' button to select the fretboard", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(frameraw, f"Move your camera to see your entire fretboard", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2) 
            cv2.putText(frameraw, f"Click 'Pause' button to select the fretboard", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        freeze = frame.copy()
        if confirm == True and paused == False:
            cv2.putText(frameraw, f"Chord to play : {random_chord}", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(frameraw, f"Chord to play : {random_chord}", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            #crop and process image for each finger
            landmarks = get_hand_landmarks(frameraw)
            #SIFT Part
            grayobject = cv2.cvtColor(object_roi, cv2.COLOR_BGR2GRAY)
            keypoints_obj, descriptors_obj = sift.detectAndCompute(grayobject, None)  
            grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            keypoints_fr, descriptors_fr = sift.detectAndCompute(grayframe, None)
            #Matchhhhhhh
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(descriptors_obj, descriptors_fr, k=2)
            #RANSAC
            obj_pts = []
            fr_pts = []
            good_matches = []
            #create list of point
            for m,n in matches:
                if m.distance < 0.7 * n.distance:
                    obj_pts.append(keypoints_obj[m.queryIdx].pt)
                    fr_pts.append(keypoints_fr[m.trainIdx].pt)
                    good_matches.append(m)
            #RESIZE
            if len(obj_pts) < 4 or len(fr_pts) < 4:
                pass
            else:
                obj_pts = np.float32(obj_pts).reshape(-1, 1, 2)
                fr_pts = np.float32(fr_pts).reshape(-1, 1, 2)
                #RANSEC
                homography, mask = cv2.findHomography(obj_pts, fr_pts, cv2.RANSAC, 6.0)
                #DRAW
                pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(-1, 1, 2)
                pts2x = np.float32([[0, 0], [width*2 - 20, 0], [width*2 - 20, height*2], [0, height*2]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts,homography)
                #area = cv2.contourArea(np.int32(dst))
                dst_change = dst + np.array([33 , 0]).reshape(-1, 1, 2)
                middle_point = np.mean(dst_change, axis=0)
                translated_dst = dst_change - middle_point
                scaled_dst = translated_dst * 1.25
                expanded_dst = scaled_dst + middle_point
                
                #Expand to cover full hand
                if len(expanded_dst) >= 4:
                    expanded_dst[0][0][1] -= 10
                    expanded_dst[1][0][1] -= 10
                    expanded_dst[2][0][1] += 20
                    expanded_dst[3][0][1] += 20
                    if expanded_dst[3][0][1] > frame.shape[0]:
                        expanded_dst[3][0][1] = frame.shape[0]
                    if expanded_dst[2][0][1] > frame.shape[0]:
                        expanded_dst[2][0][1] = frame.shape[0]

                mask = np.zeros_like(frame)
                cv2.fillPoly(mask, [np.int32(expanded_dst)], (255, 255, 255))
                roiarea = cv2.bitwise_and(frame, mask)
                matrixtransform = cv2.getPerspectiveTransform(np.array(expanded_dst, dtype=np.float32), pts2x)
                track = cv2.warpPerspective(roiarea, matrixtransform, (width*2 - 20, height*2))

            if landmarks:
                index_pic = draw_black_contours(get_cropped_frame(landmarks, frameraw, index_pointdraw))
                middle_pic = get_cropped_frame(landmarks, frameraw, middle_pointdraw)
                ring_pic = get_cropped_frame(landmarks, frameraw, ring_pointdraw)
                angle_barre_pic = draw_black_contours(get_cropped_frame(landmarks, frameraw, angle_barre_pointdraw))
                chord_pic = only_edge(get_cropped_frame_chord(landmarks, frameraw, chord_pointdraw))
                angle_open_landmark = process_realtime_landmark(landmarks, frameraw, angle_open_pointdraw)
                #cv2.imshow('chord', chord_pic)
                text_index = output_text[predict_cnn_index(model_index, index_pic, transform)]
                text_middle = output_text[predict_cnn_middle(model_middle, middle_pic, transform)]
                text_ring = output_text[predict_cnn_ring(model_ring, ring_pic, transform)]
                text_angle_open = output_text_angle[predict_rf(model_angle_open, angle_open_landmark)]
                text_angle_barre = output_text_barre[predict_cnn_barre(model_angle_barre, angle_barre_pic, transform)]
                output_chord = predict_chord(model_chord, model_f, chord_pic, transform)
            if output_chord == 'F' or output_chord == 'Fm' or output_chord == 'A#' or output_chord == 'A#m':
                if output_chord == 'F' and get_fret_num(track) != 0 and get_fret_num(track) != 1: output_chord = barre_f[get_fret_num(track)-2]
                elif output_chord == 'Fm' and get_fret_num(track) != 0 and get_fret_num(track) != 1: output_chord = barre_fm[get_fret_num(track)-2]
                elif output_chord == 'A#' and get_fret_num(track) != 0 and get_fret_num(track) != 1: output_chord = barre_bb[get_fret_num(track)-2]
                elif output_chord == 'A#m' and get_fret_num(track) != 0 and get_fret_num(track) != 1: output_chord = barre_bbm[get_fret_num(track)-2]
                cv2.putText(frameraw, f"Index : {text_angle_barre}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                cv2.putText(frameraw, f"Index : {text_angle_barre}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                #cv2.putText(frameraw, f"Fret : {get_fret_num(track)}", (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frameraw, f"Index : {text_index}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                cv2.putText(frameraw, f"Angle_Open : {text_angle_open}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                cv2.putText(frameraw, f"Index : {text_index}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frameraw, f"Angle_Open : {text_angle_open}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frameraw, f"Middle : {text_middle}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(frameraw, f"Ring : {text_ring}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            #cv2.putText(frameraw, f"Chord : {output_chord}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(frameraw, f"Middle : {text_middle}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frameraw, f"Ring : {text_ring}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            #cv2.putText(frameraw, f"Chord : {output_chord}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if output_chord == random_chord:
                random_chord = random.choice(chords)
                #cv2.putText(frameraw, f"Right!", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frameraw, f"Chord to play : {random_chord} Right!", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                cv2.putText(frameraw, f"Chord to play : {random_chord} Right!", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        img = cv2.cvtColor(frameraw, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        main_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        main_canvas.imgtk = imgtk
    else:
        if x1 != -1 and y1 != -1 and x2 != -1 and y2 != -1 and selected == True:
            sift = cv2.SIFT_create()
            object_roi = freeze[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
            if object_roi.size == 0:
                pass
            mask = np.zeros_like(freeze[:,:,0], dtype=np.uint8)
            mask[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)] = 255
            result = cv2.bitwise_and(freeze, freeze, mask=mask)
        else:
            result = freeze
        #selected part
        if not selected:
            cv2.putText(freeze, f"Select your fretboard area from fret 1-9", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(freeze, f"Select your fretboard area from fret 1-9", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            #cv2.imshow('Select object', freeze)
            img = cv2.cvtColor(freeze, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            main_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            main_canvas.imgtk = imgtk
            if move == True and down == True:
                temp_framemove = freeze.copy()
                cv2.rectangle(temp_framemove, (x1, y1), (x2, y2), (0, 255, 0), 2)
                img = cv2.cvtColor(temp_framemove, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)
                main_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
                main_canvas.imgtk = imgtk
                #cv2.imshow('Select object', temp_framemove)
        else:
            temp_frame = freeze.copy()
            cv2.rectangle(temp_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            img = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            main_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            main_canvas.imgtk = imgtk
    main_canvas.after(10, update_frame)

cap = cv2.VideoCapture(0)
update_frame()

root.mainloop()