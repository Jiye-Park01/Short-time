import re
import torch
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
from pathlib import Path
from boxmot import DeepOCSORT
import cv2
from ultralytics import YOLO
import numpy as np
from video import frame_to_time, clip_video, crop_frame, frames_to_video, create_final_video, clip_audio
import subprocess
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4'}
app.config['PROCESS_FOLDER'] = 'process'
app.config['RESULTS_FOLDER'] = 'results'
app.config['STATIC_FOLDER'] = 'static/frames'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if not os.path.exists(app.config['PROCESS_FOLDER']):
    os.makedirs(app.config['PROCESS_FOLDER'])

if not os.path.exists(app.config['RESULTS_FOLDER']):
    os.makedirs(app.config['RESULTS_FOLDER'])

if not os.path.exists(app.config['STATIC_FOLDER']):
    os.makedirs(app.config['STATIC_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('base/index.html')  # 메인 페이지 렌더링

@app.route('/about')
def about():
    return render_template('base/about.html')  # about 페이지 렌더링

@app.route('/img_processing')
def img_processing():
    return render_template('base/img_processing.html')  # 이미지 처리 페이지 렌더링

@app.route('/img_choose')
def img_choose():
    return render_template('base/img_choose.html')  # 이미지 선택 페이지 렌더링

@app.route('/upload', methods=['POST'])
def upload_file():
    start_time = time.time()  # 시작 시간 기록

    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)  # Safe filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename).replace('\\', '/')
        file.save(file_path)

        # Run the tracking script
        reid_model_path = "osnet_x0_25_msmt17.pt"
        result_dir = app.config['RESULTS_FOLDER']
        name = filename.rsplit('.', 1)[0]

        frame_count = 0
        p_boxes = {}
        tracker = DeepOCSORT(
            model_weights=Path(reid_model_path),
            # device=torch.device("cuda:0"),  # Use GPU
            device=torch.device("mps"),
            fp16=False
        )

        cap = cv2.VideoCapture(file_path)  # Extract frames
        w, h, fps = (int(cap.get(x)) for x in
                     (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))  # Save frame size

        out = cv2.VideoWriter(os.path.join(app.config['PROCESS_FOLDER'], f'{name}.mp4').replace('\\', '/'),
                              cv2.VideoWriter_fourcc(*'mp4v'),
                              fps,
                              (w, h))
        # model = YOLO("tracking/weights/yolov8n.pt").to("cuda:0")  # Use GPU from Window

        model = YOLO("tracking/weights/yolov8n.pt").to("mps")  # Use GPU from MAC

        while cap.isOpened():
            frame_count += 1
            success, frame = cap.read()
            if success:
                results = model(frame)
                dets = []

                for result in results:
                    for detection in result.boxes.data.cpu().numpy():
                        x1, y1, x2, y2, conf, cls = detection
                        dets.append([x1, y1, x2, y2, conf, int(cls)])
                dets = np.array(dets)

                tracks = tracker.update(dets, frame)

                boxes = tracks[:, :4].tolist()
                track_ids = tracks[:, -1].tolist()

                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = box
                    track_id = int(track_id)
                    p_box = [frame_count, x1, y1, x2, y2, track_id]

                    if track_id not in p_boxes:
                        p_boxes[track_id] = []

                    p_boxes[track_id].append(p_box)
            else:
                break

        cap.release()
        np.save(os.path.join(app.config['PROCESS_FOLDER'], f'{name}_boxes.npy').replace('\\', '/'), p_boxes)

        # Save middle frame of each ID
        ids = []

        cap = cv2.VideoCapture(file_path)
        for track_id in p_boxes:
            ids.append(track_id)
            mid_index = len(p_boxes[track_id]) // 2
            mid_frame = p_boxes[track_id][mid_index]
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame[0])
            success, frame = cap.read()
            if success:
                x1, y1, x2, y2 = mid_frame[1:5]
                crop_img = frame[int(y1):int(y2), int(x1):int(x2)]
                crop_img_filename = f'{filename}_{track_id}.jpg'  # Add .mp4 to filename
                cv2.imwrite(os.path.join(app.config['STATIC_FOLDER'], crop_img_filename).replace('\\', '/'), crop_img)


        end_time = time.time()  # 종료 시간 기록
        elapsed_time = end_time - start_time  # 수행 시간 계산
        print(f"Time taken for /upload: {elapsed_time} seconds")  # 수행 시간 출력

        return render_template('base/img_choose.html', ids=ids, filename=filename)

    return redirect(request.url)

@app.route('/process', methods=['POST'])
def process_video():
    start_time = time.time()  # 시작 시간 기록

    if 'select_id' not in request.form or 'filename' not in request.form:
        return redirect(request.url)
    select_id = int(request.form['select_id'])
    
    filename = request.form['filename']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename).replace('\\', '/')
    name = filename.rsplit('.', 1)[0]

    p_boxes = []

    cap = cv2.VideoCapture(file_path)  # Extract frames
    w, h, fps = (int(cap.get(x)) for x in
                 (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))  # Save frame size

    with open(os.path.join(app.config['PROCESS_FOLDER'], f'{name}_boxes.npy').replace('\\', '/'), 'rb') as f:
        p_boxes_dict = np.load(f, allow_pickle=True).item()

    if select_id in p_boxes_dict:
        p_boxes = p_boxes_dict[select_id]

    if p_boxes:
        s_frame_num = int(p_boxes[0][0])
        e_frame_num = int(p_boxes[-1][0])
        s_time = frame_to_time(s_frame_num, fps)
        e_time = frame_to_time(e_frame_num, fps)

        if s_time == e_time:
            e_time = frame_to_time(s_frame_num + 1, fps)

        output_path = os.path.join(app.config['PROCESS_FOLDER'], name, str(select_id)).replace('\\', '/')
        os.makedirs(output_path, exist_ok=True)

        audio_file = clip_audio(name, file_path, s_time, e_time, output_path)
        video_file = clip_video(name, file_path, s_time, e_time, output_path)
        frame_file_path = crop_frame(p_boxes, video_file, output_path)
        video_file = frames_to_video(fps, frame_file_path, name, output_path)
        create_final_video(name, video_file, audio_file, app.config['RESULTS_FOLDER'])

    end_time = time.time()  # 종료 시간 기록
    elapsed_time = end_time - start_time  # 수행 시간 계산
    print(f"Time taken for /process: {elapsed_time} seconds")  # 수행 시간 출력

    return redirect(url_for('result', name=f'{name}_{select_id}'))

@app.route('/result')
def result():
    start_time = time.time()  # 시작 시간 기록
    name = request.args.get('name')

    # 숫자와 밑줄을 제거하여 기본 파일 이름을 가져옴
    base_name = re.sub(r'_\d+$', '', name)
    video_path = f"{base_name}.mp4"

    end_time = time.time()  # 종료 시간 기록
    elapsed_time = end_time - start_time  # 수행 시간 계산
    print(f"Time taken for /result: {elapsed_time} seconds")  # 수행 시간 출력

    return render_template('base/img_result.html', video_path=video_path)

@app.route('/results/<path:filename>')
def download_file(filename):
    # 숫자와 밑줄을 제거하여 기본 파일 이름을 가져옴
    base_name = re.sub(r'_\d+$', '', filename.rsplit('.', 1)[0])
    base_name_with_extension = f"{base_name}.mp4"
    return send_from_directory(app.config['RESULTS_FOLDER'], base_name_with_extension)

if __name__ == "__main__":
    app.run()
