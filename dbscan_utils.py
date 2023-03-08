import numpy as np
import cv2
from tqdm import tqdm
from Football_Recognize.person_det import PersonDetectionSMSV2
from Football_Recognize.get_keypoints import KPDetection
from Football_Recognize.utils.visualize import draw_bboxes_xy
from Football_Recognize.detection_utils import get_vectors, get_relative_bboxes
from Football_Recognize.visualization_utils import color_fn

def get_arrays_from_video(video_path):
    
    person_detector = PersonDetectionSMSV2()
    kps_detector = KPDetection()

    video_cap = cv2.VideoCapture(video_path)

    window_counter = 0
    
    all_frames = []
    all_vectors = []
    number_of_vectors_in_frame = []
    all_rel_bboxes = []
    all_scores = []
        
    while True:
        ret, frame = video_cap.read()
        fps = int(video_cap.get(cv2.CAP_PROP_FPS))
        if not ret:
            break

        bboxes, probs = person_detector(frame)
        kps = kps_detector.get_kps_by_bboxes(frame,bboxes,scale_bboxes=True)

        height, width = frame.shape[:2]
        rel_bboxes = get_relative_bboxes(bboxes, height, width)

        vectors = get_vectors(frame, kps)
        number_of_vectors_in_frame += [len(vectors)]
        all_vectors += vectors 
        all_frames += [frame]
        all_rel_bboxes += [rel_bboxes]
        all_scores += [probs]

               
        if window_counter % 50 == 0:
            print(f"Processed {window_counter} frames.")
        window_counter += 1

    video_cap.release()
    return all_frames, all_vectors, number_of_vectors_in_frame, all_rel_bboxes, all_scores, fps


def export_video_with_predictor(output_path, predictor, frames, fps, all_vectors, number_of_vectors_in_frame, rel_bboxes, scores):
    
    num_frames = len(frames)
    vectors = all_vectors.copy()
    all_labels = []
    processed_frames = []

    print("Processing frames:")
    for i in tqdm(range(num_frames)):
        vec = vectors[:number_of_vectors_in_frame[i]]
        labels = predictor.predict(vec)
        # labels = dbscan_predict_batches(dbscan, vec)
        all_labels += list(labels)
        frame_teams = draw_bboxes_xy(frames[i].copy(), classes=labels, bboxes=rel_bboxes[i], 
                                     scores=scores[i], color_fn=color_fn, 
                                     is_relative_coordinate=True, thickness=1,
                                     show_text=False, font_scale=0.6, text_color=(255, 255, 255))

        processed_frames += [frame_teams]
        vectors = vectors[number_of_vectors_in_frame[i]:, :]

    height, width = frames[i].shape[:2]
    print("Saving video...")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in processed_frames:
        writer.write(frame)
    writer.release()
    print("Done!")
    return all_labels
