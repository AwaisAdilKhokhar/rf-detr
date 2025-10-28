# -*- coding: utf-8 -*-
"""
RF-DETR POWERED PTZ TRACKER
- Replaces YOLO with RF-DETR for detection
- Keeps improved ByteTrack-style tracker and PTZ simulation
- Supports detection interval and optional raw detection overlay
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import cv2
import numpy as np
import torch
import time
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

from simulated_ptz import SimulatedPTZ
from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES


@dataclass
class STrack:
    """Simple track object for ByteTrack"""
    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    score: float
    class_name: str
    state: str = 'tracked'  # tracked, lost, removed
    frames_lost: int = 0
    hit_streak: int = 0
    age: int = 0
    exit_time: Optional[float] = None  # Timestamp when track is removed
    
    @property
    def tlbr(self):
        return self.bbox
    
    @property
    def center(self):
        x1, y1, x2, y2 = self.bbox
        return np.array([(x1+x2)/2, (y1+y2)/2])


class ImprovedByteTracker:
    """Improved ByteTrack with better ghost track prevention"""
    def __init__(self, track_thresh=0.3, match_thresh=0.5, max_lost=20, min_box_area=50):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.low_thresh = 0.1
        self.max_lost = max_lost
        self.min_box_area = min_box_area
        self.max_lost_selected = max_lost * 3
        
        self.tracks_active: List[STrack] = []
        self.tracks_lost: List[STrack] = []
        self.track_id_count = 0
        
        self.min_hit_streak = 1
    
    def update(self, detections, selected_track_id: Optional[int] = None):
        removed = []
        
        for track in self.tracks_active + self.tracks_lost:
            track.age += 1
        
        if len(detections) == 0:
            for track in self.tracks_active + self.tracks_lost:
                track.frames_lost += 1
                track.hit_streak = 0
                max_lost_allowed = self.max_lost_selected if (selected_track_id is not None and track.track_id == selected_track_id) else self.max_lost
                if track.frames_lost > max_lost_allowed:
                    track.state = 'removed'
                    if track.exit_time is None:
                        track.exit_time = time.time()
            removed = [t for t in self.tracks_active + self.tracks_lost if t.state == 'removed']
            self.tracks_active = [t for t in self.tracks_active if t.state != 'removed']
            self.tracks_lost = [t for t in self.tracks_lost if t.state != 'removed']
            return self.tracks_active, removed
        
        filtered = []
        for det in detections:
            x1, y1, x2, y2 = det[:4]
            if (x2 - x1) * (y2 - y1) >= self.min_box_area:
                filtered.append(det)
        
        if len(filtered) == 0:
            for track in self.tracks_active + self.tracks_lost:
                track.frames_lost += 1
                track.hit_streak = 0
                max_lost_allowed = self.max_lost_selected if (selected_track_id is not None and track.track_id == selected_track_id) else self.max_lost
                if track.frames_lost > max_lost_allowed:
                    track.state = 'removed'
                    if track.exit_time is None:
                        track.exit_time = time.time()
            removed = [t for t in self.tracks_active + self.tracks_lost if t.state == 'removed']
            self.tracks_active = [t for t in self.tracks_active if t.state != 'removed']
            self.tracks_lost = [t for t in self.tracks_lost if t.state != 'removed']
            return self.tracks_active, removed
        
        detections = filtered
        high_dets, low_dets = [], []
        for det in detections:
            if det[4] >= self.track_thresh:
                high_dets.append(det)
            elif det[4] >= self.low_thresh:
                low_dets.append(det)
        
        matched_active, unmatched_tracks_active, unmatched_dets_high = self._match_detections(
            self.tracks_active, high_dets, thresh=self.match_thresh
        )
        for track_idx, det_idx in matched_active:
            self.tracks_active[track_idx].bbox = np.array(high_dets[det_idx][:4])
            self.tracks_active[track_idx].score = high_dets[det_idx][4]
            self.tracks_active[track_idx].frames_lost = 0
            self.tracks_active[track_idx].hit_streak += 1
            self.tracks_active[track_idx].state = 'tracked'
        
        unmatched_tracks_lost = []
        if len(self.tracks_lost) > 0 and len(unmatched_dets_high) > 0:
            remaining_high = [high_dets[i] for i in unmatched_dets_high]
            # First, try to re-associate the selected lost track with a lower threshold
            if selected_track_id is not None:
                sel_indices = [i for i, t in enumerate(self.tracks_lost) if t.track_id == selected_track_id]
                if len(sel_indices) > 0:
                    sel_idx = sel_indices[0]
                    ious = []
                    for j, det in enumerate(remaining_high):
                        ious.append((j, self._iou(self.tracks_lost[sel_idx].bbox, det[:4])))
                    if len(ious) > 0:
                        best_det_idx, best_iou = max(ious, key=lambda x: x[1])
                        if best_iou >= 0.2:
                            track = self.tracks_lost[sel_idx]
                            track.bbox = np.array(remaining_high[best_det_idx][:4])
                            track.score = remaining_high[best_det_idx][4]
                            track.frames_lost = 0
                            track.hit_streak = max(1, track.hit_streak + 1)
                            track.state = 'tracked'
                            self.tracks_active.append(track)
                            self.tracks_lost.remove(track)
                            # remove this det from unmatched_dets_high
                            det_global_idx = unmatched_dets_high[best_det_idx]
                            unmatched_dets_high = [i for i in unmatched_dets_high if i != det_global_idx]
                            remaining_high = [high_dets[i] for i in unmatched_dets_high]
            matched_lost, unmatched_tracks_lost, unmatched_dets_high2 = self._match_detections(
                self.tracks_lost, remaining_high, thresh=0.4
            )
            reactivated = []
            for track_idx, det_idx in matched_lost:
                track = self.tracks_lost[track_idx]
                track.bbox = np.array(remaining_high[det_idx][:4])
                track.score = remaining_high[det_idx][4]
                track.frames_lost = 0
                track.hit_streak = 1
                track.state = 'tracked'
                reactivated.append(track)
                self.tracks_active.append(track)
            for track in reactivated:
                self.tracks_lost.remove(track)
            unmatched_dets_high = [unmatched_dets_high[j] for j in unmatched_dets_high2]
        
        for idx in unmatched_tracks_lost:
            if idx < len(self.tracks_lost):
                self.tracks_lost[idx].frames_lost += 1
                self.tracks_lost[idx].hit_streak = 0
                max_lost_allowed = self.max_lost_selected if (selected_track_id is not None and self.tracks_lost[idx].track_id == selected_track_id) else self.max_lost
                if self.tracks_lost[idx].frames_lost > max_lost_allowed:
                    self.tracks_lost[idx].state = 'removed'
                    if self.tracks_lost[idx].exit_time is None:
                        self.tracks_lost[idx].exit_time = time.time()
        
        if len(unmatched_tracks_active) > 0 and len(low_dets) > 0:
            remaining_active = [self.tracks_active[i] for i in unmatched_tracks_active]
            matched_low, unmatched_tracks_low, _ = self._match_detections(
                remaining_active, low_dets, thresh=0.3
            )
            matched_actual = []
            for track_idx, det_idx in matched_low:
                actual_idx = unmatched_tracks_active[track_idx]
                self.tracks_active[actual_idx].bbox = np.array(low_dets[det_idx][:4])
                self.tracks_active[actual_idx].score = low_dets[det_idx][4]
                self.tracks_active[actual_idx].frames_lost = 0
                self.tracks_active[actual_idx].hit_streak += 1
                self.tracks_active[actual_idx].state = 'tracked'
                matched_actual.append(actual_idx)
            unmatched_tracks_active = [i for i in unmatched_tracks_active if i not in matched_actual]
        
        for idx in list(unmatched_tracks_active):
            if idx >= len(self.tracks_active):
                continue
            track = self.tracks_active[idx]
            track.frames_lost += 1
            track.hit_streak = 0
            max_lost_allowed = self.max_lost_selected if (selected_track_id is not None and track.track_id == selected_track_id) else self.max_lost
            if track.frames_lost > max_lost_allowed:
                track.state = 'removed'
                if track.exit_time is None:
                    track.exit_time = time.time()
            elif track.frames_lost > 2:
                track.state = 'lost'
                self.tracks_lost.append(track)
        
        removed += [t for t in self.tracks_active if t.state == 'removed']
        self.tracks_active = [t for t in self.tracks_active if t.state == 'tracked']
        
        for idx in unmatched_dets_high:
            # Avoid creating a new track if this detection overlaps selected lost track; reassign instead
            if selected_track_id is not None:
                sel_lost = [t for t in self.tracks_lost if t.track_id == selected_track_id]
                if len(sel_lost) > 0:
                    if self._iou(sel_lost[0].bbox, high_dets[idx][:4]) >= 0.2:
                        track = sel_lost[0]
                        track.bbox = np.array(high_dets[idx][:4])
                        track.score = high_dets[idx][4]
                        track.frames_lost = 0
                        track.hit_streak = max(1, track.hit_streak + 1)
                        track.state = 'tracked'
                        self.tracks_active.append(track)
                        self.tracks_lost.remove(track)
                        continue
            if high_dets[idx][4] >= self.track_thresh * 0.8:
                new_track = STrack(
                    track_id=self.track_id_count,
                    bbox=np.array(high_dets[idx][:4]),
                    score=high_dets[idx][4],
                    class_name=high_dets[idx][5] if len(high_dets[idx]) > 5 else 'unknown'
                )
                self.tracks_active.append(new_track)
                self.track_id_count += 1
        
        removed += [t for t in self.tracks_lost if t.state == 'removed']
        self.tracks_lost = [t for t in self.tracks_lost if t.state != 'removed']
        return self.tracks_active, removed
    
    def _match_detections(self, tracks, detections, thresh=None):
        if thresh is None:
            thresh = self.match_thresh
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))
        iou_matrix = self._calculate_iou_matrix(tracks, detections)
        matched = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_dets = list(range(len(detections)))
        if iou_matrix.size > 0:
            sorted_indices = np.unravel_index(
                np.argsort(-iou_matrix, axis=None), iou_matrix.shape
            )
            for track_idx, det_idx in zip(sorted_indices[0], sorted_indices[1]):
                if track_idx in unmatched_tracks and det_idx in unmatched_dets:
                    if iou_matrix[track_idx, det_idx] >= thresh:
                        matched.append([track_idx, det_idx])
                        unmatched_tracks.remove(track_idx)
                        unmatched_dets.remove(det_idx)
                    elif iou_matrix[track_idx, det_idx] < 0.1:
                        break
        return matched, unmatched_tracks, unmatched_dets
    
    def _calculate_iou_matrix(self, tracks, detections):
        matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                matrix[i, j] = self._iou(track.bbox, det[:4])
        return matrix
    
    def _iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0


class OptimizedPTZTracker(SimulatedPTZ):
    def __init__(self, camera_source=0, output_width=1280, output_height=720,
                 model_variant='nano', target_class='person', 
                 detect_interval=5, use_gpu=False, inference_size=640,
                 optimize=False):
        super().__init__(camera_source, output_width, output_height)
        
        self.detect_interval = detect_interval
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        self.inference_size = inference_size  # square size or None
        self.selected_hold_frames = 30
        self.selected_last_seen_frame = 0
        
        # MODEL LOADING (native RF-DETR)
        model_map = {
            'nano': RFDETRNano,
            'small': RFDETRSmall,
            'medium': RFDETRMedium,
            'base': RFDETRBase,
        }
        ModelClass = model_map.get(model_variant, RFDETRNano)
        print(f"🚀 Loading RF-DETR ({ModelClass.__name__}) on {'CUDA' if self.use_gpu else 'CPU'}...")
        self.model = ModelClass()
        if optimize:
            try:
                print("⚡ Optimizing model for inference (TorchScript)...")
                self.model.optimize_for_inference()
                print("✓ Model optimized")
            except Exception as e:
                print(f"⚠️ Optimization failed: {e}")
        
        # Detection thresholds
        self.conf_thresh = 0.25
        
        # Improved ByteTracker
        self.tracker = ImprovedByteTracker(
            track_thresh=0.3,
            match_thresh=0.5,
            max_lost=20,
            min_box_area=50
        )
        
        # Tracking settings
        self.target_class = target_class
        self._target_class_id = COCO_CLASSES.index(target_class) if target_class in COCO_CLASSES else None
        self.tracking_enabled = True
        self.auto_zoom = True
        
        # Frame counters
        self.frame_count = 0
        self.last_detect_frame = 0
        
        # Smoothing
        self.bbox_smooth = 0.7
        self.ptz_smooth = 0.85
        self.last_pan = 0.5
        self.last_tilt = 0.5
        self.last_zoom = 1.0
        
        # Track storage
        self.selected_track_id: Optional[int] = None
        self.active_tracks: List[STrack] = []
        self.track_boxes = {}
        self.exited_tracks = []
        
        # Debug
        self.show_raw_detections = False
        self.raw_detections = []  # list of dicts
        
        # Performance
        self.fps_time = time.time()
        self.fps_count = 0
        self.current_fps = 0
        self.detect_time = 0
        
        # UI
        self.is_video = isinstance(camera_source, str) and os.path.isfile(camera_source)
        cv2.namedWindow('TRACKING VIEW', cv2.WINDOW_NORMAL)
        cv2.namedWindow('PTZ VIEW', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('TRACKING VIEW', 1280, 720)
        cv2.resizeWindow('PTZ VIEW', 960, 540)
        cv2.setMouseCallback('TRACKING VIEW', self.mouse_callback)
        
        print("✅ Optimized RF-DETR Tracker Ready!")
        print(f"📊 Detection interval: {detect_interval} frames")
    
    def is_near_edge(self, bbox, w, h, margin=20):
        x1, y1, x2, y2 = bbox
        return x1 <= margin or x2 >= w - margin or y1 <= margin or y2 >= h - margin
    
    def get_exit_direction(self, bbox, w, h):
        x1, y1, x2, y2 = map(int, bbox)
        dists = {'left': x1, 'right': w - x2, 'top': y1, 'bottom': h - y2}
        return min(dists, key=dists.get)
    
    def connect(self):
        self.cap = cv2.VideoCapture(self.camera_source)
        if not self.cap.isOpened():
            return False
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ret, _ = self.cap.read()
        if not ret:
            return False
        print("✅ Camera Connected!")
        return True
    
    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return False
        self.original_frame = frame
        self.frame_count += 1
        return True
    
    def _run_rfdetr(self, frame_bgr):
        """Run RF-DETR on a frame (handles optional resize and scaling back)."""
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        scale_x = 1.0
        scale_y = 1.0
        input_frame = frame_rgb
        if self.inference_size:
            S = self.inference_size
            input_frame = cv2.resize(frame_rgb, (S, S))
            scale_x = w / float(S)
            scale_y = h / float(S)
        detections = self.model.predict(input_frame, threshold=self.conf_thresh)
        if detections is None or len(detections) == 0:
            return None
        # Scale boxes back if needed
        if self.inference_size and detections.xyxy is not None and len(detections.xyxy) > 0:
            detections.xyxy[:, [0, 2]] *= scale_x
            detections.xyxy[:, [1, 3]] *= scale_y
        return detections
    
    def detect_and_track(self, frame):
        h, w = frame.shape[:2]
        should_detect = (self.frame_count - self.last_detect_frame) >= self.detect_interval
        det_list = []
        self.raw_detections = []
        if should_detect:
            t1 = time.time()
            dets = self._run_rfdetr(frame)
            if dets is not None and dets.xyxy is not None and len(dets.xyxy) > 0:
                boxes = dets.xyxy
                class_ids = dets.class_id
                confidences = dets.confidence
                for i in range(len(boxes)):
                    cls_id = int(class_ids[i]) if class_ids is not None else -1
                    name = COCO_CLASSES[cls_id] if 0 <= cls_id < len(COCO_CLASSES) else 'unknown'
                    conf = float(confidences[i]) if confidences is not None else 0.0
                    x1, y1, x2, y2 = map(float, boxes[i])
                    # Filter by target class
                    if self._target_class_id is not None and cls_id != self._target_class_id:
                        continue
                    # Filter by confidence
                    if conf < 0.2:
                        continue
                    # Filter by min size
                    if (x2 - x1) < 10 or (y2 - y1) < 10:
                        continue
                    det_list.append([x1, y1, x2, y2, conf, name])
                    self.raw_detections.append({
                        'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2,
                        'confidence': conf, 'name': name
                    })
            self.last_detect_frame = self.frame_count
            self.detect_time = time.time() - t1
        
        self.active_tracks, removed = self.tracker.update(det_list, self.selected_track_id)
        for track in self.tracker.tracks_lost[:]:
            bbox = self.track_boxes.get(track.track_id, track.bbox)
            if self.is_near_edge(bbox, w, h):
                track.state = 'removed'
                track.exit_time = time.time()
                self.tracker.tracks_lost.remove(track)
                removed.append(track)
        for track in removed:
            bbox = self.track_boxes.get(track.track_id, track.bbox)
            if self.is_near_edge(bbox, w, h):
                self.exited_tracks.append((track.track_id, track.exit_time, bbox))
        self.exited_tracks = [(tid, etime, bbox) for tid, etime, bbox in self.exited_tracks if etime > time.time() - 10]
        self._smooth_track_boxes()
        # Update last seen for selected
        if self.selected_track_id is not None:
            for t in self.active_tracks:
                if t.track_id == self.selected_track_id:
                    self.selected_last_seen_frame = self.frame_count
                    break
        current_ids = {track.track_id for track in self.active_tracks}
        self.track_boxes = {tid: box for tid, box in self.track_boxes.items() if tid in current_ids}
        return self.active_tracks
    
    def _smooth_track_boxes(self):
        for track in self.active_tracks:
            if track.track_id not in self.track_boxes:
                self.track_boxes[track.track_id] = track.bbox.astype(float)
            else:
                old_box = self.track_boxes[track.track_id]
                new_box = track.bbox.astype(float)
                smoothed = self.bbox_smooth * new_box + (1 - self.bbox_smooth) * old_box
                self.track_boxes[track.track_id] = smoothed
                track.bbox = smoothed.astype(int)
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.tracking_enabled:
            self.select_track_at(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clear_selection()
    
    def select_track_at(self, click_x, click_y):
        best_dist = float('inf')
        best_track_id = None
        for track in self.active_tracks:
            if track.state != 'tracked':
                continue
            x1, y1, x2, y2 = track.bbox
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            dist = np.sqrt((click_x - cx)**2 + (click_y - cy)**2)
            if dist < 100 and dist < best_dist:
                best_dist = dist
                best_track_id = track.track_id
        if best_track_id is not None:
            self.selected_track_id = best_track_id
            print(f"🎯 Selected Track ID: {best_track_id}")
    
    def clear_selection(self):
        self.selected_track_id = None
        print("✗ Selection Cleared")
    
    def get_selected_track(self):
        if self.selected_track_id is None:
            return None
        for track in self.active_tracks:
            if track.track_id == self.selected_track_id and track.state == 'tracked':
                return track
        # Fallback: return lost selected track within hold window
        for track in getattr(self.tracker, 'tracks_lost', []):
            if track.track_id == self.selected_track_id:
                # optionally clamp to last known smoothed box
                if self.selected_last_seen_frame and (self.frame_count - self.selected_last_seen_frame) <= getattr(self, 'selected_hold_frames', 30):
                    return track
        return None
    
    def smooth_ptz(self, pan, tilt, zoom):
        self.last_pan = self.ptz_smooth * pan + (1-self.ptz_smooth) * self.last_pan
        self.last_tilt = self.ptz_smooth * tilt + (1-self.ptz_smooth) * self.last_tilt
        self.last_zoom = self.ptz_smooth * zoom + (1-self.ptz_smooth) * self.last_zoom
        return self.last_pan, self.last_tilt, self.last_zoom
    
    def calculate_ptz(self, track, w, h):
        if track is None:
            return 0.5, 0.5, 1.0
        x1, y1, x2, y2 = track.bbox
        pan = (x1 + x2) / 2 / w
        tilt = (y1 + y2) / 2 / h
        if self.auto_zoom:
            obj_w, obj_h = x2 - x1, y2 - y1
            zoom = min(w / max(obj_w, 1e-6), h / max(obj_h, 1e-6)) * 0.6
            zoom = np.clip(zoom, 0.5, 3.0)
        else:
            zoom = 1.0
        return self.smooth_ptz(pan, tilt, zoom)
    
    def draw_raw_detections(self, frame):
        for det in self.raw_detections:
            x1, y1 = int(det['xmin']), int(det['ymin'])
            x2, y2 = int(det['xmax']), int(det['ymax'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
            label = f"{det['name']}: {det['confidence']:.2f}"
            cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    
    def draw_tracks(self, frame):
        tracks_to_draw = list(self.active_tracks)
        # If selected track is not active but exists as lost within hold, include it for drawing
        if self.selected_track_id is not None and all(t.track_id != self.selected_track_id for t in tracks_to_draw):
            for t in self.tracker.tracks_lost:
                if t.track_id == self.selected_track_id and (self.frame_count - self.selected_last_seen_frame) <= getattr(self, 'selected_hold_frames', 30):
                    tracks_to_draw.append(t)
                    break
        for track in tracks_to_draw:
            if track.state != 'tracked':
                # draw only if it's the selected track within hold window
                if track.track_id != self.selected_track_id:
                    continue
            x1, y1, x2, y2 = map(int, track.bbox)
            if x1 >= x2 or y1 >= y2:
                continue
            is_selected = (track.track_id == self.selected_track_id)
            color = (0, 0, 255) if is_selected else (255, 204, 102)
            thickness = 3 if is_selected else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            label_text = f"ID:{track.track_id} {track.score:.2f}"
            font = cv2.FONT_HERSHEY_DUPLEX
            (tw, th), baseline = cv2.getTextSize(label_text, font, 0.6, 1)
            bg_x1 = x1
            bg_y2 = y1
            bg_y1 = max(0, bg_y2 - th - 8)
            bg_x2 = x1 + tw + 12
            overlay = frame.copy()
            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
            cv2.putText(frame, label_text, (bg_x1 + 6, bg_y2 - 6), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    def draw_exited_indicators(self, frame):
        h, w = frame.shape[:2]
        for tid, etime, bbox in self.exited_tracks:
            direction = self.get_exit_direction(bbox, w, h)
            cx = int((bbox[0] + bbox[2]) / 2)
            cy = int((bbox[1] + bbox[3]) / 2)
            timestamp_str = datetime.fromtimestamp(etime).strftime("%H:%M:%S")
            label = f"ID:{tid} @ {timestamp_str}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
            text_x = cx - tw // 2
            text_y = cy - 20 if direction != 'top' else cy + th + 20
            text_x = max(0, min(text_x, w - tw - 10))
            text_y = max(th + 5, min(text_y, h - 10))
            cv2.rectangle(frame, (text_x - 5, text_y - th - 5), (text_x + tw + 5, text_y + 5), (0, 0, 0), -1)
            cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            arrow_length = 15
            arrow_color = (0, 0, 255)
            if direction == 'left':
                start, end = (cx, cy), (cx - arrow_length, cy)
            elif direction == 'right':
                start, end = (cx, cy), (cx + arrow_length, cy)
            elif direction == 'top':
                start, end = (cx, cy), (cx, cy - arrow_length)
            else:
                start, end = (cx, cy), (cx, cy + arrow_length)
            cv2.arrowedLine(frame, start, end, arrow_color, 2, tipLength=0.3)
    
    def draw_status(self, frame):
        self.fps_count += 1
        if time.time() - self.fps_time > 1:
            self.current_fps = self.fps_count
            self.fps_count = 0
            self.fps_time = time.time()
        active_count = len([t for t in self.active_tracks if t.state == 'tracked'])
        lines = [
            f"FPS: {self.current_fps}",
            f"Active Tracks: {active_count}",
            f"Selected: {self.selected_track_id or 'None'}",
            f"Detect Every: {self.detect_interval} frames"
        ]
        if self.show_raw_detections:
            lines.append("RAW DETECTIONS: ON")
        total_lines = len(lines)
        status_height = 25 + (total_lines - 1) * 20
        cv2.rectangle(frame, (5, 5), (250, status_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (250, status_height), (0, 255, 0), 2)
        y = 25
        for line in lines:
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y += 20
        h, _ = frame.shape[:2]
        cv2.putText(frame, "Left Click: Select | Right Click: Clear | ESC: Quit",
                    (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def process_frame(self, frame):
        if not self.tracking_enabled:
            return frame
        self.detect_and_track(frame)
        selected_track = self.get_selected_track()
        if selected_track:
            h, w = frame.shape[:2]
            pan, tilt, zoom = self.calculate_ptz(selected_track, w, h)
            self.target_pan = pan
            self.target_tilt = tilt
            self.target_zoom = zoom
            self._smooth_update()
        if self.show_raw_detections:
            self.draw_raw_detections(frame)
        self.draw_tracks(frame)
        self.draw_exited_indicators(frame)
        return frame
    
    def run(self):
        if not self.connect():
            print("❌ Failed to connect to camera")
            return
        print("\n🚀 RF-DETR PTZ TRACKER RUNNING!")
        print("📌 Controls:")
        print("   - Left Click: Select Track")
        print("   - Right Click: Clear Selection")
        print("   - 'D' key: Toggle detection interval (5/10 frames)")
        print("   - 'R' key: Toggle raw detections (Yellow boxes)")
        print("   - ESC: Exit")
        print("-" * 50)
        try:
            while True:
                if not self.read_frame():
                    if self.is_video:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.frame_count = 0
                        continue
                    break
                frame = self.original_frame.copy()
                processed = self.process_frame(frame)
                cv2.imshow('TRACKING VIEW', processed)
                ptz_frame = self.get_ptz_frame()
                if ptz_frame is not None:
                    selected_track = self.get_selected_track()
                    if selected_track:
                        info = f"Tracking ID: {selected_track.track_id}"
                        font = cv2.FONT_HERSHEY_DUPLEX
                        (tw, th), base = cv2.getTextSize(info, font, 0.7, 1)
                        ox1, oy1 = 10, 10
                        ox2, oy2 = 10 + tw + 12, 10 + th + 12
                        overlay = ptz_frame.copy()
                        cv2.rectangle(overlay, (ox1, oy1), (ox2, oy2), (0, 0, 0), -1)
                        cv2.addWeighted(overlay, 0.4, ptz_frame, 0.6, 0, ptz_frame)
                        cv2.putText(ptz_frame, info, (ox1 + 6, oy2 - 6), font, 0.7, (255, 204, 102), 1, cv2.LINE_AA)
                    cv2.imshow('PTZ VIEW', ptz_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                elif key == ord('c') or key == ord('C'):
                    self.clear_selection()
                elif key == ord('d') or key == ord('D'):
                    self.detect_interval = 10 if self.detect_interval == 5 else 5
                    print(f"⚙️ Detection interval changed to: {self.detect_interval} frames")
                elif key == ord('r') or key == ord('R'):
                    self.show_raw_detections = not self.show_raw_detections
                    print(f"🔍 Raw detections: {'ON (Yellow boxes)' if self.show_raw_detections else 'OFF'}")
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            cv2.destroyAllWindows()
            self.disconnect()
            print("\n👋 Tracker stopped")


def main():
    print("\n" + "="*60)
    print("🚀 RF-DETR PTZ TRACKER")
    print("="*60)
    print("\nSource Selection:")
    print("1. Webcam (0)")
    print("2. Video File")
    print("3. IP Camera (RTSP)")
    choice = input("\nSelect source (1-3, default=1): ").strip() or "1"
    if choice == "2":
        path = input("Enter video path: ").strip().strip('"')
        source = path if os.path.isfile(path) else 0
        if source == 0:
            print("❌ File not found, using webcam")
    elif choice == "3":
        source = input("Enter RTSP URL: ").strip()
    else:
        source = 0
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        try:
            print(f"\n✅ GPU Available: {torch.cuda.get_device_name(0)}")
        except Exception:
            print("\n✅ GPU Available")
    else:
        print("\n⚠️ GPU Not Available - Using CPU")
    interval = input("\nDetection interval (frames, default=5): ").strip()
    detect_interval = int(interval) if interval.isdigit() else 5
    tracker = OptimizedPTZTracker(
        camera_source=source,
        target_class='person',
        detect_interval=detect_interval,
        use_gpu=use_gpu,
        model_variant='nano',
        inference_size=640,
        optimize=False
    )
    tracker.run()


if __name__ == '__main__':
    main()


