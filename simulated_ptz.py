# -*- coding: utf-8 -*-
# Simulated PTZ Control for Non-PTZ Cameras
# This script simulates Pan-Tilt-Zoom operations on regular cameras using digital cropping and scaling

import cv2
import numpy as np
from threading import Thread
import time


class SimulatedPTZ:
    """
    Simulates PTZ operations on a non-PTZ camera using digital zoom and pan/tilt
    """
    
    def __init__(self, camera_source=0, output_width=1280, output_height=720):
        """
        Initialize the simulated PTZ camera
        
        Args:
            camera_source: Camera index (0 for default webcam), RTSP URL, or video file path
            output_width: Width of the output display
            output_height: Height of the output display
        """
        self.camera_source = camera_source
        self.output_width = output_width
        self.output_height = output_height
        
        # Video capture object
        self.cap = None
        self.frame = None
        self.original_frame = None
        self.running = False
        
        # PTZ parameters
        self.zoom_level = 1.0  # 1.0 = no zoom, 3.0 = 3x zoom
        self.min_zoom = 1.0
        self.max_zoom = 5.0
        self.zoom_step = 0.1
        
        # Pan and tilt (center position as percentage of frame: 0.5, 0.5 = center)
        self.pan_position = 0.5  # 0.0 (left) to 1.0 (right)
        self.tilt_position = 0.5  # 0.0 (top) to 1.0 (bottom)
        self.pan_step = 0.02
        self.tilt_step = 0.02
        
        # Smooth movement parameters
        self.target_zoom = 1.0
        self.target_pan = 0.5
        self.target_tilt = 0.5
        self.smooth_factor = 0.2  # Lower = smoother but slower
        
        # Camera properties
        self.frame_width = 0
        self.frame_height = 0
        
    def connect(self):
        """Connect to the camera"""
        print(f"Connecting to camera source: {self.camera_source}")
        self.cap = cv2.VideoCapture(self.camera_source)
        
        if not self.cap.isOpened():
            raise Exception(f"Failed to open camera source: {self.camera_source}")
        
        # Get camera resolution
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Camera connected: {self.frame_width}x{self.frame_height}")
        print(f"Output resolution: {self.output_width}x{self.output_height}")
        
        self.running = True
        return True
    
    def disconnect(self):
        """Disconnect from the camera"""
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Camera disconnected")
    
    def _smooth_update(self):
        """Smoothly interpolate PTZ parameters towards target values"""
        self.zoom_level += (self.target_zoom - self.zoom_level) * self.smooth_factor
        self.pan_position += (self.target_pan - self.pan_position) * self.smooth_factor
        self.tilt_position += (self.target_tilt - self.tilt_position) * self.smooth_factor
    
    def zoom_in(self, amount=None):
        """Zoom in (digital zoom)"""
        if amount is None:
            amount = self.zoom_step
        self.target_zoom = min(self.max_zoom, self.target_zoom + amount)
        print(f"Zoom in -> Target zoom: {self.target_zoom:.2f}x")
    
    def zoom_out(self, amount=None):
        """Zoom out"""
        if amount is None:
            amount = self.zoom_step
        self.target_zoom = max(self.min_zoom, self.target_zoom - amount)
        print(f"Zoom out -> Target zoom: {self.target_zoom:.2f}x")
    
    def pan_left(self, amount=None):
        """Pan left"""
        if amount is None:
            amount = self.pan_step
        self.target_pan = max(0.0, self.target_pan - amount)
        print(f"Pan left -> Target pan: {self.target_pan:.2f}")
    
    def pan_right(self, amount=None):
        """Pan right"""
        if amount is None:
            amount = self.pan_step
        self.target_pan = min(1.0, self.target_pan + amount)
        print(f"Pan right -> Target pan: {self.target_pan:.2f}")
    
    def tilt_up(self, amount=None):
        """Tilt up"""
        if amount is None:
            amount = self.tilt_step
        self.target_tilt = max(0.0, self.target_tilt - amount)
        print(f"Tilt up -> Target tilt: {self.target_tilt:.2f}")
    
    def tilt_down(self, amount=None):
        """Tilt down"""
        if amount is None:
            amount = self.tilt_step
        self.target_tilt = min(1.0, self.target_tilt + amount)
        print(f"Tilt down -> Target tilt: {self.target_tilt:.2f}")
    
    def reset_position(self):
        """Reset to center position with no zoom"""
        self.target_zoom = 1.0
        self.target_pan = 0.5
        self.target_tilt = 0.5
        print("Reset to center position")
    
    def set_position(self, pan, tilt, zoom):
        """
        Set PTZ position directly
        
        Args:
            pan: Pan position (0.0 to 1.0)
            tilt: Tilt position (0.0 to 1.0)
            zoom: Zoom level (min_zoom to max_zoom)
        """
        self.target_pan = np.clip(pan, 0.0, 1.0)
        self.target_tilt = np.clip(tilt, 0.0, 1.0)
        self.target_zoom = np.clip(zoom, self.min_zoom, self.max_zoom)
    
    def get_ptz_frame(self):
        """
        Apply simulated PTZ to the current frame
        
        Returns:
            Cropped and scaled frame simulating PTZ
        """
        if self.original_frame is None:
            return None
        
        # Smooth update of parameters
        self._smooth_update()
        
        frame = self.original_frame.copy()
        h, w = frame.shape[:2]
        
        # Calculate crop dimensions based on zoom level
        crop_width = int(w / self.zoom_level)
        crop_height = int(h / self.zoom_level)
        
        # Calculate crop center based on pan/tilt
        center_x = int(self.pan_position * w)
        center_y = int(self.tilt_position * h)
        
        # Calculate crop boundaries
        x1 = max(0, center_x - crop_width // 2)
        y1 = max(0, center_y - crop_height // 2)
        x2 = min(w, x1 + crop_width)
        y2 = min(h, y1 + crop_height)
        
        # Adjust if crop goes out of bounds
        if x2 - x1 < crop_width:
            x1 = max(0, x2 - crop_width)
        if y2 - y1 < crop_height:
            y1 = max(0, y2 - crop_height)
        
        # Crop the frame
        cropped = frame[y1:y2, x1:x2]
        
        # Resize to output dimensions
        if cropped.size > 0:
            result = cv2.resize(cropped, (self.output_width, self.output_height))
            
            # Add PTZ information overlay
            self._add_info_overlay(result)
            
            return result
        else:
            return frame
    
    def _add_info_overlay(self, frame):
        """Add PTZ information overlay to the frame"""
        info_text = [
            f"Zoom: {self.zoom_level:.2f}x",
            f"Pan: {self.pan_position:.2f}",
            f"Tilt: {self.tilt_position:.2f}",
            "",
            "Controls:",
            "W/S: Tilt Up/Down",
            "A/D: Pan Left/Right",
            "Q/E: Zoom Out/In",
            "R: Reset",
            "ESC: Exit"
        ]
        
        y_offset = 30
        for i, text in enumerate(info_text):
            cv2.putText(frame, text, (10, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def read_frame(self):
        """Read a frame from the camera"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.original_frame = frame
                return True
        return False
    
    def run_interactive(self):
        """
        Run interactive PTZ control with keyboard
        """
        if not self.connect():
            return
        
        print("\n" + "="*60)
        print("Simulated PTZ Camera Control")
        print("="*60)
        print("\nKeyboard Controls:")
        print("  W/S - Tilt Up/Down")
        print("  A/D - Pan Left/Right")
        print("  Q/E - Zoom Out/In")
        print("  R   - Reset to center")
        print("  ESC - Exit")
        print("="*60 + "\n")
        
        try:
            while self.running:
                if not self.read_frame():
                    print("Failed to read frame")
                    break
                
                # Get PTZ frame
                ptz_frame = self.get_ptz_frame()
                
                if ptz_frame is not None:
                    cv2.imshow('Simulated PTZ Camera', ptz_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    break
                elif key == ord('w') or key == ord('W'):
                    self.tilt_up()
                elif key == ord('s') or key == ord('S'):
                    self.tilt_down()
                elif key == ord('a') or key == ord('A'):
                    self.pan_left()
                elif key == ord('d') or key == ord('D'):
                    self.pan_right()
                elif key == ord('q') or key == ord('Q'):
                    self.zoom_out()
                elif key == ord('e') or key == ord('E'):
                    self.zoom_in()
                elif key == ord('r') or key == ord('R'):
                    self.reset_position()
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.disconnect()


def main():
    """
    Main function to run simulated PTZ
    
    Modify the camera_source parameter to use different video sources:
    - 0, 1, 2... for USB/webcam cameras
    - "rtsp://..." for IP cameras (RTSP stream)
    - "path/to/video.mp4" for video files
    """
    
    # Example configurations (uncomment the one you want to use):
    
    # Option 1: Default webcam
    camera_source = 0
    
    # Option 2: RTSP camera (like the one in Awais.py)
    # camera_source = "rtsp://192.168.100.67:554/stream"
    
    # Option 3: Video file
    # camera_source = "3.mp4"
    
    # Create and run simulated PTZ camera
    ptz = SimulatedPTZ(
        camera_source=camera_source,
        output_width=1280,
        output_height=720
    )
    
    ptz.run_interactive()


if __name__ == '__main__':
    main()


