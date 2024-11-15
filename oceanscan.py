import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from typing import List, Dict, Tuple
import time

class OceanScanner:
    def __init__(self, fish_model_path: str, trash_model_path: str):
        """
        Initialize OceanScanner with YOLOv8 models and DeepSORT tracker
        
        Args:
            fish_model_path: Path to YOLOv8 fish detection model
            trash_model_path: Path to YOLOv8 trash detection model
        """
        # Initialize YOLOv8 models
        self.fish_model = YOLO(fish_model_path)
        self.trash_model = YOLO(trash_model_path)
        
        # Initialize separate DeepSORT trackers for fish and trash
        self.fish_tracker = DeepSort(max_age=30)
        self.trash_tracker = DeepSort(max_age=50)
        
        # Define class lists
        self.trash_classes = ['Mask', 'can', 'cellphone', 'electronics', 'gbottle', 
                            'glove', 'metal', 'misc', 'net', 'pbag', 'pbottle', 
                            'plastic', 'rod', 'sunglasses', 'tire']
        self.fish_classes = ['fish', 'null']
        
        # Initialize statistics
        self.reset_stats()

    def reset_stats(self):
        """Reset all tracking statistics"""
        self.stats = {
            'fish_count': 0,
            'trash_count': 0,
            'trash_by_type': {cls: 0 for cls in self.trash_classes},
            'unique_fish': set(),
            'unique_trash': set(),
            'fps': 0
        }

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Process a single frame with both models and tracking"""
        start_time = time.time()
        
        # Run both models on the frame
        fish_results = self.fish_model(frame)[0]
        trash_results = self.trash_model(frame)[0]
        
        # Process fish detections
        fish_detections = []
        for detection in fish_results.boxes.data:
            x1, y1, x2, y2, conf, cls = detection
            if conf > 0.5:
                fish_detections.append(([x1, y1, x2, y2], conf, 'fish'))
        
        # Process trash detections
        trash_detections = []
        for detection in trash_results.boxes.data:
            x1, y1, x2, y2, conf, cls = detection
            if conf > 0.5:
                class_name = self.trash_classes[int(cls)]
                trash_detections.append(([x1, y1, x2, y2], conf, class_name))
        
        # Update trackers
        fish_tracks = self.fish_tracker.update_tracks(fish_detections, frame=frame)
        trash_tracks = self.trash_tracker.update_tracks(trash_detections, frame=frame)
        
        # Draw tracks and update statistics
        frame = self._draw_tracks(frame, fish_tracks, trash_tracks)
        self._update_statistics(fish_tracks, trash_tracks)
        
        # Calculate FPS
        self.stats['fps'] = 1.0 / (time.time() - start_time)
        
        return frame, self.stats
    
    def _draw_tracks(self, frame: np.ndarray, fish_tracks: List, trash_tracks: List) -> np.ndarray:
        """Draw bounding boxes and tracking information on frame"""
        # Draw fish tracks in green
        for track in fish_tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb()
            
            cv2.rectangle(frame, 
                         (int(ltrb[0]), int(ltrb[1])), 
                         (int(ltrb[2]), int(ltrb[3])), 
                         (0, 255, 0), 2)
            cv2.putText(frame, f'Fish #{track_id}', 
                       (int(ltrb[0]), int(ltrb[1] - 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Add to unique fish set
            self.stats['unique_fish'].add(track_id)
        
        # Draw trash tracks in red with class labels
        for track in trash_tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            ltrb = track.to_ltrb()
            class_name = track.get_det_class()
            
            cv2.rectangle(frame,
                         (int(ltrb[0]), int(ltrb[1])),
                         (int(ltrb[2]), int(ltrb[3])),
                         (0, 0, 255), 2)
            cv2.putText(frame, f'{class_name} #{track_id}',
                       (int(ltrb[0]), int(ltrb[1] - 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Add to unique trash set
            self.stats['unique_trash'].add(track_id)
        
        return frame
    
    def _update_statistics(self, fish_tracks: List, trash_tracks: List):
        """Update detection and tracking statistics"""
        self.stats['fish_count'] = len([t for t in fish_tracks if t.is_confirmed()])
        self.stats['trash_count'] = len([t for t in trash_tracks if t.is_confirmed()])
        
        trash_by_type = {cls: 0 for cls in self.trash_classes}
        for track in trash_tracks:
            if track.is_confirmed():
                class_name = track.get_det_class()
                trash_by_type[class_name] += 1
        self.stats['trash_by_type'] = trash_by_type

def process_video(video_path: str, fish_model_path: str, trash_model_path: str, output_path: str = None):
    """
    Process a video file with the OceanScanner
    
    Args:
        video_path: Path to input video
        fish_model_path: Path to YOLOv8 fish detection model
        trash_model_path: Path to YOLOv8 trash detection model
        output_path: Optional path to save processed video
    """
    # Initialize scanner
    scanner = OceanScanner(fish_model_path, trash_model_path)
    
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize video writer if output path is provided
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        processed_frame, stats = scanner.process_frame(frame)
        
        # Draw statistics overlay
        cv2.putText(processed_frame, f"FPS: {stats['fps']:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(processed_frame, f"Fish: {stats['fish_count']} (Total: {len(stats['unique_fish'])})",
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(processed_frame, f"Trash: {stats['trash_count']} (Total: {len(stats['unique_trash'])})",
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Display progress
        frame_count += 1
        progress = (frame_count / total_frames) * 100
        cv2.putText(processed_frame, f"Progress: {progress:.1f}%", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Display frame
        cv2.imshow('OceanScan', processed_frame)
        
        # Write frame if output path is provided
        if output_path:
            out.write(processed_frame)
        
        # Break if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()
    
    # Print final statistics
    print("\nFinal Statistics:")
    print(f"Total unique fish tracked: {len(stats['unique_fish'])}")
    print(f"Total unique trash items tracked: {len(stats['unique_trash'])}")
    print("\nTrash breakdown:")
    for trash_type, count in stats['trash_by_type'].items():
        if count > 0:
            print(f"{trash_type}: {count}")

# Example usage
if __name__ == "__main__":
    process_video(
        video_path='istockphoto-498319211-640_adpp_is.mp4',
        fish_model_path='fish_best.pt',
        trash_model_path='/Users/armaanthakur/Desktop/MV/trash_best.pt',
        output_path='processed_video.mp4'
    )