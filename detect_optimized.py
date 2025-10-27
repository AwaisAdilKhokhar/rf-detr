import os
import cv2
import time
import warnings
from pathlib import Path
from typing import Union

import supervision as sv
from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES

# Suppress PyTorch tracing warnings (they're expected during optimization)
warnings.filterwarnings('ignore', category=UserWarning, module='torch.jit')
warnings.filterwarnings('ignore', message='.*TracerWarning.*')
warnings.filterwarnings('ignore', message='.*trace.*')
os.environ['TORCH_JIT_LOG_LEVEL'] = 'ERROR'


def coerce_video_reference(src: str) -> Union[int, str]:
    """Accept numeric strings like '0' or '1' as webcam indices, or resolve file paths."""
    if src.isdigit():
        return int(src)
    
    # If it's a file path, convert to absolute path
    if os.path.exists(src):
        return str(Path(src).resolve())
    
    # Otherwise return as-is (could be RTSP URL or invalid path)
    return src


def get_model_class(model_choice: str):
    """Return the appropriate RF-DETR model class."""
    model_map = {
        "1": RFDETRNano,
        "2": RFDETRSmall,
        "3": RFDETRMedium,
        "4": RFDETRBase,
    }
    return model_map.get(model_choice, RFDETRNano)


def get_user_input() -> dict:
    """Interactively prompt user for configuration."""
    print("\n" + "=" * 60)
    print("  RF-DETR Real-Time Object Detection (OPTIMIZED)")
    print("=" * 60)
    
    # 1. Video source
    print("\n[1/4] Select video source:")
    print("  1 - Webcam (default)")
    print("  2 - Video file")
    print("  3 - RTSP/RTMP stream")
    
    source_choice = input("\nEnter choice (1-3) [default: 1]: ").strip() or "1"
    
    if source_choice == "1":
        webcam_index = input("Enter webcam index [default: 0]: ").strip() or "0"
        video_source = webcam_index
    elif source_choice == "2":
        video_source = input("Enter path to video file: ").strip()
        if not video_source:
            print("⚠️  No path provided, using webcam 0")
            video_source = "0"
        elif not os.path.exists(video_source):
            print(f"⚠️  File not found: {video_source}")
            print("    Please check the path and try again.")
            exit(1)
    elif source_choice == "3":
        video_source = input("Enter RTSP/RTMP URL: ").strip()
        if not video_source:
            print("⚠️  No URL provided, using webcam 0")
            video_source = "0"
    else:
        print("⚠️  Invalid choice, using webcam 0")
        video_source = "0"
    
    # 2. Model selection
    print("\n[2/4] Select RF-DETR model:")
    print("  1 - rfdetr-nano (fastest, recommended)")
    print("  2 - rfdetr-small")
    print("  3 - rfdetr-medium")
    print("  4 - rfdetr-base")
    
    model_choice = input("\nEnter choice (1-4) [default: 1]: ").strip() or "1"
    
    # 3. Confidence threshold
    print("\n[3/4] Set confidence threshold (0.0 - 1.0):")
    threshold_input = input("Enter threshold [default: 0.3]: ").strip()
    
    try:
        threshold = float(threshold_input) if threshold_input else 0.3
        threshold = max(0.0, min(1.0, threshold))  # Clamp to [0, 1]
    except ValueError:
        print("⚠️  Invalid threshold, using 0.3")
        threshold = 0.3
    
    # 4. Optimization mode
    print("\n[4/4] Enable optimize_for_inference()? (May give 2x speedup)")
    print("  1 - Yes (experimental, may fail on some systems)")
    print("  2 - No (recommended, already fast with native PyTorch)")
    
    optimize_choice = input("\nEnter choice (1-2) [default: 2]: ").strip() or "2"
    optimize = optimize_choice == "1"
    
    # Summary
    print("\nConfiguration summary:")
    print(f"  • Video source: {video_source}")
    print(f"  • Model: {['rfdetr-nano', 'rfdetr-small', 'rfdetr-medium', 'rfdetr-base'][int(model_choice)-1]}")
    print(f"  • Confidence threshold: {threshold}")
    print(f"  • TorchScript optimization: {'Enabled (experimental)' if optimize else 'Disabled (native PyTorch)'}")
    
    confirm = input("\nStart detection? (y/n) [default: y]: ").strip().lower()
    if confirm and confirm != 'y':
        print("❌ Cancelled by user")
        exit(0)
    
    return {
        "video_source": video_source,
        "model_choice": model_choice,
        "threshold": threshold,
        "optimize": optimize,
    }


def main() -> None:
    """Main entry point with native RF-DETR inference."""
    config = get_user_input()
    
    video_reference = coerce_video_reference(config["video_source"])
    
    print("\n" + "=" * 60)
    print("🚀 Loading RF-DETR model...")
    print("=" * 60)
    
    # Load model using native RF-DETR
    ModelClass = get_model_class(config["model_choice"])
    model = ModelClass()
    
    # CRITICAL: Call optimize_for_inference() for up to 2x speedup!
    if config["optimize"]:
        print("⚡ Optimizing model for inference...")
        print("   (Compiling with TorchScript - this takes ~10 seconds)")
        
        # Temporarily suppress warnings during optimization
        import sys
        from io import StringIO
        old_stderr = sys.stderr
        sys.stderr = StringIO()
        
        try:
            model.optimize_for_inference()
            sys.stderr = old_stderr
            print("✓ Model optimized for 2x speedup!")
        except Exception as e:
            sys.stderr = old_stderr
            print(f"⚠️  Optimization failed: {str(e)[:80]}")
            print("   Continuing without optimization (still fast!)")
            # Model will still work, just without TorchScript optimization
    
    # Open video source
    cap = cv2.VideoCapture(video_reference)
    
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video source: {video_reference}")
        exit(1)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n📹 Video: {width}x{height} @ {fps} FPS")
    print(f"🎯 Threshold: {config['threshold']}")
    print("\n" + "=" * 60)
    print("▶️  Processing... Press 'q' to quit")
    print("=" * 60 + "\n")
    
    # Initialize annotators
    box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.ROBOFLOW)
    label_annotator = sv.LabelAnnotator(color=sv.ColorPalette.ROBOFLOW)
    
    # FPS tracking
    frame_count = 0
    start_time = time.time()
    fps_display = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\n✓ End of video or stream interrupted")
                break
            
            # Convert BGR to RGB for RF-DETR
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run inference with native RF-DETR predict()
            detections = model.predict(frame_rgb, threshold=config["threshold"])
            
            # Create labels
            labels = [
                f"{COCO_CLASSES[class_id]} {confidence:.2f}"
                for class_id, confidence in zip(detections.class_id, detections.confidence)
            ]
            
            # Annotate frame
            annotated_frame = frame.copy()
            annotated_frame = box_annotator.annotate(annotated_frame, detections)
            annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count % 30 == 0:  # Update FPS every 30 frames
                elapsed = time.time() - start_time
                fps_display = frame_count / elapsed
            
            # Add FPS text to frame
            cv2.putText(
                annotated_frame,
                f"FPS: {fps_display:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            
            # Display
            cv2.imshow("RF-DETR Detection", annotated_frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n✓ Detection stopped by user")
                break
                
    except KeyboardInterrupt:
        print("\n\n✓ Detection stopped by user (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting tips:")
        print("  • Check that the video source is valid")
        print("  • Ensure your webcam isn't being used by another app")
        print("  • Try a different webcam index (0, 1, 2...)")
        print("  • For video files, verify the path is correct")
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final stats
        if frame_count > 0:
            elapsed = time.time() - start_time
            avg_fps = frame_count / elapsed
            print(f"\n📊 Statistics:")
            print(f"  • Frames processed: {frame_count}")
            print(f"  • Total time: {elapsed:.1f}s")
            print(f"  • Average FPS: {avg_fps:.1f}")


if __name__ == "__main__":
    main()

