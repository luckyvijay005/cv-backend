"""
CLI entry point for classroom analytics pipeline.

Usage:
    python run.py --video training_footages.mp4 --output data/outputs/report.json --max_frames 5000
    python run.py --video training_footages.mp4 --output data/outputs/report.json --visualize
"""
import argparse
import sys
from datetime import datetime
from pipeline import ClassroomAnalyticsPipeline
from utils import save_report


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="Classroom Analytics Pipeline")
    parser.add_argument('--video', '--video_path', type=str, required=True, dest='video',
                        help='Path to input video file')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument(
        '--output', '--output_path', '--output_dir', type=str, dest='output',
        default=f'data/outputs/report_{timestamp}.json', help='Path to output report JSON')
    parser.add_argument('--sample_fps', type=float, default=None,
                        help='Target frame rate for processing (e.g. 5.0, 10.0, or None for native FPS)')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Max frames to process (for testing)')
    parser.add_argument('--proximity', type=float, default=80.0,
                        help='Proximity threshold in pixels (~1.5m)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='YOLOv8 model variant (yolov8n, yolov8s, yolov8m, etc.)')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU device index (0=first GPU, -1=CPU)')
    parser.add_argument('--student_registry', type=str, default='data/student_registry.json',
                        help='Path to persistent student registry JSON')
    parser.add_argument('--teacher_zone', type=int, nargs=4, default=None,
                        help='Optional manual teacher zone: x1 y1 x2 y2')
    parser.add_argument('--visualize', action='store_true',
                        help='Save visualization frames')

    args = parser.parse_args()

    # If output points to a directory, save report.json inside it
    output_path = args.output
    if os.path.isdir(output_path) or not output_path.endswith('.json'):
        os.makedirs(output_path, exist_ok=True)
        output_path = os.path.join(output_path, f"report_{timestamp}.json")

    try:
        print(f"Initializing pipeline...", flush=True)
        pipeline = ClassroomAnalyticsPipeline(
            model_name=args.model,
            device=args.device,
            proximity_threshold_px=args.proximity,
            student_registry_path=args.student_registry,
            teacher_zone=tuple(
                args.teacher_zone) if args.teacher_zone else None,
        )

        print(f"Running analysis on: {args.video}", flush=True)
        report = pipeline.run(
            args.video,
            max_frames=args.max_frames,
            sample_fps=args.sample_fps,
            save_viz=args.visualize
        )

        # Save report
        save_report(report, output_path)

        # Print summary
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"Total students detected: {report['total_students']}")
        print(f"Video duration: {report['duration_seconds']:.1f} seconds")
        print(f"\nTop 3 Least Interactive Students:")
        for i, student in enumerate(report['top_3_least_interactive'], 1):
            print(f"  {i}. Student ID {student['track_id']}")
            print(f"     Interaction Score: {student['score']:.2f}")
            print(f"     Isolated: {student['isolated_minutes']:.1f} min")
            print(f"     With Peers: {student['near_peer_minutes']:.1f} min")
            print(f"     Confidence: {student['confidence']:.2f}")
        print("="*60)
        print(f"Report saved to: {args.output}")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
