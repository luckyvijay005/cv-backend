"""
Flask API server for classroom analytics pipeline.
Allows Electron frontend to communicate with Python backend.

Endpoints:
  GET  /api/status            - health check
  POST /api/analyze           - multipart video upload (original, kept for compatibility)
  POST /api/analyze/path      - JSON {video_path, max_frames, proximity_threshold}
  GET  /api/analyze/stream    - SSE stream for the last /api/analyze/path call
  GET  /api/export/<format>   - download pdf or csv
  POST /api/feedback          - save teacher feedback
"""

import os
import json
import queue
import threading
import tempfile
from pathlib import Path
from datetime import datetime

from flask import Flask, request, jsonify, send_file, Response, stream_with_context
from flask_cors import CORS

from pipeline import ClassroomAnalyticsPipeline


app = Flask(__name__)
CORS(app, supports_credentials=True)

# ─── Globals ──────────────────────────────────────────────────────────────────
pipeline_lock = threading.Lock()
pipeline_instance = None
current_report = None          # Cache latest report for export endpoints
analysis_queue: queue.Queue = None  # SSE progress queue for the current analysis
cancel_event = threading.Event()    # Set to request cancellation of current job


# ─── Pipeline factory ─────────────────────────────────────────────────────────
def get_pipeline(device=0, model_name='yolov8n.pt'):
    """Return the singleton pipeline, initialising it on first call."""
    global pipeline_instance
    with pipeline_lock:
        if pipeline_instance is None:
            print(f"[API] Initialising pipeline ({model_name}, device={device})…")
            pipeline_instance = ClassroomAnalyticsPipeline(
                device=device, model_name=model_name)
            print("[API] ✓ Pipeline ready")
    return pipeline_instance


# ─── SSE helper ───────────────────────────────────────────────────────────────
def _sse_event(data: dict) -> str:
    """Format a dict as a Server-Sent Event string."""
    return f"data: {json.dumps(data)}\n\n"


# ─── Progress-aware pipeline runner ───────────────────────────────────────────
def _run_analysis_with_progress(video_path: str, max_frames, proximity_threshold, sample_fps,
                                 progress_q: queue.Queue, cancel_ev: threading.Event):
    """
    Run the full pipeline in a background thread and push progress events to
    *progress_q*.  Events are dicts:
        { "type": "progress",   "stage": str, "percent": int }
        { "type": "complete",   "report": dict }
        { "type": "cancelled",  "message": str }
        { "type": "error",      "message": str }
    """
    global current_report
    from concurrent.futures import CancelledError

    def emit(stage: str, percent: int, detail: str = ""):
        progress_q.put({"type": "progress", "stage": stage, "percent": percent, "detail": detail})

    try:
        emit("Initialising pipeline", 5, "Loading AI models...")
        p = get_pipeline()

        if cancel_ev.is_set():
            progress_q.put({"type": "cancelled", "message": "Cancelled before loading video"})
            return

        kwargs = {
            'video_path': video_path,
            'max_frames': max_frames,
            'sample_fps': sample_fps,
            'save_viz': False,
            'cancel_callback': cancel_ev.is_set,   # pipeline checks this every 15 frames
            'progress_callback': emit,             # live frame progress
        }
        if proximity_threshold is not None:
            p.proximity_threshold_px = proximity_threshold

        report = p.run(**kwargs)

        # If we got here but cancel was set just after pipeline finished, still
        # honour the cancel so the user isn't confused.
        if cancel_ev.is_set():
            progress_q.put({"type": "cancelled", "message": "Analysis cancelled"})
            return

        emit("Generating report", 96, "Finalizing report...")

        # Attach report_id and save to disk
        report['report_id'] = datetime.now().isoformat()
        output_dir = Path(__file__).parent / 'data' / 'outputs'
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"report_{timestamp}.json"
        with open(report_path, 'w') as fh:
            json.dump(report, fh, indent=2)
        report['saved_report_path'] = str(report_path).replace('\\', '/')
        report['output_dir'] = str(output_dir).replace('\\', '/')

        current_report = report
        progress_q.put({"type": "complete", "report": report})
        print(f"[API] ✓ Analysis complete — {report.get('total_students', 0)} students")

    except CancelledError as ce:
        print(f"[API] ⚠ Analysis cancelled: {ce}")
        progress_q.put({"type": "cancelled", "message": str(ce)})

    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        print(f"[API] ✗ Analysis error:\n{tb}")
        progress_q.put({"type": "error", "message": str(exc)})


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get('/api/status')
def status():
    """Health check — polled by Electron until the server is ready."""
    return jsonify({
        'status': 'ready',
        'timestamp': datetime.now().isoformat(),
        'pipeline_initialised': pipeline_instance is not None,
    })


# ── Path-based analysis (recommended for large local files) ───────────────────

@app.post('/api/analyze/path')
def analyze_by_path():
    """
    Start analysis given a local file path.
    Resets the cancel flag so a fresh analysis can begin.
    """
    global analysis_queue, cancel_event

    data = request.get_json(force=True, silent=True) or {}
    video_path = data.get('video_path', '').strip()

    if not video_path:
        return jsonify({'error': 'video_path is required'}), 400
    if not os.path.isfile(video_path):
        return jsonify({'error': f'File not found: {video_path}'}), 400

    max_frames = data.get('max_frames') or None
    proximity_threshold = data.get('proximity_threshold') or None
    sample_fps = data.get('sample_fps') or None
    if sample_fps is not None:
        try:
            sample_fps = float(sample_fps)
        except (ValueError, TypeError):
            sample_fps = None

    # Clear any previous cancel request before starting a new job
    cancel_event.clear()

    # Create a fresh queue for this job
    job_q: queue.Queue = queue.Queue()
    analysis_queue = job_q

    # Kick off background thread
    t = threading.Thread(
        target=_run_analysis_with_progress,
        args=(video_path, max_frames, proximity_threshold, sample_fps, job_q, cancel_event),
        daemon=True,
    )
    t.start()

    return jsonify({'status': 'started', 'sse_url': '/api/analyze/stream'})


@app.get('/api/analyze/stream')
def analyze_stream():
    """
    SSE endpoint — streams progress events for the current analysis job.

    Events:
        data: {"type": "progress",  "stage": "...", "percent": 45}
        data: {"type": "complete",  "report": {...}}
        data: {"type": "cancelled", "message": "..."}
        data: {"type": "error",     "message": "..."}
    """
    global analysis_queue, cancel_event

    if analysis_queue is None:
        def no_job():
            yield _sse_event({"type": "error", "message": "No analysis job in progress."})
        return Response(no_job(), mimetype='text/event-stream')

    job_q = analysis_queue

    def generate():
        while True:
            # Also wake up when the cancel event fires
            if cancel_event.is_set():
                # Drain any pending item first
                try:
                    event = job_q.get_nowait()
                    yield _sse_event(event)
                    if event.get('type') in ('complete', 'error', 'cancelled'):
                        break
                except queue.Empty:
                    pass

            try:
                event = job_q.get(timeout=1)   # 1-second timeout so we check cancel_event regularly
                yield _sse_event(event)
                if event.get('type') in ('complete', 'error', 'cancelled'):
                    break
            except queue.Empty:
                # Send a heartbeat comment to keep the connection alive
                yield ": heartbeat\n\n"

    response = Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
    )
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'
    return response


@app.post('/api/analyze/cancel')
def cancel_analysis():
    """
    Request cancellation of the currently running analysis.
    The pipeline will stop at the next 50-frame checkpoint.
    """
    global cancel_event
    cancel_event.set()
    print("[API] ⚠ Cancellation requested")
    return jsonify({'status': 'cancelling'})


# ── Multipart upload (kept for compatibility) ──────────────────────────────────

@app.post('/api/analyze')
def analyze():
    """
    Analyse a video file uploaded as multipart form data.
    Kept for backwards compatibility; prefer /api/analyze/path for desktop use.
    """
    global current_report

    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No video file selected'}), 400

    max_frames = request.form.get('max_frames', type=int)
    proximity_threshold = request.form.get('proximity_threshold', type=float)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        video_file.save(tmp.name)
        temp_video_path = tmp.name

    try:
        p = get_pipeline()
        if proximity_threshold:
            p.proximity_threshold_px = proximity_threshold

        print(f"[API] Analysing uploaded file: {video_file.filename}")
        report = p.run(video_path=temp_video_path, max_frames=max_frames, save_viz=False)

        report['report_id'] = datetime.now().isoformat()

        # Save to disk
        output_dir = Path(__file__).parent / 'data' / 'outputs'
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"report_{timestamp}.json"
        with open(report_path, 'w') as fh:
            json.dump(report, fh, indent=2)
        report['saved_report_path'] = str(report_path).replace('\\', '/')
        report['output_dir'] = str(output_dir).replace('\\', '/')

        current_report = report
        print(f"[API] ✓ Analysis complete: {len(report.get('all_metrics', {}))} students detected")
        return jsonify(report)

    except Exception as e:
        print(f"[API] ✗ Analysis error: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

    finally:
        try:
            os.unlink(temp_video_path)
        except Exception:
            pass


# ── Export ────────────────────────────────────────────────────────────────────

@app.get('/api/export/<fmt>')
def export_report(fmt):
    """Export the latest report as PDF or CSV."""
    if current_report is None:
        return jsonify({'error': 'No report loaded. Analyse a video first.'}), 400

    if fmt not in ('pdf', 'csv'):
        return jsonify({'error': 'Format must be pdf or csv'}), 400

    try:
        if fmt == 'pdf':
            from export import generate_pdf_report
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                output_path = tmp.name
            generate_pdf_report(current_report, output_path)
            return send_file(output_path, mimetype='application/pdf',
                             as_attachment=True,
                             download_name=f'report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf')

        elif fmt == 'csv':
            from export import generate_csv_report
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='w') as tmp:
                output_path = tmp.name
            generate_csv_report(current_report, output_path)
            return send_file(output_path, mimetype='text/csv',
                             as_attachment=True,
                             download_name=f'report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')

    except Exception as e:
        print(f"[API] ✗ Export error: {str(e)}")
        return jsonify({'error': f'Export failed: {str(e)}'}), 500


# ── Feedback ──────────────────────────────────────────────────────────────────

@app.post('/api/feedback')
def save_feedback():
    """Save teacher feedback on a report."""
    try:
        from feedback_db import FeedbackDB

        data = request.get_json()
        required = ['report_id', 'student_id', 'rating']
        if not all(k in data for k in required):
            return jsonify({'error': f'Missing required fields: {required}'}), 400

        db = FeedbackDB()
        db.add_feedback(
            report_id=data['report_id'],
            student_id=data['student_id'],
            rating=data['rating'],
            comments=data.get('comments', ''),
            corrections=data.get('corrections'),
        )
        print(f"[API] ✓ Feedback saved: student {data['student_id']} rated {data['rating']}/5")
        return jsonify({'status': 'saved'})

    except Exception as e:
        print(f"[API] ✗ Feedback error: {str(e)}")
        return jsonify({'error': f'Feedback save failed: {str(e)}'}), 500


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("  ClassWatch Analytics API Server")
    print("  Listening on http://127.0.0.1:5000")
    print("=" * 60)

    app.run(
        host='127.0.0.1',
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False,   # Never reload when run as subprocess
    )
