#!/usr/bin/env python3
"""
Flask Web Dashboard for ADAS System
Provides Video, Camera, Analytics, and Reports pages
"""

import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, Response, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import json
import glob
from datetime import datetime
import threading
import queue
import time

# Import existing ADAS components
try:
    from adas_full_pipeline import ADASPipeline
    from report_generator import ReportGenerator
    from data_logger import DataLogger
    ADAS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import ADAS components: {e}")
    ADAS_AVAILABLE = False

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max
app.config['SECRET_KEY'] = 'adas-collision-detection-secret-key'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('reports', exist_ok=True)
os.makedirs('sessions', exist_ok=True)

# Global state for camera streaming
camera_active = False
camera_frame_queue = queue.Queue(maxsize=10)
camera_result = None
camera_lock = threading.Lock()

# Initialize pipeline if available
pipeline = None
if ADAS_AVAILABLE:
    try:
        pipeline = ADASPipeline()
    except Exception as e:
        print(f"Warning: Could not initialize ADAS pipeline: {e}")


def get_session_files():
    """Get list of session CSV files"""
    sessions = []
    csv_files = glob.glob('sessions/*.csv')
    for f in sorted(csv_files, reverse=True):
        try:
            stat = os.stat(f)
            sessions.append({
                'id': os.path.basename(f).replace('.csv', ''),
                'filename': os.path.basename(f),
                'path': f,
                'created': datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
                'size': stat.st_size
            })
        except:
            pass
    return sessions


def parse_session_data(filepath):
    """Parse session CSV data for analytics"""
    data = {
        'timestamps': [],
        'distances': [],
        'speeds': [],
        'ttcs': [],
        'risks': [],
        'brake_pcts': [],
        'directions': [],
        'vehicles': [],
        'animals': [],
        'potholes': [],
        'humps': [],
        'danger_events': [],
        'warning_events': []
    }
    
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            if len(lines) < 2:
                return data
            
            # Skip header
            for line in lines[1:]:
                parts = line.strip().split(',')
                if len(parts) < 10:
                    continue
                
                try:
                    data['timestamps'].append(parts[0] if len(parts) > 0 else '')
                    data['distances'].append(float(parts[5]) if len(parts) > 5 and parts[5] else 0)
                    data['speeds'].append(float(parts[6]) if len(parts) > 6 and parts[6] else 0)
                    data['ttcs'].append(float(parts[7]) if len(parts) > 7 and parts[7] else 0)
                    data['risks'].append(float(parts[8]) if len(parts) > 8 and parts[8] else 0)
                    data['brake_pcts'].append(float(parts[9]) if len(parts) > 9 and parts[9] else 0)
                    data['directions'].append(parts[10] if len(parts) > 10 else 'STRAIGHT')
                    
                    # Count objects
                    if len(parts) > 11:
                        data['vehicles'].append(int(parts[11]) if parts[11] else 0)
                    if len(parts) > 12:
                        data['animals'].append(int(parts[12]) if parts[12] else 0)
                    if len(parts) > 13:
                        data['potholes'].append(int(parts[13]) if parts[13] else 0)
                    if len(parts) > 14:
                        data['humps'].append(int(parts[14]) if parts[14] else 0)
                    
                    # Events
                    risk_val = float(parts[8]) if len(parts) > 8 and parts[8] else 0
                    if risk_val >= 0.8:
                        data['danger_events'].append(1)
                    elif risk_val >= 0.5:
                        data['warning_events'].append(1)
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        print(f"Error parsing session data: {e}")
    
    return data


@app.route('/')
def index():
    """Dashboard home page"""
    sessions = get_session_files()
    total_sessions = len(sessions)
    
    # Get latest session stats if available
    latest_stats = {
        'vehicles': 0,
        'animals': 0,
        'potholes': 0,
        'humps': 0,
        'risk': 'LOW',
        'distance': 'N/A',
        'speed': 'N/A',
        'ttc': 'N/A',
        'brake_pct': 0,
        'direction': 'STRAIGHT'
    }
    
    if sessions:
        latest = sessions[0]
        data = parse_session_data(latest['path'])
        if data['vehicles']:
            latest_stats['vehicles'] = sum(data['vehicles'])
        if data['animals']:
            latest_stats['animals'] = sum(data['animals'])
        if data['potholes']:
            latest_stats['potholes'] = sum(data['potholes'])
        if data['humps']:
            latest_stats['humps'] = sum(data['humps'])
        if data['risks']:
            avg_risk = sum(data['risks']) / len(data['risks'])
            if avg_risk >= 0.8:
                latest_stats['risk'] = 'HIGH'
            elif avg_risk >= 0.5:
                latest_stats['risk'] = 'MEDIUM'
            else:
                latest_stats['risk'] = 'LOW'
    
    return render_template('index.html', 
                         total_sessions=total_sessions,
                         stats=latest_stats,
                         sessions=sessions[:5])


@app.route('/video', methods=['GET', 'POST'])
def video():
    """Video processing page"""
    if request.method == 'POST':
        if 'video' not in request.files:
            return jsonify({'error': 'No video file uploaded'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No video file selected'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process video with ADAS pipeline
        if pipeline:
            try:
                output_path = pipeline.process_video(filepath, display=False)
                return jsonify({
                    'success': True,
                    'output_path': output_path,
                    'message': 'Video processed successfully'
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        else:
            return jsonify({'error': 'ADAS pipeline not available'}), 500
    
    return render_template('video.html')


def generate_camera_frames():
    """Generate MJPEG frames for camera stream"""
    global camera_active, camera_frame_queue
    
    while camera_active:
        try:
            frame = camera_frame_queue.get(timeout=1)
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Camera stream error: {e}")
            time.sleep(0.1)


@app.route('/camera')
def camera():
    """Live camera ADAS page"""
    return render_template('camera.html')


@app.route('/camera/feed')
def camera_feed():
    """MJPEG camera feed endpoint"""
    global camera_active
    camera_active = True
    return Response(generate_camera_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/camera/start', methods=['POST'])
def camera_start():
    """Start camera processing"""
    global camera_active, pipeline
    
    if not pipeline:
        return jsonify({'error': 'ADAS pipeline not available'}), 500
    
    camera_active = True
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        camera_active = False
        return jsonify({'error': 'Could not open camera'}), 500
    
    def process_camera():
        global camera_active, camera_frame_queue, camera_result
        
        while camera_active:
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                # Process frame with ADAS pipeline
                result = pipeline.process_frame(frame.copy(), display=False)
                
                if result and 'frame' in result:
                    with camera_lock:
                        camera_result = result
                        if not camera_frame_queue.full():
                            camera_frame_queue.put(result['frame'])
            except Exception as e:
                print(f"Camera processing error: {e}")
            
            time.sleep(0.03)  # ~30 FPS
        
        cap.release()
    
    thread = threading.Thread(target=process_camera, daemon=True)
    thread.start()
    
    return jsonify({'success': True})


@app.route('/camera/stop', methods=['POST'])
def camera_stop():
    """Stop camera processing"""
    global camera_active
    camera_active = False
    return jsonify({'success': True})


@app.route('/camera/status')
def camera_status():
    """Get current camera processing status"""
    global camera_result
    
    if camera_result:
        return jsonify({
            'active': camera_active,
            'vehicles': len(camera_result.get('vehicles', [])),
            'animals': len(camera_result.get('animals', [])),
            'potholes': len(camera_result.get('potholes', [])),
            'humps': len(camera_result.get('humps', [])),
            'risk': camera_result.get('risk_level', 'LOW'),
            'distance': camera_result.get('min_distance', 0),
            'speed': camera_result.get('relative_speed', 0),
            'ttc': camera_result.get('ttc', 0),
            'brake_pct': camera_result.get('brake_percentage', 0),
            'direction': camera_result.get('direction', 'STRAIGHT')
        })
    
    return jsonify({'active': camera_active})


@app.route('/analytics')
def analytics():
    """Analytics page with graphs"""
    sessions = get_session_files()
    return render_template('analytics.html', sessions=sessions)


@app.route('/analytics/data/<session_id>')
def analytics_data(session_id):
    """Get analytics data for a session"""
    filepath = f'sessions/{session_id}.csv'
    if not os.path.exists(filepath):
        return jsonify({'error': 'Session not found'}), 404
    
    data = parse_session_data(filepath)
    return jsonify(data)


@app.route('/reports')
def reports():
    """Reports listing page"""
    sessions = get_session_files()
    return render_template('reports.html', sessions=sessions)


@app.route('/report/<session_id>')
def report_detail(session_id):
    """Detailed report page"""
    filepath = f'sessions/{session_id}.csv'
    if not os.path.exists(filepath):
        return redirect(url_for('reports'))
    
    data = parse_session_data(filepath)
    
    # Calculate summary statistics
    summary = {
        'session_id': session_id,
        'duration': len(data['timestamps']) if data['timestamps'] else 0,
        'frames': len(data['timestamps']),
        'fps': 30,  # Approximate
        'total_vehicles': sum(data['vehicles']) if data['vehicles'] else 0,
        'total_animals': sum(data['animals']) if data['animals'] else 0,
        'total_potholes': sum(data['potholes']) if data['potholes'] else 0,
        'total_humps': sum(data['humps']) if data['humps'] else 0,
        'danger_events': len(data['danger_events']),
        'warning_events': len(data['warning_events']),
        'min_distance': min(data['distances']) if data['distances'] and any(d > 0 for d in data['distances']) else 0,
        'min_ttc': min([t for t in data['ttcs'] if t > 0]) if data['ttcs'] and any(t > 0 for t in data['ttcs']) else 0
    }
    
    return render_template('report_detail.html', 
                         session_id=session_id,
                         data=data,
                         summary=summary)


@app.route('/download/<session_id>')
def download_report(session_id):
    """Download HTML report"""
    filepath = f'reports/report_{session_id}.html'
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return jsonify({'error': 'Report not found'}), 404


@app.route('/download-csv/<session_id>')
def download_csv(session_id):
    """Download session CSV"""
    filepath = f'sessions/{session_id}.csv'
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True, download_name=f'{session_id}.csv')
    return jsonify({'error': 'CSV not found'}), 404


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
