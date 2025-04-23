import os
import json
from flask import Flask, request, jsonify, g, session, Response  # Import Response
from flask_cors import CORS
from werkzeug.exceptions import ClientDisconnected
import threading
import signal
import uuid  # For request IDs

# Import config
import config
# --- Step 2a: Import the STREAMING function ---
from model import GemmaModelSingleton, get_chatbot_response_stream, get_health_check, logger

# Initialize Flask app
app = Flask(__name__)

# --- 1a. Set Secret Key ---
# This is REQUIRED for Flask sessions to work securely.
app.secret_key = config.SECRET_KEY
# Add a check/warning if the key is weak or missing
if not app.secret_key or app.secret_key == "a-default-development-secret-key":
    logger.warning("Using default or missing Flask SECRET_KEY. Set a strong secret in config.py or environment variable for production.")

# --- 1b. Configure CORS for Credentials ---
# Allow requests from your frontend origin AND allow cookies to be sent/received.
# Replace "http://localhost:5173" with your frontend's actual URL if different.
CORS(app, supports_credentials=True, origins=["http://localhost:5173"])

# Track ongoing requests and their abort flags
active_requests = {}
request_lock = threading.Lock()

# --- Health Check Route (No changes needed here) ---
@app.route('/health', methods=['GET'])
def health():
    logger.info("Health check requested.")
    return jsonify(get_health_check())

# --- Step 2b: Rewrite the /chat route for streaming ---
@app.route('/chat', methods=['POST'])
def chat():
    """Handles chatbot interaction using Server-Sent Events (SSE) for streaming."""
    request_id = str(uuid.uuid4())
    logger.info(f"Received streaming chat request {request_id}")

    # --- Session Handling (Done BEFORE starting the stream response) ---
    if 'conversation' not in session:
        session['conversation'] = []
        logger.info(f"Initialized new conversation history for session.")

    current_history = session['conversation']

    try:
        data = request.get_json()
        if not data or 'message' not in data:
            logger.warning(f"Request {request_id}: Bad request - Missing 'message'.")
            # Return a standard JSON error if setup fails
            return jsonify({'status': 'error', 'message': 'Missing message in request body'}), 400

        user_message = data['message']
        logger.debug(f"Request {request_id}: User message received.")

        # Add user message and trim history
        current_history.append({"role": "user", "content": user_message})
        if len(current_history) > 10:  # Trim history
            current_history = current_history[-10:]
            logger.debug(f"Request {request_id}: Trimmed history.")

        # --- IMPORTANT: Save history to session BEFORE starting the stream ---
        # We cannot reliably modify the session *after* the streaming response starts.
        session['conversation'] = current_history
        session.modified = True
        logger.info(f"Request {request_id}: History updated in session, preparing stream.")

        # --- Prepare Abort Event ---
        abort_event = threading.Event()
        with request_lock:
            active_requests[request_id] = abort_event

        # --- Define the Streaming Generator Function for Flask ---
        def generate_sse():
            """This inner function yields SSE formatted strings."""
            try:
                # Get the generator from the model function
                # Pass the history *as it was before this request's bot response*
                stream_generator = get_chatbot_response_stream(current_history, abort_event)

                full_bot_response_for_log = ""  # Only for logging the final result

                for result in stream_generator:
                    status = result.get("status")

                    # Format data according to SSE specification: "data: json_payload\n\n"
                    sse_data = f"data: {json.dumps(result)}\n\n"
                    yield sse_data  # Send this chunk to the client

                    # Log and check for end conditions
                    if status == "streaming":
                        full_bot_response_for_log += result.get("chunk", "")
                    elif status == "success":
                        logger.info(f"Request {request_id}: Stream finished successfully.")
                        # Log the full response if needed
                        logger.debug(f"Request {request_id}: Full response: {result.get('full_response', '')[:100]}...")
                        # NOTE: Cannot add bot response to session here easily.
                        break  # Stop yielding
                    elif status == "error" or status == "aborted":
                        logger.warning(f"Request {request_id}: Stream ended with status: {status} - {result.get('message')}")
                        break  # Stop yielding

            except Exception as e:
                logger.exception(f"Request {request_id}: Error during SSE generation loop: {e}")
                # Yield a final error message if something breaks mid-stream
                error_payload = {"status": "error", "message": "Streaming failed internally"}
                yield f"data: {json.dumps(error_payload)}\n\n"
            finally:
                # Clean up active request tracking
                with request_lock:
                    if request_id in active_requests:
                        del active_requests[request_id]
                logger.debug(f"Cleaned up streaming request {request_id}")

        # --- Return the Streaming Response ---
        # mimetype 'text/event-stream' is crucial for SSE to work.
        return Response(generate_sse(), mimetype='text/event-stream')

    # --- Error Handling (For errors BEFORE stream starts) ---
    except Exception as e:
        logger.exception(f"Error setting up chat stream request {request_id}: {e}")
        # Clean up just in case abort event was registered
        with request_lock:
            if request_id in active_requests:
                del active_requests[request_id]
        return jsonify({'status': 'error', 'message': "Failed to initiate chat stream."}), 500

# --- Shutdown Handling (No changes needed here) ---
def handle_shutdown(signum, frame):
    logger.info("Shutdown signal received, aborting active requests...")
    with request_lock:
        for req_id, abort_event in active_requests.items():
            logger.info(f"Signalling abort for request {req_id}")
            abort_event.set()

# --- Main Execution (Ensure debug=False for proper session testing) ---
if __name__ == '__main__':
    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting Flask server on port {port}...")
    # IMPORTANT: Use debug=False when testing sessions, as debug mode can interfere.
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)