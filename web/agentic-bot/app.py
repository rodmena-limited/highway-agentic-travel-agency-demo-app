"""Highway Agentic Bot Demo - Conference-Level Flask Application.

Issue #712: Conference demo showcasing Highway's enterprise workflow capabilities.

Architecture:
- Session-based workflow tracking (one active workflow per session)
- Parallel message handler for LLM-powered contextual responses
- Conditional branching on approval (rejected workflows don't continue)
- Real-time status updates via polling

Features demonstrated:
- Long-running activity workers with heartbeats (30+ seconds)
- Updates: Synchronous RPC-style calls to running workflows
- Human-in-the-loop: Approval flow with proper rejection handling
- Parallel execution: Message handler runs alongside main flow
- Session continuity: Subsequent messages get LLM responses from WITHIN workflow
"""

import os
import uuid
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, session


load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(24))

# In-memory store for workflow step results (pushed via webhook)
# Key: workflow_run_id, Value: dict of step results
WORKFLOW_RESULTS_STORE: dict[str, dict] = {}

# Highway API configuration
HIGHWAY_API_URL = os.getenv("HIGHWAY_API_URL", "https://highway.rodmena.app/api/v1")
HIGHWAY_API_KEY = os.getenv("HIGHWAY_API_KEY", "")

# Webhook URL for workflow callbacks (must be accessible from Docker containers)
# Default: http://host.docker.internal:5001 for local Docker development
WEBHOOK_BASE_URL = os.getenv("WEBHOOK_BASE_URL", "http://host.docker.internal:5001")

if not HIGHWAY_API_KEY:
    raise ValueError("HIGHWAY_API_KEY not set in environment")


def get_headers() -> dict[str, str]:
    """Get HTTP headers for Highway API calls."""
    return {
        "Authorization": "Bearer " + HIGHWAY_API_KEY,
        "Content-Type": "application/json",
    }


def get_workflow_status(workflow_run_id: str) -> dict | None:
    """Get current workflow status from Highway API."""
    try:
        resp = requests.get(
            HIGHWAY_API_URL + "/workflows/" + workflow_run_id,
            headers=get_headers(),
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json().get("data", {})
    except requests.exceptions.RequestException:
        pass
    return None


def send_update_to_workflow(workflow_run_id: str, user_message: str) -> dict | None:
    """Send a blocking update to the workflow and get LLM response.

    This uses Highway's Update API (Temporal-style) to send a message to the
    running workflow. The message handler task inside the workflow processes
    the message with LLM and returns a contextual response.

    The call blocks until the workflow processes the update (up to 60 seconds).
    """
    try:
        resp = requests.post(
            HIGHWAY_API_URL + "/workflows/" + workflow_run_id + "/updates",
            headers=get_headers(),
            json={
                "update_name": "user_message",
                "update_payload": {"message": user_message},
                "timeout_seconds": 60,
            },
            timeout=65,  # Slightly longer than update timeout
        )
        if resp.status_code == 200:
            result = resp.json().get("data", {})
            return result.get("result", {})
    except requests.exceptions.RequestException:
        pass
    return None


def get_travel_workflow_dsl(
    session_id: str, webhook_url: str, rag_storage_ids: str = "", approval_key: str = ""
) -> str:
    """Generate the Python DSL code for the travel agent workflow.

    This returns the DSL as a string that Highway will compile.
    All dynamic values use {{inputs.X}} template syntax for runtime resolution.
    Webhook callbacks push results to the Flask app for real-time display.

    Args:
        session_id: Session identifier for approval key
        webhook_url: URL for webhook callbacks
        rag_storage_ids: Comma-separated file IDs for RAG knowledge base
    """
    return (
        '''"""Travel Agent Demo Workflow - Generated DSL.

Demonstrates:
- Parallel branches (main_flow + message_handler)
- Long-running activities with heartbeats
- Human-in-the-loop approval workflow
- Message handler loop with LLM-powered responses
- Conditional branching based on approval
- Webhook callbacks for real-time result streaming
- RAG-powered responses using knowledge base
"""

from datetime import timedelta
from highway_dsl import WorkflowBuilder, RetryPolicy, TimeoutPolicy

# Webhook URL for pushing results to the Flask app
WEBHOOK_URL = "'''
        + webhook_url
        + '''"

# RAG knowledge base storage IDs (comma-separated)
RAG_STORAGE_IDS = "'''
        + rag_storage_ids
        + '''"


def _create_message_loop_body():
    """Create loop body for message handling."""
    def loop_body(b):
        return (
            b.task(
                "handle_one_message",
                "tools.update.handle",
                kwargs={
                    "update_name": "user_message",
                    "response_tool": "tools.llm.call",
                    "response_kwargs": {
                        "provider": "ollama",
                        "model": "qwen3-vl:235b-instruct-cloud",
                        "base_url": "https://ollama.com",
                        "prompt": (
                            "You are a professional travel agency assistant. "
                            "Original request: {{user_message}}\\n\\n"
                            "User follow-up: {{update_payload.message}}\\n\\n"
                            "Respond professionally and concisely (under 80 words). "
                            "No emojis. Formal business tone."
                        ),
                        "temperature": 0.5,
                    },
                    "timeout_seconds": 300,
                },
                result_key="last_response",
            )
            .task(
                "increment_msg_counter",
                "tools.simple_counter.increment_counter",
                result_key="msg_counter",
            )
        )
    return loop_body


def get_workflow():
    """Build the travel agent workflow.

    Structure (v5.2.0 - simple, no early validation):
    1. store_context - Initialize workflow
    2. parallel_execution - Main flow (analyze + search + compile + approval) + message_handler
    3. wait_for_main - Wait for main flow to complete
    4. cancel_message_handler - Stop message handler
    5. check_approval - Switch on approval status
    6. confirm_booking/handle_rejection - Final response

    Note: Early validation was removed due to multiple engine issues:
    - Consecutive LLM calls have async context bugs
    - in/not in operators are unsafe (default to True!)
    - Condition operator has skip propagation bug
    For non-travel requests, the LLM handles gracefully.
    """
    builder = WorkflowBuilder(name="travel_agent_demo", version="5.2.0")
    builder.set_description("AI Travel Agent with Parallel Message Handler and Webhook Callbacks")

    # Store original request as workflow variable
    builder.task(
        "store_context",
        "tools.shell.run",
        args=["echo 'Context stored'"],
        result_key="context_stored",
    )

    # Parallel structure: main_flow + message_handler
    builder.parallel(
        "parallel_execution",
        result_key="parallel_result",
        dependencies=["store_context"],
        branches={
            # Branch 1: Main travel agent flow with webhook callbacks
            "main_flow": lambda b: (
                b.task(
                    "analyze_request",
                    "tools.llm.call",
                    kwargs={
                        "provider": "ollama",
                        "model": "qwen3-vl:235b-instruct-cloud",
                        "base_url": "https://ollama.com",
                        "prompt": "{{user_message}}",
                        "system_prompt": (
                            "Parse travel request. Output ONLY 4 lines:\\n"
                            "Departure: <city after 'Departing from:', or 'London' if empty/missing>\\n"
                            "Destination: <where they want to go - look for city names>\\n"
                            "Dates: <explicit dates, or 'today' if they say 'next flight/ASAP/soon', or 'next month' if unclear>\\n"
                            "Budget: <budget mentioned or 'mid-range'>\\n\\n"
                            "NEVER output anything else. Just the 4 lines."
                        ),
                        "temperature": 0.1,
                    },
                    result_key="analysis",
                    timeout_policy=TimeoutPolicy(timeout=timedelta(seconds=60)),
                    retry_policy=RetryPolicy(max_retries=3, delay=timedelta(seconds=5)),
                )
                # Webhook: push analysis result to Flask app
                .task(
                    "webhook_analysis",
                    "tools.http.request",
                    kwargs={
                        "url": WEBHOOK_URL,
                        "method": "POST",
                        "json_data": {
                            "workflow_run_id": "{{workflow_run_id}}",
                            "step_name": "analysis",
                            "result": "{{analysis}}",
                        },
                        "allow_internal": True,
                        "timeout": 10,
                    },
                    retry_policy=RetryPolicy(max_retries=2, delay=timedelta(seconds=3)),
                )
                .task(
                    "search_flights",
                    "tools.long_running.execute_no_manual_heartbeat",
                    kwargs={"iterations": 3, "delay_seconds": 1.0},
                    result_key="flights",
                    timeout_policy=TimeoutPolicy(timeout=timedelta(minutes=2)),
                    retry_policy=RetryPolicy(max_retries=2, delay=timedelta(seconds=10)),
                )
                # Webhook: push flights result
                .task(
                    "webhook_flights",
                    "tools.http.request",
                    kwargs={
                        "url": WEBHOOK_URL,
                        "method": "POST",
                        "json_data": {
                            "workflow_run_id": "{{workflow_run_id}}",
                            "step_name": "flights",
                            "result": "{{flights}}",
                        },
                        "allow_internal": True,
                        "timeout": 10,
                    },
                    retry_policy=RetryPolicy(max_retries=2, delay=timedelta(seconds=3)),
                )
                .task(
                    "search_hotels",
                    "tools.long_running.execute_no_manual_heartbeat",
                    kwargs={"iterations": 2, "delay_seconds": 1.0},
                    result_key="hotels",
                    timeout_policy=TimeoutPolicy(timeout=timedelta(minutes=2)),
                    retry_policy=RetryPolicy(max_retries=2, delay=timedelta(seconds=10)),
                )
                # Webhook: push hotels result
                .task(
                    "webhook_hotels",
                    "tools.http.request",
                    kwargs={
                        "url": WEBHOOK_URL,
                        "method": "POST",
                        "json_data": {
                            "workflow_run_id": "{{workflow_run_id}}",
                            "step_name": "hotels",
                            "result": "{{hotels}}",
                        },
                        "allow_internal": True,
                        "timeout": 10,
                    },
                    retry_policy=RetryPolicy(max_retries=2, delay=timedelta(seconds=3)),
                )
                .task(
                    "compile_options",
                    "tools.rag.query" if RAG_STORAGE_IDS else "tools.llm.call",
                    kwargs={
                        "storage_ids": RAG_STORAGE_IDS,
                        "query": (
                            "Use the Travel Agent Knowledge Base formatting rules to present ONE optimal package.\\n\\n"
                            "USER REQUEST ANALYSIS:\\n"
                            "{{analysis.response}}\\n\\n"
                            "MANDATORY CONSTRAINTS:\\n"
                            "1. Use the DEPARTURE and DESTINATION cities from the analysis above exactly.\\n"
                            "2. Use REALISTIC flight durations for the actual route.\\n"
                            "3. Follow the 'Standard Response Structure' from the knowledge base.\\n"
                            "4. Do NOT use example cities from the knowledge base (Tokyo, Paris, etc) - "
                            "use ONLY what's in the analysis.\\n\\n"
                            "Generate the package now."
                        ),
                        "model": "qwen3-vl:235b-instruct-cloud",
                        "top_k": 5,
                        "temperature": 0.3,
                    } if RAG_STORAGE_IDS else {
                        "provider": "ollama",
                        "model": "qwen3-vl:235b-instruct-cloud",
                        "base_url": "https://ollama.com",
                        "prompt": (
                            "Create ONE travel package based on this analysis:\\n"
                            "{{analysis.response}}\\n\\n"
                            "Use REALISTIC flight times. Include: outbound flight, return flight, hotel, total price. "
                            "Formal tone. No emojis."
                        ),
                        "temperature": 0.3,
                    },
                    result_key="options",
                    timeout_policy=TimeoutPolicy(timeout=timedelta(seconds=90)),
                    retry_policy=RetryPolicy(max_retries=3, delay=timedelta(seconds=5)),
                )
                # Webhook: push options result (THE KEY ONE for approval display)
                .task(
                    "webhook_options",
                    "tools.http.request",
                    kwargs={
                        "url": WEBHOOK_URL,
                        "method": "POST",
                        "json_data": {
                            "workflow_run_id": "{{workflow_run_id}}",
                            "step_name": "options",
                            "result": "{{options}}",
                        },
                        "allow_internal": True,
                        "timeout": 10,
                    },
                    retry_policy=RetryPolicy(max_retries=2, delay=timedelta(seconds=3)),
                )
                .task(
                    "wait_approval",
                    "tools.approval.request",
                    args=["{{approval_key}}", "Approve Travel Booking"],
                    kwargs={
                        "description": "Review the travel options and approve to proceed with booking.",
                        "approval_data": {"session_id": "{{session_id}}", "message": "{{user_message}}"},
                        "timeout_seconds": 1800,
                    },
                    result_key="approval",
                    timeout_policy=TimeoutPolicy(timeout=timedelta(minutes=30)),
                )
            ),
            # Branch 2: Message handler loop
            "message_handler": lambda b: (
                b.task(
                    "init_msg_counter",
                    "tools.simple_counter.init_counter",
                    kwargs={"start": 0, "max_value": 50},
                    result_key="msg_counter",
                )
                .while_loop(
                    "message_loop",
                    condition="{{msg_counter.counter}} < 50",
                    loop_body=_create_message_loop_body(),
                )
            ),
        },
    )

    # Wait for main_flow branch ONLY - captures last task result (wait_approval)
    builder.task(
        "wait_for_main",
        "tools.workflow.wait_for_branch",
        args=["{{parallel_result}}", "main_flow"],
        kwargs={"timeout_seconds": 2100},
        result_key="main_flow_result",
        dependencies=["parallel_execution"],
        retry_policy=RetryPolicy(max_retries=2, delay=timedelta(seconds=10)),
    )

    # Cancel the message_handler branch
    builder.task(
        "cancel_message_handler",
        "tools.workflow.cancel_branch",
        args=["{{parallel_result}}", "message_handler"],
        dependencies=["wait_for_main"],
        retry_policy=RetryPolicy(max_retries=2, delay=timedelta(seconds=5)),
    )

    # Switch based on approval status
    # main_flow_result is the last task result from main_flow branch (wait_approval)
    builder.switch(
        "check_approval",
        switch_on="{{main_flow_result.status}}",
        cases={
            "approved": "confirm_booking",
            "rejected": "handle_rejection",
        },
        default="handle_rejection",
        dependencies=["cancel_message_handler"],
    )

    # Confirmation (only if approved)
    builder.task(
        "confirm_booking",
        "tools.llm.call",
        kwargs={
            "provider": "ollama",
            "model": "qwen3-vl:235b-instruct-cloud",
            "base_url": "https://ollama.com",
            "prompt": (
                "Generate a formal booking confirmation for this travel package:\\n"
                "{{options.response}}\\n\\n"
                "Include: confirmation number, booking summary, next steps. "
                "Professional business tone. No emojis."
            ),
            "temperature": 0.5,
        },
        result_key="final_message",
        dependencies=["check_approval"],
        timeout_policy=TimeoutPolicy(timeout=timedelta(seconds=60)),
        retry_policy=RetryPolicy(max_retries=3, delay=timedelta(seconds=5)),
    )

    # Webhook for confirm_booking result
    builder.task(
        "webhook_confirm",
        "tools.http.request",
        kwargs={
            "url": WEBHOOK_URL,
            "method": "POST",
            "json_data": {
                "workflow_run_id": "{{workflow_run_id}}",
                "step_name": "final_message",
                "result": "{{final_message}}",
            },
            "allow_internal": True,
            "timeout": 10,
        },
        dependencies=["confirm_booking"],
        retry_policy=RetryPolicy(max_retries=2, delay=timedelta(seconds=3)),
    )

    # Rejection handler (only if rejected)
    builder.task(
        "handle_rejection",
        "tools.llm.call",
        kwargs={
            "provider": "ollama",
            "model": "qwen3-vl:235b-instruct-cloud",
            "base_url": "https://ollama.com",
            "prompt": (
                "The customer declined this travel package:\\n"
                "{{options.response}}\\n\\n"
                "Write a brief professional message acknowledging their decision "
                "and offering to search with different criteria. Keep it short. "
                "Formal business tone. No emojis."
            ),
            "temperature": 0.5,
        },
        result_key="final_message",
        dependencies=["check_approval"],
        timeout_policy=TimeoutPolicy(timeout=timedelta(seconds=60)),
        retry_policy=RetryPolicy(max_retries=3, delay=timedelta(seconds=5)),
    )

    # Webhook for handle_rejection result
    builder.task(
        "webhook_reject",
        "tools.http.request",
        kwargs={
            "url": WEBHOOK_URL,
            "method": "POST",
            "json_data": {
                "workflow_run_id": "{{workflow_run_id}}",
                "step_name": "final_message",
                "result": "{{final_message}}",
            },
            "allow_internal": True,
            "timeout": 10,
        },
        dependencies=["handle_rejection"],
        retry_policy=RetryPolicy(max_retries=2, delay=timedelta(seconds=3)),
    )

    return builder.build().model_dump(mode="json", exclude_none=True)
'''
    )


# ============================================================================
# ROUTES
# ============================================================================


@app.route("/")
def index():
    """Render the chat interface."""
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """Handle chat messages with session-aware workflow management.

    Logic:
    1. If no active workflow -> create new workflow
    2. If workflow running (not at approval) -> send update to get LLM response
    3. If workflow waiting for approval -> prompt user to approve/reject
    4. If workflow completed/failed -> create new workflow
    """
    data = request.get_json()
    user_message = data.get("message", "").strip()
    app.logger.info("Chat received message: %s", user_message)

    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    session_id = session.get("session_id", str(uuid.uuid4()))
    session["session_id"] = session_id

    active_workflow_id = session.get("active_workflow_id")

    if active_workflow_id:
        status = get_workflow_status(active_workflow_id)

        if status:
            wf_status = status.get("status")
            current_step = status.get("current_step")

            # Check if this looks like a new trip request vs follow-up message
            is_new_request = any(
                keyword in user_message.lower()
                for keyword in [
                    "book",
                    "trip",
                    "travel",
                    "flight",
                    "hotel",
                    "vacation",
                    "paris",
                    "london",
                    "tokyo",
                ]
            )

            if wf_status in ("pending", "running", "sleeping") and not is_new_request:
                # Send follow-up messages to existing workflow's message handler
                update_result = send_update_to_workflow(active_workflow_id, user_message)

                if update_result and update_result.get("status") == "success":
                    llm_result = update_result.get("response", {})
                    llm_response = (
                        llm_result.get("response")
                        if isinstance(llm_result, dict)
                        else str(llm_result)
                    )
                    if not llm_response:
                        llm_response = "I'm working on your request..."
                else:
                    llm_response = (
                        "I'm currently working on your travel request. "
                        "I'm at the '%s' step. Please wait a moment..."
                        % (current_step or "processing")
                    )

                return jsonify(
                    {
                        "type": "processing",
                        "workflow_run_id": active_workflow_id,
                        "current_step": current_step,
                        "message": llm_response,
                    }
                )

            # Clear old workflow if it's done OR if user wants a new trip
            if wf_status in ("completed", "failed", "cancelled") or is_new_request:
                WORKFLOW_RESULTS_STORE.pop(active_workflow_id, None)
                session.pop("active_workflow_id", None)
                session.pop("original_request", None)

    # Create new workflow using Python DSL submission
    # The DSL uses {{inputs.X}} for dynamic values resolved at runtime
    # Webhook URL allows workflow to push results directly to the app
    webhook_url = WEBHOOK_BASE_URL + "/api/webhook/step-result"
    # RAG knowledge base for travel recommendations
    rag_storage_ids = os.getenv("RAG_STORAGE_ID", "")
    # Generate unique approval key per workflow (fixes stale approval reuse bug)
    approval_key = str(uuid.uuid4())
    python_dsl = get_travel_workflow_dsl(session_id, webhook_url, rag_storage_ids, approval_key)

    try:
        resp = requests.post(
            HIGHWAY_API_URL + "/workflows",
            headers=get_headers(),
            json={
                "python_dsl": python_dsl,
                "inputs": {
                    "user_message": user_message,
                    "session_id": session_id,
                    "original_request": user_message,
                    "approval_key": approval_key,
                },
            },
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()

        workflow_run_id = result.get("data", {}).get("workflow_run_id")
        if not workflow_run_id:
            return jsonify({"error": "No workflow_run_id in response"}), 500

        session["active_workflow_id"] = workflow_run_id
        session["original_request"] = user_message
        session["approval_key"] = approval_key  # Track for this workflow

        return jsonify(
            {
                "type": "workflow_started",
                "workflow_run_id": workflow_run_id,
                "session_id": session_id,
                "message": "I'm on it! Starting to plan your trip...",
                "approval_key": approval_key,
            }
        )

    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/status/<workflow_run_id>")
def get_status(workflow_run_id: str):
    """Get detailed workflow status."""
    try:
        resp = requests.get(
            HIGHWAY_API_URL + "/workflows/" + workflow_run_id,
            headers=get_headers(),
            timeout=10,
        )
        resp.raise_for_status()
        return jsonify(resp.json())
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/poll/<workflow_run_id>")
def poll_events(workflow_run_id: str):
    """Poll for workflow events and format for UI.

    Returns step results formatted as chat messages for display.
    Frontend should track which steps it has already displayed.
    """
    try:
        resp = requests.get(
            HIGHWAY_API_URL + "/workflows/" + workflow_run_id,
            headers=get_headers(),
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json().get("data", {})

        status = data.get("status")
        current_step = data.get("current_step")
        result = data.get("result") or {}

        # Get webhook-pushed results (takes precedence over API result)
        webhook_results = WORKFLOW_RESULTS_STORE.get(workflow_run_id, {})

        # Extract step results as displayable messages
        step_messages = {}

        # Analysis result (from webhook)
        if webhook_results.get("analysis"):
            analysis = webhook_results["analysis"]
            if isinstance(analysis, dict):
                text = analysis.get("response", "")
            else:
                text = str(analysis)
            if text:
                step_messages["analyze_request"] = "📋 **Analysis**\n\n" + text

        # Flights result (from webhook)
        if webhook_results.get("flights"):
            flights = webhook_results["flights"]
            if isinstance(flights, dict):
                iterations = flights.get(
                    "iterations_completed", flights.get("total_iterations", "?")
                )
                step_messages["search_flights"] = (
                    "✈️ **Flights Found**\n\nSearched %s flight options for your trip!" % iterations
                )
            else:
                step_messages["search_flights"] = "✈️ **Flights Found**\n\nFlight search completed!"

        # Hotels result (from webhook)
        if webhook_results.get("hotels"):
            hotels = webhook_results["hotels"]
            if isinstance(hotels, dict):
                iterations = hotels.get("iterations_completed", hotels.get("total_iterations", "?"))
                step_messages["search_hotels"] = (
                    "🏨 **Hotels Found**\n\nSearched %s hotel options for your stay!" % iterations
                )
            else:
                step_messages["search_hotels"] = "🏨 **Hotels Found**\n\nHotel search completed!"

        # Options result - THE KEY ONE (from webhook)
        if webhook_results.get("options"):
            options = webhook_results["options"]
            if isinstance(options, dict):
                text = options.get("response", "")
            else:
                text = str(options)
            if text:
                step_messages["compile_options"] = "🎯 **Your Travel Options**\n\n" + text

        # Final message if completed (from webhook or API result)
        final_message = None
        if webhook_results.get("final_message"):
            fm = webhook_results["final_message"]
            final_message = fm.get("response") if isinstance(fm, dict) else fm
        elif status == "completed" and result:
            fm = result.get("final_message")
            if fm:
                final_message = fm.get("response") if isinstance(fm, dict) else fm
            if not final_message:
                output = result.get("output")
                if output:
                    final_message = output.get("response") if isinstance(output, dict) else output

        # Fetch real approval_key when waiting for approval
        approval_key = None
        if status == "sleeping" and current_step == "wait_approval":
            try:
                approval_resp = requests.get(
                    HIGHWAY_API_URL + "/approvals",
                    headers=get_headers(),
                    params={"workflow_run_id": workflow_run_id},
                    timeout=5,
                )
                if approval_resp.ok:
                    approvals = approval_resp.json().get("data", {}).get("approvals", [])
                    for a in approvals:
                        if a.get("status") == "pending":
                            approval_key = a.get("approval_key")
                            break
            except Exception as e:
                app.logger.warning("Failed to fetch approval_key: %s", e)

        return jsonify(
            {
                "status": status,
                "current_step": current_step,
                "result": result,
                "step_messages": step_messages,
                "final_message": final_message,
                "approval_key": approval_key,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    except requests.exceptions.RequestException as e:
        return jsonify({"error": "Highway API error: %s" % str(e)}), 500
    except Exception as e:
        # Catch ALL exceptions and return JSON (not HTML 500 page)
        return jsonify({"error": "Poll error: %s" % str(e)}), 500


@app.route("/api/webhook/step-result", methods=["POST"])
def webhook_step_result():
    """Receive step results from workflow via tools.http.request.

    This webhook allows the workflow to push results directly to the app,
    bypassing the parallel branch result storage limitation.
    """
    data = request.get_json(force=True, silent=True) or {}
    app.logger.info("Webhook received data: %s", data)

    workflow_run_id = data.get("workflow_run_id")
    step_name = data.get("step_name")
    result = data.get("result")

    if not workflow_run_id or not step_name:
        app.logger.warning(
            "Webhook missing fields: workflow_run_id=%s, step_name=%s", workflow_run_id, step_name
        )
        return jsonify({"error": "workflow_run_id and step_name required", "received": data}), 400

    # Store result
    if workflow_run_id not in WORKFLOW_RESULTS_STORE:
        WORKFLOW_RESULTS_STORE[workflow_run_id] = {}

    WORKFLOW_RESULTS_STORE[workflow_run_id][step_name] = result
    app.logger.info("Webhook received: %s.%s", workflow_run_id[:8], step_name)

    return jsonify({"status": "ok", "step_name": step_name})


@app.route("/api/signal/<workflow_run_id>", methods=["POST"])
def send_signal(workflow_run_id: str):
    """Send a signal to a running workflow."""
    data = request.get_json()
    signal_name = data.get("signal_name")
    signal_payload = data.get("signal_payload", {})

    if not signal_name:
        return jsonify({"error": "signal_name is required"}), 400

    try:
        resp = requests.post(
            HIGHWAY_API_URL + "/workflows/" + workflow_run_id + "/signals",
            headers=get_headers(),
            json={"signal_name": signal_name, "signal_payload": signal_payload},
            timeout=10,
        )
        resp.raise_for_status()
        return jsonify(resp.json())
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/cancel/<workflow_run_id>", methods=["POST"])
def cancel_workflow(workflow_run_id: str):
    """Cancel a running workflow."""
    try:
        resp = requests.post(
            HIGHWAY_API_URL + "/workflows/" + workflow_run_id + "/cancel",
            headers=get_headers(),
            json={"reason": "User cancelled via UI"},
            timeout=30,
        )
        resp.raise_for_status()

        if session.get("active_workflow_id") == workflow_run_id:
            session.pop("active_workflow_id", None)
            session.pop("original_request", None)

        return jsonify(resp.json())
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/approve", methods=["POST"])
def approve_booking():
    """Approve or reject a pending approval."""
    data = request.get_json()
    approval_key = data.get("approval_key")
    approved = data.get("approved", False)
    comment = data.get("comment", "")

    if not approval_key:
        return jsonify({"error": "approval_key is required"}), 400

    try:
        endpoint = "/approve" if approved else "/reject"
        resp = requests.post(
            HIGHWAY_API_URL + "/approvals/" + approval_key + endpoint,
            headers=get_headers(),
            json={"comment": comment},
            timeout=10,
        )
        resp.raise_for_status()
        result = resp.json()

        return jsonify(
            {
                "status": "approved" if approved else "rejected",
                "approval_key": approval_key,
                "workflow_run_id": result.get("data", {}).get("workflow_run_id"),
                "message": "Booking " + ("approved" if approved else "rejected"),
            }
        )

    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/approvals")
def list_approvals():
    """List pending approvals."""
    try:
        resp = requests.get(
            HIGHWAY_API_URL + "/approvals",
            headers=get_headers(),
            params={"status": "pending"},
            timeout=10,
        )
        resp.raise_for_status()
        return jsonify(resp.json())
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/session/clear", methods=["POST"])
def clear_session():
    """Clear session and start fresh."""
    # Clear cached workflow results for old workflow
    old_workflow_id = session.get("active_workflow_id")
    if old_workflow_id and old_workflow_id in WORKFLOW_RESULTS_STORE:
        del WORKFLOW_RESULTS_STORE[old_workflow_id]

    session.clear()
    session["session_id"] = str(uuid.uuid4())
    return jsonify({"status": "cleared", "session_id": session["session_id"]})


if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5001))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    print("\n" + "=" * 60)
    print("Highway Agentic Bot Demo v3.0")
    print("=" * 60)
    print("API URL: %s" % HIGHWAY_API_URL)
    print("Running on: http://localhost:%d" % port)
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
