"""
===================================================================================
run_all.py — Launch all A2A agent servers, then start the orchestrator
===================================================================================

What this does:
    1. Starts news_agent  (port 8001) as a background subprocess
    2. Starts price_agent (port 8002) as a background subprocess
    3. Starts risk_agent  (port 8003) as a background subprocess
    4. Waits for all servers to be ready (polls /.well-known/agent.json)
    5. Starts the orchestrator in the foreground (interactive input loop)

Why subprocesses instead of threads?
    Each agent runs as a FastAPI/Uvicorn ASGI server, which is designed to run
    as an OS process. Using threads would fight the event loop. Subprocesses
    match how you'd deploy these in production (separate containers/VMs).

Alternative (manual launch in 4 terminals):
    terminal 1:  python agents/news_agent.py
    terminal 2:  python agents/price_agent.py
    terminal 3:  python agents/risk_agent.py
    terminal 4:  python orchestrator.py
===================================================================================
"""

import subprocess
import sys
import os
import time
import httpx
import signal

# Resolve paths relative to this file so it works from any working directory
HERE = os.path.dirname(os.path.abspath(__file__))

AGENT_SERVERS = [
    {"name": "news_agent",  "file": "agents/news_agent.py",  "port": 8001},
    {"name": "price_agent", "file": "agents/price_agent.py", "port": 8002},
    {"name": "risk_agent",  "file": "agents/risk_agent.py",  "port": 8003},
]

# Hold subprocess handles so we can terminate them on exit
processes = []


def start_agent_servers():
    """
    Start each agent server as a background subprocess.
    Each subprocess inherits this process's Python executable and environment
    (so venv, GROQ_API_KEY etc. are all available without extra setup).
    """
    print("=" * 60)
    print("  STARTING A2A AGENT SERVERS")
    print("=" * 60)

    for agent in AGENT_SERVERS:
        script_path = os.path.join(HERE, agent["file"])
        proc = subprocess.Popen(
            [sys.executable, script_path],  # use the same Python that's running this file
            stdout=subprocess.PIPE,         # suppress uvicorn startup noise in main terminal
            stderr=subprocess.PIPE,
        )
        processes.append(proc)
        print(f"  → {agent['name']} started (PID {proc.pid}) on port {agent['port']}")

    print()


def wait_for_agents(timeout_seconds: int = 30):
    """
    Poll each agent's Agent Card endpoint until all respond or timeout.

    WHY POLL INSTEAD OF SLEEP?
        Uvicorn startup time varies (0.5s to 3s depending on system load).
        A fixed sleep either wastes time (too long) or fails intermittently (too short).
        Polling is robust — we proceed exactly when agents are ready.
    """
    print("Waiting for agent servers to be ready...")
    deadline = time.time() + timeout_seconds
    ready = set()

    while time.time() < deadline:
        for agent in AGENT_SERVERS:
            if agent["name"] in ready:
                continue
            try:
                url = f"http://localhost:{agent['port']}/.well-known/agent.json"
                httpx.get(url, timeout=1.0)
                ready.add(agent["name"])
                print(f"  ✓ {agent['name']} is ready (port {agent['port']})")
            except Exception:
                pass    # not ready yet, will retry

        if len(ready) == len(AGENT_SERVERS):
            print("\nAll agent servers are ready!\n")
            return True

        time.sleep(0.5)

    # Timeout — report which agents never came up
    not_ready = [a["name"] for a in AGENT_SERVERS if a["name"] not in ready]
    print(f"\n[WARNING] Timed out waiting for: {not_ready}")
    print("Orchestrator will start — unreachable agents will be excluded.\n")
    return False


def shutdown(sig, frame):
    """
    Gracefully terminate all agent server subprocesses on Ctrl+C.
    Without this, the child processes would continue running in the background.
    """
    print("\n\n[run_all.py] Shutting down all agent servers...")
    for proc in processes:
        proc.terminate()
    print("[run_all.py] All servers stopped. Goodbye!")
    sys.exit(0)


if __name__ == "__main__":
    # Register Ctrl+C handler so child processes are cleaned up
    signal.signal(signal.SIGINT, shutdown)

    # Step 1: Start all agent servers in background
    start_agent_servers()

    # Step 2: Wait until all agents are accepting requests
    wait_for_agents()

    # Step 3: Start orchestrator in foreground (same process — takes over stdin)
    print("=" * 60)
    print("  STARTING ORCHESTRATOR")
    print("=" * 60)

    # Import and run orchestrator in this process
    # We could also subprocess.run() it, but running in-process gives cleaner output
    sys.path.insert(0, HERE)
    from orchestrator import main
    main()
