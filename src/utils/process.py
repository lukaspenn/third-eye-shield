"""Process management utilities for Third Eye Shield."""
import os
import signal
import time

# Scripts that compete for the RealSense camera
MONITOR_SCRIPTS = (
    'wellness_monitor.py',
    'touchscreen_ui.py',
    'demo_depth.py',
    'demo_showcase.py',
    # Legacy names (handle transition period)
    'touchscreen_launcher.py',
    'rule_based_demo_depth_overlay.py',
    'science_fair_showcase.py',
)


def kill_previous_instances(script_names=MONITOR_SCRIPTS):
    """Kill any previous Python processes running the given scripts.

    This prevents RealSense camera contention when relaunching a monitor script.
    """
    my_pid = os.getpid()
    import subprocess
    try:
        out = subprocess.check_output(['ps', 'aux'], text=True)
        for line in out.splitlines():
            if not any(s in line for s in script_names):
                continue
            parts = line.split()
            pid = int(parts[1])
            if pid == my_pid:
                continue
            try:
                os.kill(pid, signal.SIGTERM)
                print(f"[INIT] Killed previous instance (PID {pid})")
                time.sleep(0.5)
            except ProcessLookupError:
                pass
    except Exception:
        pass
