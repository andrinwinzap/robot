# Required pip packages: fastapi uvicorn[standard]
#
# Usage: python gui.py [--fake-hardware] [--debug] [--port PORT]

import argparse
import asyncio
import json
import os
import queue
import threading
import time
import webbrowser

import numpy as np

from robot_api import Robot
from robot_api.cartesian_space import CartesianSpace
from robot_api.joint_space import JointSpace

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

# ---------------------------------------------------------------------------
# ARGUMENT PARSING
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Robot Command Center")
parser.add_argument("--fake-hardware", action="store_true", help="Run in simulation mode")
parser.add_argument("--debug", action="store_true", help="Enable robot debug logs")
parser.add_argument("--port", type=int, default=8080, help="HTTP port (default: 8080)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# ROBOT SETUP
# ---------------------------------------------------------------------------
robot = Robot()
robot.set_fake_hardware_mode(args.fake_hardware)
robot.set_debug_mode(args.debug)

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
GRIPPER_MIN, GRIPPER_MAX = 0.0, 0.045
GRIPPER_SPD_MIN,  GRIPPER_SPD_MAX,  GRIPPER_SPD_STEP,  GRIPPER_SPD_DEFAULT  = 0.005, 0.5,  0.005, 0.05
LIN_SPD_MIN,      LIN_SPD_MAX,      LIN_SPD_STEP,      LIN_SPD_DEFAULT      = 0.01,  1.0,  0.01,  0.02
ANG_SPD_MIN,      ANG_SPD_MAX,      ANG_SPD_STEP,      ANG_SPD_DEFAULT      = 0.1,   2.0,  0.05,  0.2
LIN_ACCEL_MIN,    LIN_ACCEL_MAX,    LIN_ACCEL_STEP,    LIN_ACCEL_DEFAULT    = 0.01,  1.0,  0.01,  0.02
ANG_ACCEL_MIN,    ANG_ACCEL_MAX,    ANG_ACCEL_STEP,    ANG_ACCEL_DEFAULT    = 0.05,  1.0,  0.05,  0.2
JNT_VEL_MIN,      JNT_VEL_MAX,      JNT_VEL_STEP,      JNT_VEL_DEFAULT      = 0.1,   3.0,  0.1,   1.0

POSITIONS_FILE = os.path.expanduser("~/.robot_saved_positions.json")

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def apply_deadzone(value, deadzone=0.1):
    return value if abs(value) > deadzone else 0.0


def clamp(value, low, high):
    return max(low, min(value, high))


def limit_acceleration(current, target, max_accel, dt):
    max_delta = max_accel * dt
    if max_delta <= 0.0:
        return current
    return current + np.clip(target - current, -max_delta, max_delta)


def load_positions():
    try:
        with open(POSITIONS_FILE) as f:
            return json.load(f)
    except Exception:
        return []


def save_positions_to_file(positions):
    try:
        with open(POSITIONS_FILE, "w") as f:
            json.dump(positions, f, indent=2)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# TOOL MAP
# ---------------------------------------------------------------------------
tool_map = {
    name: getattr(robot.tools, name)
    for name in dir(robot.tools)
    if not name.startswith("_") and isinstance(getattr(robot.tools, name), robot.tools.Tool)
}

try:
    robot.tool_changer.attach_tool(robot.tools.gripper)
except Exception as e:
    try:
        robot.node.get_logger().warn(f"Failed to attach gripper: {e}")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# APP STATE
# ---------------------------------------------------------------------------

class AppState:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = True

        self.l_spd          = LIN_SPD_DEFAULT
        self.a_spd          = ANG_SPD_DEFAULT
        self.max_lin_accel  = LIN_ACCEL_DEFAULT
        self.max_ang_accel  = ANG_ACCEL_DEFAULT
        self.joint_vel      = JNT_VEL_DEFAULT
        self.gripper_speed  = GRIPPER_SPD_DEFAULT

        self.fake_mode        = args.fake_hardware
        self.orientation_lock = False
        self.idle_mode        = False
        self.enforce_linearity = True
        self.motion_mode      = "cartesian"

        self.target_lin_vel = np.zeros(3)
        self.target_ang_vel = np.zeros(3)
        self.current_lin_vel = np.zeros(3)
        self.current_ang_vel = np.zeros(3)

        self.gripper_pos           = 0.0
        self.last_sent_gripper_pos = None

        self.saved_positions = load_positions()

    def get_state_dict(self):
        with self.lock:
            def find_current_tool_name():
                ct = robot.tool_changer.current_tool
                for name, tool in tool_map.items():
                    if tool is ct:
                        return name
                return "none"

            return {
                "type": "state",
                "l_spd": self.l_spd,
                "a_spd": self.a_spd,
                "max_lin_accel": self.max_lin_accel,
                "max_ang_accel": self.max_ang_accel,
                "joint_vel": self.joint_vel,
                "gripper_speed": self.gripper_speed,
                "fake_mode": self.fake_mode,
                "orientation_lock": self.orientation_lock,
                "idle_mode": self.idle_mode,
                "enforce_linearity": self.enforce_linearity,
                "motion_mode": self.motion_mode,
                "gripper_pos": self.gripper_pos,
                "current_tool": find_current_tool_name(),
                "tool_names": list(tool_map.keys()),
                "saved_positions": self.saved_positions,
            }


state = AppState()

# ---------------------------------------------------------------------------
# TELEMETRY QUEUE (bridge between sync thread and async broadcaster)
# ---------------------------------------------------------------------------
telemetry_queue: queue.Queue = queue.Queue(maxsize=2)

# ---------------------------------------------------------------------------
# CONNECTED WEBSOCKET CLIENTS
# ---------------------------------------------------------------------------
connected_clients: list[WebSocket] = []
clients_lock = threading.Lock()

# ---------------------------------------------------------------------------
# ROBOT CONTROL THREAD (~60 fps)
# ---------------------------------------------------------------------------

def control_thread():
    dt = 1.0 / 60.0
    while state.running:
        t0 = time.monotonic()
        with state.lock:
            tl = state.target_lin_vel.copy()
            ta = state.target_ang_vel.copy()
            if state.orientation_lock:
                ta = np.zeros(3)
            max_la = state.max_lin_accel
            max_aa = state.max_ang_accel
            idle   = state.idle_mode
            gpos   = state.gripper_pos
            last_g = state.last_sent_gripper_pos

        if not idle:
            with state.lock:
                state.current_lin_vel = limit_acceleration(state.current_lin_vel, tl, max_la, dt)
                state.current_ang_vel = limit_acceleration(state.current_ang_vel, ta, max_aa, dt)
                clv = state.current_lin_vel.copy()
                cav = state.current_ang_vel.copy()
            try:
                robot.cartesian_space.twist(tuple(clv), tuple(cav))
            except Exception:
                pass
        else:
            with state.lock:
                state.current_lin_vel = np.zeros(3)
                state.current_ang_vel = np.zeros(3)

        if last_g is None or abs(gpos - (last_g if last_g is not None else -1)) > 1e-4:
            try:
                robot.tools.gripper.set_distance(gpos)
                with state.lock:
                    state.last_sent_gripper_pos = gpos
            except Exception:
                pass

        elapsed = time.monotonic() - t0
        remaining = dt - elapsed
        if remaining > 0:
            time.sleep(remaining)


# ---------------------------------------------------------------------------
# TELEMETRY THREAD
# ---------------------------------------------------------------------------

def telemetry_thread():
    while state.running:
        try:
            pose     = robot.cartesian_space.read()
            joint_pt = robot.joint_space.read()
            payload = {
                "type": "telemetry",
                "pose": {
                    "x":     pose.position[0],
                    "y":     pose.position[1],
                    "z":     pose.position[2],
                    "roll":  pose.orientation[0],
                    "pitch": pose.orientation[1],
                    "yaw":   pose.orientation[2],
                },
                "joints": list(joint_pt.joint_configuration),
            }
            try:
                telemetry_queue.put_nowait(payload)
            except queue.Full:
                pass
        except Exception:
            pass
        time.sleep(1.0 / 30.0)


# ---------------------------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------------------------
app = FastAPI()

# Store the running event loop so threads can schedule coroutines
_app_loop: asyncio.AbstractEventLoop = None


@app.on_event("startup")
async def startup():
    global _app_loop
    _app_loop = asyncio.get_event_loop()

    # Start background threads
    ct = threading.Thread(target=control_thread, daemon=True, name="control")
    tt = threading.Thread(target=telemetry_thread, daemon=True, name="telemetry")
    ct.start()
    tt.start()

    # Start async telemetry broadcaster
    asyncio.create_task(telemetry_broadcaster())

    # Open browser
    webbrowser.open(f"http://localhost:{args.port}")


# ---------------------------------------------------------------------------
# ASYNC TELEMETRY BROADCASTER
# ---------------------------------------------------------------------------

async def telemetry_broadcaster():
    while True:
        await asyncio.sleep(0)
        try:
            payload = telemetry_queue.get_nowait()
        except queue.Empty:
            await asyncio.sleep(1.0 / 60.0)
            continue

        msg = json.dumps(payload)
        with clients_lock:
            clients = list(connected_clients)
        dead = []
        for ws in clients:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        if dead:
            with clients_lock:
                for ws in dead:
                    if ws in connected_clients:
                        connected_clients.remove(ws)


# ---------------------------------------------------------------------------
# WEBSOCKET MESSAGE HANDLER
# ---------------------------------------------------------------------------

async def handle_ws_message(ws: WebSocket, data: dict):
    msg_type = data.get("type")

    if msg_type == "input":
        lin = data.get("lin", [0, 0, 0])
        ang = data.get("ang", [0, 0, 0])
        with state.lock:
            state.target_lin_vel = np.array(lin, dtype=float)
            state.target_ang_vel = np.array(ang, dtype=float)

    elif msg_type == "gripper_delta":
        delta = float(data.get("delta", 0.0))
        with state.lock:
            state.gripper_pos = float(np.clip(state.gripper_pos + delta, GRIPPER_MIN, GRIPPER_MAX))

    elif msg_type == "gripper_set":
        val = float(np.clip(float(data.get("value", 0.0)), GRIPPER_MIN, GRIPPER_MAX))
        with state.lock:
            state.gripper_pos = val
            state.last_sent_gripper_pos = None  # force send

    elif msg_type == "toggle":
        key = data.get("key")
        with state.lock:
            if key == "fake_mode":
                state.fake_mode = not state.fake_mode
                try:
                    robot.set_fake_hardware_mode(state.fake_mode)
                except Exception:
                    pass
            elif key == "orientation_lock":
                state.orientation_lock = not state.orientation_lock
            elif key == "idle_mode":
                state.idle_mode = not state.idle_mode
                try:
                    robot.set_idle_mode(state.idle_mode)
                except Exception:
                    pass
            elif key == "enforce_linearity":
                state.enforce_linearity = not state.enforce_linearity

    elif msg_type == "set":
        key   = data.get("key")
        value = float(data.get("value", 0.0))
        with state.lock:
            if key == "l_spd":
                state.l_spd = clamp(value, LIN_SPD_MIN, LIN_SPD_MAX)
            elif key == "a_spd":
                state.a_spd = clamp(value, ANG_SPD_MIN, ANG_SPD_MAX)
            elif key == "max_lin_accel":
                state.max_lin_accel = clamp(value, LIN_ACCEL_MIN, LIN_ACCEL_MAX)
            elif key == "max_ang_accel":
                state.max_ang_accel = clamp(value, ANG_ACCEL_MIN, ANG_ACCEL_MAX)
            elif key == "joint_vel":
                state.joint_vel = clamp(value, JNT_VEL_MIN, JNT_VEL_MAX)
            elif key == "gripper_speed":
                state.gripper_speed = clamp(value, GRIPPER_SPD_MIN, GRIPPER_SPD_MAX)

    elif msg_type == "set_motion_mode":
        mode = data.get("mode", "cartesian")
        with state.lock:
            state.motion_mode = mode

    elif msg_type == "use_current":
        mode = data.get("mode", "cartesian")
        try:
            if mode == "cartesian":
                p    = robot.cartesian_space.read()
                vals = list(p.position) + list(p.orientation)
            else:
                p    = robot.joint_space.read()
                vals = list(p.joint_configuration)
            await ws.send_text(json.dumps({"type": "current_pose", "mode": mode, "values": vals}))
        except Exception as e:
            await ws.send_text(json.dumps({"type": "current_pose", "mode": mode, "values": [0]*6, "error": str(e)}))

    elif msg_type == "get_current_pose":
        mode = data.get("mode", "cartesian")
        try:
            if mode == "cartesian":
                p    = robot.cartesian_space.read()
                vals = list(p.position) + list(p.orientation)
            else:
                p    = robot.joint_space.read()
                vals = list(p.joint_configuration)
            await ws.send_text(json.dumps({"type": "current_pose", "mode": mode, "values": vals}))
        except Exception as e:
            await ws.send_text(json.dumps({"type": "current_pose", "mode": mode, "values": [0]*6, "error": str(e)}))

    elif msg_type == "move":
        mode   = data.get("mode", "cartesian")
        values = data.get("values", [0]*6)
        enf_lin = data.get("enforce_linearity", True)

        async def run_move():
            await ws.send_text(json.dumps({"type": "move_status", "status": "Moving..."}))
            try:
                with state.lock:
                    l_spd_s      = state.l_spd
                    a_spd_s      = state.a_spd
                    max_la_s     = state.max_lin_accel
                    jv_s         = state.joint_vel
                if mode == "cartesian":
                    robot.cartesian_space.max_linear_velocity     = l_spd_s
                    robot.cartesian_space.max_angular_velocity    = a_spd_s
                    robot.cartesian_space.max_linear_acceleration = max_la_s
                    ok = robot.cartesian_space.move(
                        CartesianSpace.Pose(values[:3], values[3:]),
                        enforce_linearity=enf_lin,
                    )
                else:
                    robot.joint_space.max_joint_velocity = jv_s
                    ok = robot.joint_space.move(JointSpace.Point(values))
                status = "Done" if ok else "Failed"
            except Exception as e:
                status = f"Error: {e}"
            try:
                await ws.send_text(json.dumps({"type": "move_status", "status": status}))
            except Exception:
                pass

        asyncio.create_task(run_move())

    elif msg_type == "tool_attach":
        tool_name = data.get("tool")
        def _attach():
            try:
                robot.tool_changer.attach_tool(tool_map[tool_name])
                return f"Attached: {tool_name}"
            except Exception as e:
                return f"Error: {e}"

        loop = _app_loop
        result = await asyncio.get_event_loop().run_in_executor(None, _attach)
        try:
            await ws.send_text(json.dumps({"type": "tool_status", "status": result, "current_tool": tool_name if "Attached" in result else _current_tool_name()}))
        except Exception:
            pass

    elif msg_type == "tool_detach":
        def _detach():
            try:
                robot.tool_changer.detach_tool()
                return "Detached"
            except Exception as e:
                return f"Error: {e}"

        result = await asyncio.get_event_loop().run_in_executor(None, _detach)
        try:
            await ws.send_text(json.dumps({"type": "tool_status", "status": result, "current_tool": "none"}))
        except Exception:
            pass

    elif msg_type == "save_position":
        name   = data.get("name", "pos")
        mode   = data.get("mode", "cartesian")
        values = data.get("values", [0]*6)
        idx    = data.get("idx", None)
        with state.lock:
            if idx is not None and 0 <= idx < len(state.saved_positions):
                state.saved_positions[idx] = {"name": name, "mode": mode, "values": values}
            else:
                state.saved_positions.append({"name": name, "mode": mode, "values": values})
            save_positions_to_file(state.saved_positions)
            positions = state.saved_positions[:]
        await ws.send_text(json.dumps({"type": "positions", "positions": positions}))

    elif msg_type == "delete_position":
        idx = data.get("idx")
        with state.lock:
            if idx is not None and 0 <= idx < len(state.saved_positions):
                state.saved_positions.pop(idx)
                save_positions_to_file(state.saved_positions)
            positions = state.saved_positions[:]
        await ws.send_text(json.dumps({"type": "positions", "positions": positions}))


def _current_tool_name():
    ct = robot.tool_changer.current_tool
    for name, tool in tool_map.items():
        if tool is ct:
            return name
    return "none"


# ---------------------------------------------------------------------------
# WEBSOCKET ENDPOINT
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    with clients_lock:
        connected_clients.append(ws)
    try:
        # Send full state on connect
        await ws.send_text(json.dumps(state.get_state_dict()))
        while True:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            await handle_ws_message(ws, data)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        with clients_lock:
            if ws in connected_clients:
                connected_clients.remove(ws)


# ---------------------------------------------------------------------------
# HTML PAGE (embedded)
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Robot Command Center</title>
<style>
  :root {
    --bg: #0a0a0e;
    --panel: #13131a;
    --card: #0e0e14;
    --border: #242434;
    --text: #d7dae4;
    --dim: #64697a;
    --header: #00d278;
    --accent: #379bff;
    --danger: #821923;
    --danger-hover: #b43737;
    --attach: #195a28;
    --attach-hover: #237d3a;
    --btn: #1b1d2a;
    --btn-active: #007dd7;
    --hover: #303244;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 14px;
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  #app {
    display: flex;
    flex: 1;
    gap: 14px;
    padding: 14px;
    overflow: hidden;
  }
  #left-panel {
    width: 380px;
    flex-shrink: 0;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow-y: auto;
    padding: 16px 16px;
    display: flex;
    flex-direction: column;
    gap: 0;
  }
  #right-panel {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 14px;
    overflow: hidden;
  }
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
  }
  #telemetry-card { flex-shrink: 0; }
  #move-card { flex: 1; overflow-y: auto; }

  /* Section label */
  .section-label {
    font-family: Arial, sans-serif;
    font-size: 11px;
    font-weight: bold;
    color: var(--header);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
    margin-top: 14px;
  }
  .section-label::before {
    content: '';
    display: inline-block;
    width: 3px;
    height: 12px;
    background: var(--header);
    border-radius: 2px;
  }
  .section-label:first-child { margin-top: 0; }
  hr.sep {
    border: none;
    border-top: 1px solid var(--border);
    margin: 14px 0 0 0;
  }

  /* Toggle buttons */
  .toggle-btn {
    width: 100%;
    padding: 8px 12px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--btn);
    color: var(--text);
    cursor: pointer;
    font-size: 14px;
    font-family: inherit;
    margin-bottom: 6px;
    transition: background 0.1s, border-color 0.1s;
  }
  .toggle-btn:hover { background: var(--hover); }
  .toggle-btn.hw-real  { background: #163826; border-color: #00a046; color: #00c85a; }
  .toggle-btn.hw-fake  { background: #461919; border-color: #b43232; color: #dc4646; }
  .toggle-btn.ol-on    { background: #163250; border-color: #379bff; color: #50b4ff; }
  .toggle-btn.ol-off   { background: #281946; border-color: #7843b4; color: #a064e0; }
  .toggle-btn.idle-on  { background: #46300a; border-color: #dc8c1e; color: #ffaf32; }
  .toggle-btn.idle-off { background: #193719; border-color: #00aa50; color: #00d764; }
  .toggle-btn.el-on    { background: #163c26; border-color: #00b45a; color: #00d46e; }
  .toggle-btn.el-off   { background: #3c1e16; border-color: #c86e28; color: #e68c3c; }

  /* Select / dropdown */
  select {
    width: 100%;
    padding: 7px 10px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--btn);
    color: var(--text);
    font-size: 14px;
    font-family: inherit;
    margin-bottom: 6px;
    cursor: pointer;
    appearance: none;
    -webkit-appearance: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'%3E%3Cpath fill='%2364697a' d='M0 0l6 8 6-8z'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 10px center;
  }
  select:focus { outline: none; border-color: var(--accent); }

  /* Legend */
  .legend { margin-top: 6px; }
  .legend-line {
    color: var(--dim);
    font-size: 12px;
    line-height: 1.9;
    font-family: 'Consolas', 'Courier New', monospace;
  }
  .legend-line:last-child { color: var(--accent); }

  /* Kinematics rows */
  .kin-row {
    display: flex;
    align-items: center;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 6px;
    margin-bottom: 6px;
    height: 38px;
    overflow: hidden;
  }
  .kin-btn {
    width: 34px;
    height: 100%;
    background: var(--btn);
    border: none;
    border-right: 1px solid var(--border);
    color: var(--text);
    font-size: 18px;
    cursor: pointer;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: background 0.1s;
  }
  .kin-btn:hover { background: var(--hover); }
  .kin-btn.inc {
    border-right: none;
    border-left: 1px solid var(--border);
  }
  .kin-label {
    font-family: Arial, sans-serif;
    font-size: 11px;
    font-weight: bold;
    color: var(--dim);
    text-transform: uppercase;
    padding-left: 10px;
    flex: 1;
  }
  .kin-value {
    font-size: 13px;
    color: var(--text);
    padding-right: 10px;
    text-align: right;
  }

  /* Tool changer */
  .tool-row {
    display: flex;
    gap: 7px;
    margin-bottom: 6px;
    align-items: stretch;
  }
  .tool-row select { flex: 1; margin-bottom: 0; }
  .btn-attach {
    padding: 7px 14px;
    border-radius: 6px;
    border: 1px solid #00b95a;
    background: var(--attach);
    color: var(--text);
    cursor: pointer;
    font-family: inherit;
    font-size: 14px;
    white-space: nowrap;
    transition: background 0.1s;
  }
  .btn-attach:hover { background: var(--attach-hover); }
  .btn-danger {
    width: 100%;
    padding: 7px 12px;
    border-radius: 6px;
    border: 1px solid #a03030;
    background: var(--danger);
    color: var(--text);
    cursor: pointer;
    font-family: inherit;
    font-size: 14px;
    margin-bottom: 6px;
    transition: background 0.1s;
  }
  .btn-danger:hover { background: var(--danger-hover); }
  .tool-status-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 13px;
    margin-bottom: 6px;
  }
  .tool-status-row .current { color: var(--dim); }
  .tool-status-row .status  { color: var(--dim); }
  .tool-status-row .status.ok  { color: #00be6e; }
  .tool-status-row .status.err { color: #dc6450; }
  .gripper-row {
    display: flex;
    align-items: center;
    gap: 7px;
    margin-top: 4px;
  }
  .gripper-row label { color: var(--dim); font-size: 13px; white-space: nowrap; }
  .gripper-row input[type=number], .gripper-row input[type=text] {
    flex: 1;
    padding: 5px 8px;
    border-radius: 5px;
    border: 1px solid var(--border);
    background: #141420;
    color: var(--text);
    font-family: inherit;
    font-size: 13px;
  }
  .gripper-row input:focus { outline: none; border-color: var(--accent); }
  .btn-set {
    padding: 5px 12px;
    border-radius: 5px;
    border: 1px solid var(--border);
    background: var(--btn);
    color: var(--text);
    cursor: pointer;
    font-family: inherit;
    font-size: 13px;
    white-space: nowrap;
    transition: background 0.1s;
  }
  .btn-set:hover { background: var(--hover); }

  /* Telemetry */
  .tel-title {
    font-family: Arial, sans-serif;
    font-size: 17px;
    font-weight: bold;
    color: var(--header);
    text-align: center;
    margin-bottom: 10px;
  }
  .tel-cols {
    display: flex;
    gap: 40px;
    justify-content: center;
    flex-wrap: wrap;
  }
  .tel-col-title {
    font-family: Arial, sans-serif;
    font-size: 13px;
    font-weight: bold;
    color: var(--header);
    margin-bottom: 4px;
  }
  .tel-row {
    font-size: 13px;
    line-height: 1.85;
    font-family: 'Consolas', 'Courier New', monospace;
    color: var(--text);
    white-space: pre;
  }

  /* Move card */
  .move-title {
    font-family: Arial, sans-serif;
    font-size: 17px;
    font-weight: bold;
    color: var(--header);
    text-align: center;
    margin-bottom: 10px;
  }
  .mode-btns {
    display: flex;
    gap: 7px;
    margin-bottom: 10px;
  }
  .mode-btn {
    flex: 1;
    padding: 7px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--btn);
    color: var(--text);
    cursor: pointer;
    font-family: inherit;
    font-size: 14px;
    transition: background 0.1s, border-color 0.1s;
  }
  .mode-btn.active {
    background: #005a9e;
    border-color: var(--btn-active);
  }
  .mode-btn:hover:not(.active) { background: var(--hover); }
  .inputs-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: repeat(3, auto);
    gap: 5px 10px;
    margin-bottom: 8px;
  }
  .input-row {
    display: flex;
    align-items: center;
    gap: 5px;
  }
  .input-row label {
    width: 50px;
    color: var(--dim);
    font-size: 13px;
    flex-shrink: 0;
  }
  .input-row input {
    flex: 1;
    padding: 5px 8px;
    border-radius: 5px;
    border: 1px solid var(--border);
    background: #141420;
    color: var(--text);
    font-family: inherit;
    font-size: 13px;
    min-width: 0;
  }
  .input-row input:focus { outline: none; border-color: var(--accent); }
  .move-btns {
    display: flex;
    gap: 7px;
    margin-bottom: 6px;
  }
  .btn-move {
    flex: 2;
    padding: 8px;
    border-radius: 6px;
    border: 1px solid var(--btn-active);
    background: #005a9e;
    color: var(--text);
    cursor: pointer;
    font-family: inherit;
    font-size: 14px;
    transition: background 0.1s;
  }
  .btn-move:hover { background: var(--btn-active); }
  .btn-move:disabled {
    background: #2a2c3a;
    border-color: var(--border);
    cursor: not-allowed;
    opacity: 0.6;
  }
  .btn-use-cur {
    flex: 1;
    padding: 8px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--btn);
    color: var(--text);
    cursor: pointer;
    font-family: inherit;
    font-size: 13px;
    transition: background 0.1s;
  }
  .btn-use-cur:hover { background: var(--hover); }
  .move-status {
    font-size: 13px;
    height: 18px;
    margin-bottom: 4px;
  }
  .move-status.ok  { color: #00be6e; }
  .move-status.err { color: #dc6450; }
  .move-status.run { color: var(--accent); }

  /* Saved positions */
  .save-row {
    display: flex;
    gap: 7px;
    margin-bottom: 6px;
    align-items: center;
  }
  .save-row input[type=text] {
    flex: 1;
    padding: 6px 8px;
    border-radius: 5px;
    border: 1px solid var(--border);
    background: #141420;
    color: var(--text);
    font-family: inherit;
    font-size: 13px;
  }
  .save-row input:focus { outline: none; border-color: var(--accent); }
  .btn-save {
    padding: 6px 12px;
    border-radius: 5px;
    border: 1px solid var(--border);
    background: var(--btn);
    color: var(--text);
    cursor: pointer;
    font-family: inherit;
    font-size: 13px;
    white-space: nowrap;
    transition: background 0.1s;
  }
  .btn-save:hover { background: var(--hover); }
  .btn-save.active { background: #005a9e; border-color: var(--btn-active); }
  .btn-cancel { background: var(--btn); }
  .saved-list { display: flex; flex-direction: column; gap: 4px; }
  .saved-item {
    display: flex;
    align-items: center;
    gap: 7px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 4px 8px;
    min-height: 32px;
  }
  .saved-item.editing { background: #162038; border-color: var(--accent); }
  .saved-tag {
    font-size: 11px;
    font-weight: bold;
    font-family: Arial, sans-serif;
    width: 14px;
    flex-shrink: 0;
  }
  .saved-tag.cart { color: var(--accent); }
  .saved-tag.joint { color: #b478ff; }
  .saved-name { flex: 1; font-size: 13px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .saved-btns { display: flex; gap: 4px; flex-shrink: 0; }
  .btn-go {
    padding: 3px 10px;
    border-radius: 4px;
    border: 1px solid #00b95a;
    background: var(--attach);
    color: var(--text);
    cursor: pointer;
    font-family: inherit;
    font-size: 12px;
    transition: background 0.1s;
  }
  .btn-go:hover { background: var(--attach-hover); }
  .btn-edit {
    padding: 3px 10px;
    border-radius: 4px;
    border: 1px solid var(--border);
    background: var(--btn);
    color: var(--text);
    cursor: pointer;
    font-family: inherit;
    font-size: 12px;
    transition: background 0.1s;
  }
  .btn-edit:hover { background: var(--hover); }
  .btn-edit.active { background: #005a9e; border-color: var(--btn-active); }
  .btn-del {
    padding: 3px 8px;
    border-radius: 4px;
    border: 1px solid #6e1818;
    background: var(--danger);
    color: #ffb0a0;
    cursor: pointer;
    font-family: inherit;
    font-size: 12px;
    transition: background 0.1s;
  }
  .btn-del:hover { background: var(--danger-hover); }
  ::-webkit-scrollbar { width: 5px; }
  ::-webkit-scrollbar-track { background: var(--card); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  .warn { color: #e09c28; font-size: 12px; margin-bottom: 4px; }
</style>
</head>
<body>
<div id="app">
  <!-- LEFT PANEL -->
  <div id="left-panel">
    <div class="section-label">Control</div>

    <button id="btn-hw"       class="toggle-btn">...</button>
    <button id="btn-ol"       class="toggle-btn">...</button>
    <button id="btn-idle"     class="toggle-btn">...</button>

    <select id="sel-input">
      <option value="keyboard">Input: Keyboard</option>
      <option value="controller">Input: Controller</option>
    </select>

    <select id="sel-mapping" style="display:none">
      <option value="standard">Mapping: Standard Dual Stick</option>
      <option value="drone">Mapping: Drone Style (Legacy)</option>
    </select>

    <div id="warn-no-ctrl" class="warn" style="display:none">⚠ No controller detected</div>

    <div class="legend" id="legend"></div>

    <hr class="sep"/>
    <div class="section-label">Kinematics</div>
    <div id="kin-rows"></div>

    <hr class="sep"/>
    <div class="section-label">Tool Changer</div>
    <div class="tool-row">
      <select id="sel-tool"></select>
      <button class="btn-attach" id="btn-attach">ATTACH</button>
    </div>
    <button class="btn-danger" id="btn-detach">DETACH</button>
    <div class="tool-status-row">
      <span class="current" id="lbl-current-tool">Current: none</span>
      <span class="status" id="lbl-tool-status">Ready</span>
    </div>
    <div id="gripper-section" style="display:none">
      <div class="gripper-row">
        <label>Gripper</label>
        <input type="number" id="inp-gripper-pos" step="0.0001" min="0" max="0.045" value="0.0000"/>
        <button class="btn-set" id="btn-grip-set">SET</button>
      </div>
    </div>
  </div>

  <!-- RIGHT PANEL -->
  <div id="right-panel">
    <!-- TELEMETRY CARD -->
    <div class="card" id="telemetry-card">
      <div class="tel-title">SYSTEM TELEMETRY</div>
      <div class="tel-cols">
        <div>
          <div class="tel-col-title">Cartesian Pose</div>
          <div class="tel-row" id="tel-cart"></div>
        </div>
        <div>
          <div class="tel-col-title">Joint Angles</div>
          <div class="tel-row" id="tel-joints"></div>
        </div>
      </div>
    </div>

    <!-- MOVE TO TARGET CARD -->
    <div class="card" id="move-card">
      <div class="move-title">MOVE TO TARGET</div>
      <div class="mode-btns">
        <button class="mode-btn" id="btn-cart-mode">Cartesian</button>
        <button class="mode-btn" id="btn-joint-mode">Joint</button>
      </div>
      <div class="inputs-grid" id="move-inputs-grid"></div>
      <button id="btn-enforce-lin" class="toggle-btn el-on" style="margin-bottom:8px">Enforce Linearity: ON</button>
      <div class="move-btns">
        <button class="btn-move" id="btn-move">MOVE</button>
        <button class="btn-use-cur" id="btn-use-cur">USE CURRENT</button>
      </div>
      <div class="move-status" id="move-status"></div>

      <hr class="sep" style="margin-bottom:10px"/>
      <div class="section-label" style="margin-top:0">Saved Positions</div>
      <div class="save-row">
        <input type="text" id="inp-save-name" value="pos_1" placeholder="name"/>
        <button class="btn-save" id="btn-save">SAVE</button>
        <button class="btn-save btn-cancel" id="btn-cancel-edit" style="display:none">CANCEL</button>
      </div>
      <div class="saved-list" id="saved-list"></div>
    </div>
  </div>
</div>

<script>
// ============================================================
// STATE
// ============================================================
const S = {
  l_spd: 0.02, a_spd: 0.2,
  max_lin_accel: 0.02, max_ang_accel: 0.2,
  joint_vel: 1.0, gripper_speed: 0.05,
  fake_mode: false, orientation_lock: false,
  idle_mode: false, enforce_linearity: true,
  motion_mode: 'cartesian',
  gripper_pos: 0.0,
  current_tool: 'none',
  tool_names: [],
  saved_positions: [],
};

let ws = null;
let wsReady = false;
let editingIdx = null;
let inputFocused = false;

// ============================================================
// WEBSOCKET
// ============================================================
function connectWS() {
  const url = `ws://${location.host}/ws`;
  ws = new WebSocket(url);
  ws.onopen  = () => { wsReady = true; };
  ws.onclose = () => { wsReady = false; setTimeout(connectWS, 1500); };
  ws.onerror = () => { ws.close(); };
  ws.onmessage = (e) => {
    let msg;
    try { msg = JSON.parse(e.data); } catch { return; }
    handleMessage(msg);
  };
}

function send(obj) {
  if (wsReady && ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(obj));
  }
}

// ============================================================
// MESSAGE HANDLER
// ============================================================
function handleMessage(msg) {
  switch (msg.type) {
    case 'state':
      Object.assign(S, msg);
      applyFullState();
      break;
    case 'telemetry':
      updateTelemetry(msg);
      break;
    case 'move_status':
      setMoveStatus(msg.status);
      break;
    case 'tool_status':
      setToolStatus(msg.status, msg.current_tool);
      break;
    case 'positions':
      S.saved_positions = msg.positions;
      renderSavedPositions();
      break;
    case 'current_pose':
      fillMoveInputs(msg.values);
      if (msg.mode) { S.motion_mode = msg.mode; updateMotionModeUI(); }
      break;
  }
}

// ============================================================
// FULL STATE APPLY
// ============================================================
function applyFullState() {
  // hardware toggles
  updateHWBtn();
  updateOLBtn();
  updateIdleBtn();
  updateELBtn();

  // input method
  document.getElementById('sel-input').value = S.input_method || 'keyboard';
  updateInputMethodUI();

  // kinematics
  updateKinRow(0, S.l_spd,         'Lin Vel',  'm/s');
  updateKinRow(1, S.a_spd,         'Ang Vel',  'rad/s');
  updateKinRow(2, S.max_lin_accel, 'Lin Acc',  'm/s²');
  updateKinRow(3, S.max_ang_accel, 'Ang Acc',  'rad/s²');
  updateKinRow(4, S.joint_vel,     'Jnt Vel',  'rad/s');
  updateKinRow(5, S.gripper_speed, 'Grp Spd',  'm/s');

  // tools
  rebuildToolDropdown();
  updateCurrentTool(S.current_tool);

  // motion mode
  updateMotionModeUI();

  // saved positions
  S.saved_positions = S.saved_positions || [];
  renderSavedPositions();

  // gripper input
  document.getElementById('inp-gripper-pos').value = S.gripper_pos.toFixed(4);
}

// ============================================================
// TOGGLE BUTTONS
// ============================================================
function updateHWBtn() {
  const b = document.getElementById('btn-hw');
  b.className = 'toggle-btn ' + (S.fake_mode ? 'hw-fake' : 'hw-real');
  b.textContent = S.fake_mode ? '● FAKE HARDWARE' : '● REAL HARDWARE';
}
function updateOLBtn() {
  const b = document.getElementById('btn-ol');
  b.className = 'toggle-btn ' + (S.orientation_lock ? 'ol-on' : 'ol-off');
  b.textContent = S.orientation_lock ? 'ORIENTATION LOCKED' : 'ORIENTATION FREE';
}
function updateIdleBtn() {
  const b = document.getElementById('btn-idle');
  b.className = 'toggle-btn ' + (S.idle_mode ? 'idle-on' : 'idle-off');
  b.textContent = S.idle_mode ? '● IDLE' : '● ACTIVE';
}
function updateELBtn() {
  const b = document.getElementById('btn-enforce-lin');
  b.className = 'toggle-btn ' + (S.enforce_linearity ? 'el-on' : 'el-off');
  b.textContent = 'Enforce Linearity: ' + (S.enforce_linearity ? 'ON' : 'OFF');
}

document.getElementById('btn-hw').onclick = () => {
  send({type:'toggle', key:'fake_mode'});
  S.fake_mode = !S.fake_mode; updateHWBtn();
};
document.getElementById('btn-ol').onclick = () => {
  send({type:'toggle', key:'orientation_lock'});
  S.orientation_lock = !S.orientation_lock; updateOLBtn();
};
document.getElementById('btn-idle').onclick = () => {
  send({type:'toggle', key:'idle_mode'});
  S.idle_mode = !S.idle_mode; updateIdleBtn();
};
document.getElementById('btn-enforce-lin').onclick = () => {
  send({type:'toggle', key:'enforce_linearity'});
  S.enforce_linearity = !S.enforce_linearity; updateELBtn();
};

// ============================================================
// INPUT METHOD & MAPPING
// ============================================================
const KEYBOARD_LEGEND = [
  'W / S:          Fwd / Back',
  'A / D:          Side',
  'Space / LShift: Up / Down',
  'Left / Right:   Roll (inverted)',
  'Up / Down:      Pitch',
  'Q / E:          Yaw',
  'O / P:          Gripper',
];
const MAPPINGS = {
  standard: {
    label: 'Standard Dual Stick',
    legend: [
      'L-Stick Vert:  Fwd / Back',
      'L-Stick Horiz: Side',
      'R-Stick Horiz: Roll',
      'R-Stick Vert:  Pitch',
      'L2 / R2:       Up / Down',
      'L1 / R1:       Yaw',
      'Circle / X:    Gripper Open/Close',
    ],
  },
  drone: {
    label: 'Drone Style (Legacy)',
    legend: [
      'L-Stick Vert:  Up / Down',
      'L-Stick Horiz: Yaw',
      'R-Stick Vert:  Fwd / Back',
      'R-Stick Horiz: Side',
      'D-Pad Vert:    Pitch',
      'D-Pad Horiz:   Roll',
      'LB / RB:       Gripper',
    ],
  },
};

let selectedInputMethod = 'keyboard';
let selectedMapping     = 'standard';

document.getElementById('sel-input').onchange = function() {
  selectedInputMethod = this.value;
  updateInputMethodUI();
};
document.getElementById('sel-mapping').onchange = function() {
  selectedMapping = this.value;
  updateLegend();
};

function updateInputMethodUI() {
  const isCtrl = selectedInputMethod === 'controller';
  document.getElementById('sel-mapping').style.display = isCtrl ? '' : 'none';
  const gpads = navigator.getGamepads ? navigator.getGamepads() : [];
  const hasGP = Array.from(gpads).some(g => g && g.connected);
  document.getElementById('warn-no-ctrl').style.display = (isCtrl && !hasGP) ? '' : 'none';
  updateLegend();
}

function updateLegend() {
  const lines = selectedInputMethod === 'controller'
    ? MAPPINGS[selectedMapping].legend
    : KEYBOARD_LEGEND;
  const el = document.getElementById('legend');
  el.innerHTML = lines.map((l, i) =>
    `<div class="legend-line${i === lines.length-1 ? ' last' : ''}">${l}</div>`
  ).join('');
}

// ============================================================
// KINEMATICS
// ============================================================
const KIN_CONFIG = [
  { key:'l_spd',         label:'Lin Vel',  unit:'m/s',   min:0.01, max:1.0,  step:0.01  },
  { key:'a_spd',         label:'Ang Vel',  unit:'rad/s', min:0.1,  max:2.0,  step:0.05  },
  { key:'max_lin_accel', label:'Lin Acc',  unit:'m/s²',  min:0.01, max:1.0,  step:0.01  },
  { key:'max_ang_accel', label:'Ang Acc',  unit:'rad/s²',min:0.05, max:1.0,  step:0.05  },
  { key:'joint_vel',     label:'Jnt Vel',  unit:'rad/s', min:0.1,  max:3.0,  step:0.1   },
  { key:'gripper_speed', label:'Grp Spd',  unit:'m/s',   min:0.005,max:0.5,  step:0.005 },
];

function buildKinRows() {
  const container = document.getElementById('kin-rows');
  KIN_CONFIG.forEach((cfg, i) => {
    const row = document.createElement('div');
    row.className = 'kin-row';
    row.innerHTML = `
      <button class="kin-btn dec" data-idx="${i}">−</button>
      <span class="kin-label">${cfg.label.toUpperCase()}</span>
      <span class="kin-value" id="kin-val-${i}">...</span>
      <button class="kin-btn inc" data-idx="${i}">+</button>
    `;
    container.appendChild(row);
  });
  container.querySelectorAll('.dec').forEach(b => b.onclick = () => adjustKin(+b.dataset.idx, -1));
  container.querySelectorAll('.inc').forEach(b => b.onclick = () => adjustKin(+b.dataset.idx, +1));
}

function adjustKin(idx, dir) {
  const cfg = KIN_CONFIG[idx];
  let val = parseFloat(S[cfg.key]) || 0;
  val = Math.round((val + dir * cfg.step) * 1e6) / 1e6;
  val = Math.min(cfg.max, Math.max(cfg.min, val));
  S[cfg.key] = val;
  send({type:'set', key: cfg.key, value: val});
  updateKinRow(idx, val, cfg.label, cfg.unit);
}

function updateKinRow(idx, val, label, unit) {
  const el = document.getElementById(`kin-val-${idx}`);
  if (el) el.textContent = `${val.toFixed(idx >= 4 ? 2 : 3)} ${unit}`;
}

// ============================================================
// TOOL CHANGER
// ============================================================
function rebuildToolDropdown() {
  const sel = document.getElementById('sel-tool');
  sel.innerHTML = '';
  (S.tool_names || []).forEach(name => {
    const o = document.createElement('option');
    o.value = name; o.textContent = name;
    sel.appendChild(o);
  });
}

function updateCurrentTool(name) {
  S.current_tool = name;
  document.getElementById('lbl-current-tool').textContent = `Current: ${name}`;
  document.getElementById('gripper-section').style.display = name === 'gripper' ? '' : 'none';
}

function setToolStatus(status, currentTool) {
  const el = document.getElementById('lbl-tool-status');
  el.textContent = status;
  el.className = 'status';
  if (status.toLowerCase().includes('error') || status.toLowerCase().includes('fail')) {
    el.classList.add('err');
  } else if (status.toLowerCase().includes('attach') || status.toLowerCase().includes('detach')) {
    el.classList.add('ok');
  }
  if (currentTool !== undefined) updateCurrentTool(currentTool);
}

document.getElementById('btn-attach').onclick = () => {
  const tool = document.getElementById('sel-tool').value;
  if (tool) send({type:'tool_attach', tool});
};
document.getElementById('btn-detach').onclick = () => {
  send({type:'tool_detach'});
};
document.getElementById('btn-grip-set').onclick = () => {
  const val = parseFloat(document.getElementById('inp-gripper-pos').value) || 0;
  send({type:'gripper_set', value: val});
};

// ============================================================
// TELEMETRY
// ============================================================
function fmt(v) { return (v >= 0 ? ' ' : '') + v.toFixed(4); }

function updateTelemetry(msg) {
  const p = msg.pose;
  document.getElementById('tel-cart').textContent =
    `X      ${fmt(p.x)} m\n` +
    `Y      ${fmt(p.y)} m\n` +
    `Z      ${fmt(p.z)} m\n` +
    `Roll   ${fmt(p.roll)} rad\n` +
    `Pitch  ${fmt(p.pitch)} rad\n` +
    `Yaw    ${fmt(p.yaw)} rad`;

  const joints = msg.joints;
  document.getElementById('tel-joints').textContent =
    joints.map((v, i) => `q${i+1}     ${fmt(v)} rad`).join('\n');
}

// ============================================================
// MOVE TO TARGET
// ============================================================
const CART_LABELS  = ['X (m)', 'Y (m)', 'Z (m)', 'Roll', 'Pitch', 'Yaw'];
const JOINT_LABELS = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6'];

function buildMoveInputs() {
  const grid = document.getElementById('move-inputs-grid');
  grid.innerHTML = '';
  // 2 columns, 3 rows — indices: col=i//3, row=i%3
  // We'll lay them out as: left col = i in [0,1,2], right col = i in [3,4,5]
  // Grid is column-major: we want col 0 on left, col 1 on right
  // CSS grid is row-major, so: items 0,1,2 go to rows 1,2,3 of col 1; items 3,4,5 to col 2
  // Use order: 0→grid-area(1,1), 1→(2,1), 2→(3,1), 3→(1,2), 4→(2,2), 5→(3,2)
  for (let i = 0; i < 6; i++) {
    const row = document.createElement('div');
    row.className = 'input-row';
    // col-major: i<3 → col 1 (left), i>=3 → col 2 (right)
    // grid row = i%3+1
    const col = Math.floor(i / 3) + 1;
    const gr  = (i % 3) + 1;
    row.style.gridColumn = col;
    row.style.gridRow    = gr;
    row.innerHTML = `<label id="move-lbl-${i}"></label><input type="number" id="move-inp-${i}" step="0.0001" value="0.0000"/>`;
    grid.appendChild(row);
  }
  updateMoveLabels();
}

function updateMoveLabels() {
  const labels = S.motion_mode === 'cartesian' ? CART_LABELS : JOINT_LABELS;
  labels.forEach((lbl, i) => {
    const el = document.getElementById(`move-lbl-${i}`);
    if (el) el.textContent = lbl;
  });
}

function fillMoveInputs(vals) {
  vals.forEach((v, i) => {
    const el = document.getElementById(`move-inp-${i}`);
    if (el) el.value = v.toFixed(4);
  });
}

function getMoveValues() {
  return Array.from({length:6}, (_, i) => {
    const el = document.getElementById(`move-inp-${i}`);
    return el ? parseFloat(el.value) || 0 : 0;
  });
}

function updateMotionModeUI() {
  document.getElementById('btn-cart-mode').classList.toggle('active', S.motion_mode === 'cartesian');
  document.getElementById('btn-joint-mode').classList.toggle('active', S.motion_mode === 'joint');
  document.getElementById('btn-enforce-lin').style.display = S.motion_mode === 'cartesian' ? '' : 'none';
  updateMoveLabels();
}

document.getElementById('btn-cart-mode').onclick = () => {
  if (S.motion_mode !== 'cartesian') {
    S.motion_mode = 'cartesian';
    send({type:'set_motion_mode', mode:'cartesian'});
    send({type:'get_current_pose', mode:'cartesian'});
    updateMotionModeUI();
  }
};
document.getElementById('btn-joint-mode').onclick = () => {
  if (S.motion_mode !== 'joint') {
    S.motion_mode = 'joint';
    send({type:'set_motion_mode', mode:'joint'});
    send({type:'get_current_pose', mode:'joint'});
    updateMotionModeUI();
  }
};

document.getElementById('btn-use-cur').onclick = () => {
  send({type:'use_current', mode: S.motion_mode});
};

document.getElementById('btn-move').onclick = () => {
  send({
    type: 'move',
    mode: S.motion_mode,
    values: getMoveValues(),
    enforce_linearity: S.enforce_linearity,
  });
};

function setMoveStatus(status) {
  const el = document.getElementById('move-status');
  el.textContent = status;
  el.className = 'move-status';
  if (status.toLowerCase().includes('error') || status.toLowerCase().includes('fail')) {
    el.classList.add('err');
  } else if (status.toLowerCase().includes('done')) {
    el.classList.add('ok');
  } else if (status.toLowerCase().includes('moving')) {
    el.classList.add('run');
  }
}

// ============================================================
// SAVED POSITIONS
// ============================================================
function renderSavedPositions() {
  const list = document.getElementById('saved-list');
  list.innerHTML = '';
  (S.saved_positions || []).forEach((pos, idx) => {
    const item = document.createElement('div');
    item.className = 'saved-item' + (editingIdx === idx ? ' editing' : '');
    const tag = pos.mode === 'cartesian' ? 'C' : 'J';
    const tagClass = pos.mode === 'cartesian' ? 'cart' : 'joint';
    item.innerHTML = `
      <span class="saved-tag ${tagClass}">${tag}</span>
      <span class="saved-name">${pos.name}</span>
      <div class="saved-btns">
        <button class="btn-go"  data-idx="${idx}">GO</button>
        <button class="btn-edit${editingIdx===idx?' active':''}" data-idx="${idx}">EDIT</button>
        <button class="btn-del" data-idx="${idx}">✕</button>
      </div>
    `;
    list.appendChild(item);
  });

  list.querySelectorAll('.btn-go').forEach(b => b.onclick = () => {
    const idx = +b.dataset.idx;
    const pos = S.saved_positions[idx];
    if (!pos) return;
    S.motion_mode = pos.mode;
    send({type:'set_motion_mode', mode: pos.mode});
    updateMotionModeUI();
    fillMoveInputs(pos.values);
  });
  list.querySelectorAll('.btn-edit').forEach(b => b.onclick = () => {
    const idx = +b.dataset.idx;
    const pos = S.saved_positions[idx];
    if (!pos) return;
    if (editingIdx === idx) {
      // cancel editing
      editingIdx = null;
      document.getElementById('inp-save-name').value = `pos_${S.saved_positions.length + 1}`;
      document.getElementById('btn-save').textContent = 'SAVE';
      document.getElementById('btn-save').classList.remove('active');
      document.getElementById('btn-cancel-edit').style.display = 'none';
      renderSavedPositions();
      return;
    }
    editingIdx = idx;
    document.getElementById('inp-save-name').value = pos.name;
    S.motion_mode = pos.mode;
    send({type:'set_motion_mode', mode: pos.mode});
    updateMotionModeUI();
    fillMoveInputs(pos.values);
    document.getElementById('btn-save').textContent = 'UPDATE';
    document.getElementById('btn-save').classList.add('active');
    document.getElementById('btn-cancel-edit').style.display = '';
    renderSavedPositions();
  });
  list.querySelectorAll('.btn-del').forEach(b => b.onclick = () => {
    const idx = +b.dataset.idx;
    if (editingIdx === idx) {
      editingIdx = null;
      document.getElementById('inp-save-name').value = `pos_${S.saved_positions.length}`;
      document.getElementById('btn-save').textContent = 'SAVE';
      document.getElementById('btn-save').classList.remove('active');
      document.getElementById('btn-cancel-edit').style.display = 'none';
    }
    send({type:'delete_position', idx});
  });
}

document.getElementById('btn-save').onclick = () => {
  const name = document.getElementById('inp-save-name').value.trim() || `pos_${S.saved_positions.length+1}`;
  const payload = {
    type: 'save_position',
    name,
    mode: S.motion_mode,
    values: getMoveValues(),
  };
  if (editingIdx !== null) {
    payload.idx = editingIdx;
    editingIdx = null;
    document.getElementById('btn-save').textContent = 'SAVE';
    document.getElementById('btn-save').classList.remove('active');
    document.getElementById('btn-cancel-edit').style.display = 'none';
  }
  send(payload);
  document.getElementById('inp-save-name').value = `pos_${S.saved_positions.length + 1}`;
};

document.getElementById('btn-cancel-edit').onclick = () => {
  editingIdx = null;
  document.getElementById('inp-save-name').value = `pos_${S.saved_positions.length + 1}`;
  document.getElementById('btn-save').textContent = 'SAVE';
  document.getElementById('btn-save').classList.remove('active');
  document.getElementById('btn-cancel-edit').style.display = 'none';
  renderSavedPositions();
};

// ============================================================
// INPUT FOCUS TRACKING (disable keyboard movement when typing)
// ============================================================
document.addEventListener('focusin', e => {
  const tag = e.target.tagName;
  inputFocused = (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT');
});
document.addEventListener('focusout', () => {
  inputFocused = false;
});

// ============================================================
// KEYBOARD STATE
// ============================================================
const keys = {};
document.addEventListener('keydown', e => {
  // Allow Ctrl+A/C/V/X always
  if (e.ctrlKey && ['a','c','v','x'].includes(e.key.toLowerCase())) return;
  keys[e.code] = true;
});
document.addEventListener('keyup', e => { keys[e.code] = false; });

// ============================================================
// GAMEPAD SUPPORT
// ============================================================
let gamepadIndex = null;
window.addEventListener('gamepadconnected',    e => { gamepadIndex = e.gamepad.index; updateInputMethodUI(); });
window.addEventListener('gamepaddisconnected', e => { if (gamepadIndex === e.gamepad.index) gamepadIndex = null; updateInputMethodUI(); });

function applyDeadzone(v, dz=0.1) { return Math.abs(v) > dz ? v : 0; }

function readGamepad(gp, lspd, aspd) {
  const axes = gp.axes;
  const btns = gp.buttons;
  const readBtn = (i) => btns[i] && btns[i].pressed;
  const readTrigger = (axisIdx, btnIdx) => {
    let t = 0;
    if (axes[axisIdx] !== undefined) {
      const raw = axes[axisIdx];
      t = raw < -0.05 ? (raw + 1.0) / 2.0 : raw;
      t = Math.max(0, t);
    }
    if (t <= 0 && readBtn(btnIdx)) t = 1.0;
    return t;
  };

  let lin = [0, 0, 0];
  let ang = [0, 0, 0];
  let gripDelta = 0;

  if (selectedMapping === 'standard') {
    lin[0] = -applyDeadzone(axes[1] || 0) * lspd;
    lin[1] = -applyDeadzone(axes[0] || 0) * lspd;
    ang[0] =  applyDeadzone(axes[3] || 0) * aspd;
    ang[1] = -applyDeadzone(axes[4] || 0) * aspd;
    const l2 = readTrigger(2, 6);
    const r2 = readTrigger(5, 7);
    lin[2] = (l2 - r2) * lspd;
    ang[2] = (readBtn(5) ? 1 : 0) - (readBtn(4) ? 1 : 0);
    ang[2] *= aspd;
    // Circle=btn1 open, X=btn0 close
    if (readBtn(1)) gripDelta =  S.gripper_speed * (1/60);
    if (readBtn(0)) gripDelta = -S.gripper_speed * (1/60);
  } else if (selectedMapping === 'drone') {
    lin[0] = -applyDeadzone(axes[4] || 0) * lspd;
    lin[1] = -applyDeadzone(axes[3] || 0) * lspd;
    lin[2] = -applyDeadzone(axes[1] || 0) * lspd;
    ang[2] =  applyDeadzone(axes[0] || 0) * aspd;
    // D-pad via buttons (hat): buttons 12-15 on standard layout: up=12, down=13, left=14, right=15
    if (readBtn(12)) ang[1] =  aspd;
    if (readBtn(13)) ang[1] = -aspd;
    if (readBtn(14)) ang[0] =  aspd;
    if (readBtn(15)) ang[0] = -aspd;
    // LB=4 open, RB=5 close
    if (readBtn(4)) gripDelta =  S.gripper_speed * (1/60);
    if (readBtn(5)) gripDelta = -S.gripper_speed * (1/60);
  }

  return { lin, ang, gripDelta };
}

// ============================================================
// INPUT LOOP (requestAnimationFrame)
// ============================================================
function inputLoop() {
  requestAnimationFrame(inputLoop);
  if (!wsReady) return;

  let lin = [0, 0, 0];
  let ang = [0, 0, 0];
  let gripDelta = 0;

  if (selectedInputMethod === 'controller' && gamepadIndex !== null) {
    const gpads = navigator.getGamepads();
    const gp = gpads[gamepadIndex];
    if (gp && gp.connected) {
      const r = readGamepad(gp, S.l_spd, S.a_spd);
      lin = r.lin;
      ang = r.ang;
      gripDelta = r.gripDelta;
    }
  } else if (selectedInputMethod === 'keyboard' && !inputFocused) {
    if (keys['KeyW'])      lin[0] += S.l_spd;
    if (keys['KeyS'])      lin[0] -= S.l_spd;
    if (keys['KeyA'])      lin[1] += S.l_spd;
    if (keys['KeyD'])      lin[1] -= S.l_spd;
    if (keys['Space'])     lin[2] += S.l_spd;
    if (keys['ShiftLeft'] || keys['ShiftRight']) lin[2] -= S.l_spd;
    if (keys['ArrowLeft'])  ang[0] -= S.a_spd;
    if (keys['ArrowRight']) ang[0] += S.a_spd;
    if (keys['ArrowUp'])    ang[1] += S.a_spd;
    if (keys['ArrowDown'])  ang[1] -= S.a_spd;
    if (keys['KeyQ'])       ang[2] += S.a_spd;
    if (keys['KeyE'])       ang[2] -= S.a_spd;
  }

  // Gripper keys always active regardless of input method (O/P)
  if (!inputFocused) {
    if (keys['KeyO']) gripDelta =  S.gripper_speed * (1/60);
    if (keys['KeyP']) gripDelta = -S.gripper_speed * (1/60);
  }

  send({type:'input', lin, ang});

  if (Math.abs(gripDelta) > 1e-7) {
    send({type:'gripper_delta', delta: gripDelta});
  }
}

// ============================================================
// INIT
// ============================================================
buildKinRows();
buildMoveInputs();
updateLegend();

// Placeholder telemetry
document.getElementById('tel-cart').textContent =
  'X       0.0000 m\nY       0.0000 m\nZ       0.0000 m\nRoll    0.0000 rad\nPitch   0.0000 rad\nYaw     0.0000 rad';
document.getElementById('tel-joints').textContent =
  'q1      0.0000 rad\nq2      0.0000 rad\nq3      0.0000 rad\nq4      0.0000 rad\nq5      0.0000 rad\nq6      0.0000 rad';

connectWS();
requestAnimationFrame(inputLoop);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP ENDPOINT
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=HTML_PAGE)


# ---------------------------------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
