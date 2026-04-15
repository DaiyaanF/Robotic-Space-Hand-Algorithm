
import csv
import os
import threading
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped

# Must match pose_publisher.py exactly
HAND_LANDMARKS = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "IDX_MCP",   "IDX_PIP",   "IDX_DIP",  "IDX_TIP",
    "MID_MCP",   "MID_PIP",   "MID_DIP",  "MID_TIP",
    "RNG_MCP",   "RNG_PIP",   "RNG_DIP",  "RNG_TIP",
    "PKY_MCP",   "PKY_PIP",   "PKY_DIP",  "PKY_TIP",
]

ARM_JOINTS = ["shoulder", "elbow", "wrist"]

GESTURES = ["point", "stop", "assist", "grab", "hand_over"]
 
# Snapshot timer frequency — matches pose_publisher ~30 fps
SAMPLE_RATE_HZ = 30

# CSV column order (hand landmarks first, then arm joints)
COLUMNS = (
    ["timestamp", "label", "session_id"]
    + [f"{lm}_{ax}" for lm in HAND_LANDMARKS for ax in ("x", "y", "z")]
    + [f"{jt}_{ax}" for jt in ARM_JOINTS for ax in ("x", "y", "z")]
)


class GestureCollector(Node):
    def __init__(self):
        super().__init__("gesture_collector")

        # ── Parameter: output directory ───────────────────────────
        self.declare_parameter("output_dir", os.path.expanduser("~/gesture_data"))
        output_dir = (
            self.get_parameter("output_dir").get_parameter_value().string_value
        )
        os.makedirs(output_dir, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(output_dir, f"gestures_{ts}.csv")

        self._csv_file = open(csv_path, "w", newline="")
        self._writer = csv.DictWriter(self._csv_file, fieldnames=COLUMNS)
        self._writer.writeheader()
        self._csv_file.flush()

        self.get_logger().info(f"Output: {csv_path}")

        # ── Shared state ──────────────────────────────────────────
        # _latest holds the most recent (x, y, z) for each key.
        # Keys: HAND_LANDMARK names  +  ARM_JOINT names.
        self._latest: dict[str, tuple[float, float, float]] = {}
        self._lock = threading.Lock()

        self._recording = False
        self._current_label = ""
        self._session_id = 0
        self._frame_count = 0

        # ── Subscribers: 21 hand landmarks ───────────────────────
        for name in HAND_LANDMARKS:
            self.create_subscription(
                PointStamped,
                f"human/hand/{name}",
                lambda msg, n=name: self._update(n, msg),
                10,
            )

        # ── Subscribers: 3 arm joints ─────────────────────────────
        for joint in ARM_JOINTS:
            self.create_subscription(
                PointStamped,
                f"human/{joint}",
                lambda msg, j=joint: self._update(j, msg),
                10,
            )

        # ── Snapshot timer ────────────────────────────────────────
        self._timer = self.create_timer(1.0 / SAMPLE_RATE_HZ, self._snapshot_cb)

        # ── Interactive CLI (background thread) ───────────────────
        self._cli_thread = threading.Thread(target=self._cli_loop, daemon=True)
        self._cli_thread.start()

        self.get_logger().info(
            f"Gestures: {', '.join(GESTURES)}"
        )

    # ── Topic callbacks ───────────────────────────────────────────

    def _update(self, key: str, msg: PointStamped) -> None:
        with self._lock:
            self._latest[key] = (msg.point.x, msg.point.y, msg.point.z)

    # ── Snapshot timer callback ───────────────────────────────────

    def _snapshot_cb(self) -> None:
        with self._lock:
            if not self._recording:
                return

            # Wait until all hand landmarks have arrived at least once
            if not all(lm in self._latest for lm in HAND_LANDMARKS):
                return

            row: dict = {
                "timestamp": time.time(),
                "label": self._current_label,
                "session_id": self._session_id,
            }

            for lm in HAND_LANDMARKS:
                x, y, z = self._latest[lm]
                row[f"{lm}_x"] = round(x, 6)
                row[f"{lm}_y"] = round(y, 6)
                row[f"{lm}_z"] = round(z, 6)

            for jt in ARM_JOINTS:
                # Arm joints are optional — default to 0.0 if not published
                x, y, z = self._latest.get(jt, (0.0, 0.0, 0.0))
                row[f"{jt}_x"] = round(x, 6)
                row[f"{jt}_y"] = round(y, 6)
                row[f"{jt}_z"] = round(z, 6)

            self._writer.writerow(row)
            self._frame_count += 1

    # ── Interactive CLI ───────────────────────────────────────────

    def _cli_loop(self) -> None:
        """
        Runs in a background thread.  Prompts the user to enter a
        gesture label, then records until ENTER is pressed again.
        """
        print("\n" + "=" * 52)
        print("  Gesture Collector  —  Phase 2 Data Collection")
        print("=" * 52)
        print(f"  Gestures : {', '.join(GESTURES)}")
        print("  Commands : <gesture_name> | list | quit")
        print("=" * 52 + "\n")

        while rclpy.ok():
            try:
                raw = input("Gesture label: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                break

            if not raw:
                continue

            if raw == "quit":
                self.get_logger().info("Shutting down.")
                rclpy.shutdown()
                break

            if raw == "list":
                print(f"  Available: {', '.join(GESTURES)}")
                continue

            if raw not in GESTURES:
                print(
                    f"  Unknown: '{raw}'.  "
                    f"Available: {', '.join(GESTURES)}"
                )
                continue

            # ── Start recording ───────────────────────────────────
            with self._lock:
                self._session_id += 1
                self._current_label = raw
                self._frame_count = 0
                self._recording = True

            print(
                f"  [REC] '{raw}'  session={self._session_id}  "
                f"— press ENTER to stop ..."
            )

            try:
                input()
            except (EOFError, KeyboardInterrupt):
                pass

            # ── Stop recording ────────────────────────────────────
            with self._lock:
                self._recording = False
                frames = self._frame_count

            self._csv_file.flush()
            print(f"  [STOP] Saved {frames} frames for '{raw}'\n")

    # ── Cleanup ───────────────────────────────────────────────────

    def destroy_node(self) -> None:
        self._csv_file.flush()
        self._csv_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GestureCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
