import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
import socket
import json
import threading
import numpy as np
from filterpy.kalman import KalmanFilter

HAND_LANDMARKS = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "IDX_MCP",   "IDX_PIP",   "IDX_DIP",  "IDX_TIP",
    "MID_MCP",   "MID_PIP",   "MID_DIP",  "MID_TIP",
    "RNG_MCP",   "RNG_PIP",   "RNG_DIP",  "RNG_TIP",
    "PKY_MCP",   "PKY_PIP",   "PKY_DIP",  "PKY_TIP"
]

def make_kalman(dt=0.033):
    """
    Build a 6-state, 3-measurement Kalman filter for one joint.

    State vector:  [x, y, z, vx, vy, vz]
    Measurement:   [x, y, z]  (raw MediaPipe output)

    dt = time between frames (~0.033s = 30 fps)
    """
    kf = KalmanFilter(dim_x=6, dim_z=3)

    # ── State transition (F): assumes constant velocity ──────────
    # x_new = x + vx*dt,  vx_new = vx  (same for y, z)
    kf.F = np.array([
        [1, 0, 0, dt, 0,  0 ],
        [0, 1, 0, 0,  dt, 0 ],
        [0, 0, 1, 0,  0,  dt],
        [0, 0, 0, 1,  0,  0 ],
        [0, 0, 0, 0,  1,  0 ],
        [0, 0, 0, 0,  0,  1 ],
    ], dtype=float)

    # ── Measurement function (H): we only observe position, not velocity
    kf.H = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
    ], dtype=float)

    # ── Measurement noise (R): how much we trust MediaPipe ───────
    # Higher = more smoothing, more lag.  Lower = more jitter.
    kf.R = np.eye(3) * 0.01

    # ── Process noise (Q): how much the motion model can drift ───
    # Higher = filter reacts faster to real movement.
    kf.Q = np.eye(6) * 0.001

    # ── Initial covariance (P): high uncertainty at start ────────
    kf.P = np.eye(6) * 1.0

    # ── Initial state: start at origin, zero velocity ────────────
    kf.x = np.zeros((6, 1))

    return kf


class PosePublisher(Node):
    def __init__(self):
        super().__init__('pose_publisher')

        # ── Arm publishers ────────────────────────────────────────
        self.shoulder_pub = self.create_publisher(PointStamped, 'human/shoulder', 10)
        self.elbow_pub    = self.create_publisher(PointStamped, 'human/elbow',    10)
        self.wrist_pub    = self.create_publisher(PointStamped, 'human/wrist',    10)

        # ── Hand publishers ───────────────────────────────────────
        self.hand_pubs = {
            name: self.create_publisher(PointStamped, f'human/hand/{name}', 10)
            for name in HAND_LANDMARKS
        }

        # ── Kalman filters: one per arm joint ─────────────────────
        self.kf = {
            'shoulder': make_kalman(),
            'elbow':    make_kalman(),
            'wrist':    make_kalman(),
        }

        # ── Kalman filters: one per hand landmark ─────────────────
        self.hand_kf = {
            name: make_kalman()
            for name in HAND_LANDMARKS
        }

        # ── Track whether a filter has been seeded ────────────────
        # On the very first measurement we set the filter state
        # directly instead of running a predict/update cycle,
        # so it doesn't snap from (0,0,0) to the real position.
        self.kf_initialized    = {k: False for k in self.kf}
        self.hand_kf_initialized = {k: False for k in self.hand_kf}

        # ── UDP socket ────────────────────────────────────────────
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", 5006))
        self.sock.settimeout(1.0)

        self.get_logger().info('Pose publisher ready — Kalman filter active')
        self.get_logger().info(f'  Arm  : human/shoulder, human/elbow, human/wrist')
        self.get_logger().info(f'  Hand : human/hand/[LANDMARK] x{len(HAND_LANDMARKS)}')

        self.running = True
        self.thread = threading.Thread(target=self.listen_loop, daemon=True)
        self.thread.start()

    # ── Kalman helper ─────────────────────────────────────────────
    def kalman_smooth(self, kf, initialized_flag, x, y, z):
        """
        Feed one raw measurement into a filter, return smoothed (x, y, z).
        On the first call we seed the state so there's no jump from origin.
        """
        measurement = np.array([[x], [y], [z]], dtype=float)

        if not initialized_flag:
            # Seed position directly; velocity stays zero
            kf.x[0] = x
            kf.x[1] = y
            kf.x[2] = z
            return x, y, z          # return raw on first frame

        kf.predict()                # step the motion model forward
        kf.update(measurement)      # blend in the new measurement

        sx, sy, sz = float(kf.x[0]), float(kf.x[1]), float(kf.x[2])
        return sx, sy, sz

    # ── UDP loop ──────────────────────────────────────────────────
    def listen_loop(self):
        while self.running:
            try:
                data, _ = self.sock.recvfrom(4096)
                payload = json.loads(data.decode())
                self.publish_joints(payload)
            except socket.timeout:
                continue
            except Exception as e:
                self.get_logger().warn(f'Error: {e}')

    def make_point(self, stamp, frame_id, x, y, z):
        msg = PointStamped()
        msg.header.stamp    = stamp
        msg.header.frame_id = frame_id
        msg.point.x = float(x)
        msg.point.y = float(y)
        msg.point.z = float(z)
        return msg

    # ── Publish ───────────────────────────────────────────────────
    def publish_joints(self, payload):
        stamp = self.get_clock().now().to_msg()

        # ── Arm joints ────────────────────────────────────────────
        arm = payload.get('arm', {})
        if arm:
            for joint_name, pub in [
                ('shoulder', self.shoulder_pub),
                ('elbow',    self.elbow_pub),
                ('wrist',    self.wrist_pub),
            ]:
                if joint_name not in arm:
                    continue

                raw = arm[joint_name]

                # Smooth with Kalman
                sx, sy, sz = self.kalman_smooth(
                    self.kf[joint_name],
                    self.kf_initialized[joint_name],
                    raw['x'], raw['y'], raw['z']
                )
                self.kf_initialized[joint_name] = True

                pub.publish(self.make_point(stamp, 'camera', sx, sy, sz))

            self.get_logger().info(
                f"[ARM]  W → "
                f"raw({arm.get('wrist',{}).get('x',0):.3f}, "
                f"{arm.get('wrist',{}).get('y',0):.3f})  "
                f"smooth({float(self.kf['wrist'].x[0]):.3f}, "
                f"{float(self.kf['wrist'].x[1]):.3f})"
            )

        # ── Hand landmarks ────────────────────────────────────────
        hand = payload.get('hand', {})
        if hand:
            for name, pub in self.hand_pubs.items():
                if name not in hand:
                    continue

                raw = hand[name]

                sx, sy, sz = self.kalman_smooth(
                    self.hand_kf[name],
                    self.hand_kf_initialized[name],
                    raw['x'], raw['y'], raw['z']
                )
                self.hand_kf_initialized[name] = True

                pub.publish(self.make_point(stamp, 'camera', sx, sy, sz))

            tips = ['IDX_TIP', 'MID_TIP', 'RNG_TIP', 'PKY_TIP', 'THUMB_TIP']
            tip_log = {
                k: f"({float(self.hand_kf[k].x[0]):.3f}, {float(self.hand_kf[k].x[1]):.3f})"
                for k in tips if k in hand
            }
            self.get_logger().info(f"[HAND] Tips → {tip_log}")

    def destroy_node(self):
        self.running = False
        self.sock.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PosePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()