import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
import socket
import json
import threading

# All 21 hand landmark names matching pose_detect.py
HAND_LANDMARKS = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "IDX_MCP",   "IDX_PIP",   "IDX_DIP",  "IDX_TIP",
    "MID_MCP",   "MID_PIP",   "MID_DIP",  "MID_TIP",
    "RNG_MCP",   "RNG_PIP",   "RNG_DIP",  "RNG_TIP",
    "PKY_MCP",   "PKY_PIP",   "PKY_DIP",  "PKY_TIP"
]

class PosePublisher(Node):
    def __init__(self):
        super().__init__('pose_publisher')

        # ── Arm joint publishers (unchanged) ──────────────────────
        self.shoulder_pub = self.create_publisher(PointStamped, 'human/shoulder', 10)
        self.elbow_pub    = self.create_publisher(PointStamped, 'human/elbow',    10)
        self.wrist_pub    = self.create_publisher(PointStamped, 'human/wrist',    10)

        # ── Hand landmark publishers (one per node) ───────────────
        # Topics will be: human/hand/WRIST, human/hand/IDX_TIP, etc.
        self.hand_pubs = {
            name: self.create_publisher(PointStamped, f'human/hand/{name}', 10)
            for name in HAND_LANDMARKS
        }

        # ── UDP socket ────────────────────────────────────────────
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", 5006))
        self.sock.settimeout(1.0)

        self.get_logger().info('Pose publisher waiting for data...')
        self.get_logger().info(f'  Arm topics  : human/shoulder, human/elbow, human/wrist')
        self.get_logger().info(f'  Hand topics : human/hand/[LANDMARK] x{len(HAND_LANDMARKS)}')

        self.running = True
        self.thread = threading.Thread(target=self.listen_loop, daemon=True)
        self.thread.start()

    def listen_loop(self):
        while self.running:
            try:
                data, _ = self.sock.recvfrom(4096)   # bumped from 2048 — hand data is larger
                payload = json.loads(data.decode())
                self.publish_joints(payload)
            except socket.timeout:
                continue
            except Exception as e:
                self.get_logger().warn(f'Error: {e}')

    def make_point(self, stamp, frame_id, x, y, z):
        """Helper — build a PointStamped message."""
        msg = PointStamped()
        msg.header.stamp    = stamp
        msg.header.frame_id = frame_id
        msg.point.x = float(x)
        msg.point.y = float(y)
        msg.point.z = float(z)
        return msg

    def publish_joints(self, payload):
        stamp = self.get_clock().now().to_msg()

        # ── Arm joints ────────────────────────────────────────────
        arm = payload.get('arm', {})
        if arm:
            for joint_name, pub in [
                ('shoulder', self.shoulder_pub),
                ('elbow',    self.elbow_pub),
                ('wrist',    self.wrist_pub)
            ]:
                if joint_name in arm:
                    pub.publish(self.make_point(
                        stamp, 'camera',
                        arm[joint_name]['x'],
                        arm[joint_name]['y'],
                        arm[joint_name]['z']
                    ))

            self.get_logger().info(
                f"[ARM]  W → X:{arm.get('wrist',{}).get('x',0):.3f} "
                f"Y:{arm.get('wrist',{}).get('y',0):.3f} "
                f"Z:{arm.get('wrist',{}).get('z',0):.3f}"
            )

        # ── Hand landmarks ────────────────────────────────────────
        hand = payload.get('hand', {})
        if hand:
            for name, pub in self.hand_pubs.items():
                if name in hand:
                    pub.publish(self.make_point(
                        stamp, 'camera',
                        hand[name]['x'],
                        hand[name]['y'],
                        hand[name]['z']
                    ))

            # Log just the fingertips so terminal isn't flooded
            tips = {k: hand[k] for k in ['IDX_TIP','MID_TIP','RNG_TIP','PKY_TIP','THUMB_TIP'] if k in hand}
            self.get_logger().info(f"[HAND] Tips → {tips}")

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