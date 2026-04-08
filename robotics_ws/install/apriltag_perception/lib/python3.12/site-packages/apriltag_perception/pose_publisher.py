import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
import socket
import json
import threading

class PosePublisher(Node):
    def __init__(self):
        super().__init__('pose_publisher')

        # One publisher per joint
        self.shoulder_pub = self.create_publisher(PointStamped, 'human/shoulder', 10)
        self.elbow_pub    = self.create_publisher(PointStamped, 'human/elbow', 10)
        self.wrist_pub    = self.create_publisher(PointStamped, 'human/wrist', 10)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", 5006))
        self.sock.settimeout(1.0)

        self.get_logger().info('Pose publisher waiting for data...')

        self.running = True
        self.thread = threading.Thread(target=self.listen_loop, daemon=True)
        self.thread.start()

    def listen_loop(self):
        while self.running:
            try:
                data, _ = self.sock.recvfrom(2048)
                payload = json.loads(data.decode())
                self.publish_joints(payload)
            except socket.timeout:
                continue
            except Exception as e:
                self.get_logger().warn(f'Error: {e}')

    def publish_joints(self, payload):
        stamp = self.get_clock().now().to_msg()

        for joint_name, pub in [
            ('shoulder', self.shoulder_pub),
            ('elbow',    self.elbow_pub),
            ('wrist',    self.wrist_pub)
        ]:
            msg = PointStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = 'camera'
            msg.point.x = payload[joint_name]['x']
            msg.point.y = payload[joint_name]['y']
            msg.point.z = payload[joint_name]['z']
            pub.publish(msg)

        self.get_logger().info(
            f"Wrist → X:{payload['wrist']['x']:.3f} "
            f"Y:{payload['wrist']['y']:.3f} "
            f"Z:{payload['wrist']['z']:.3f}"
        )

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