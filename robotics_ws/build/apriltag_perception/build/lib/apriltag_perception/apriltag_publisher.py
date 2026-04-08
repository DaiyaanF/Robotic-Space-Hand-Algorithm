import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32
import socket
import json
import threading

class AprilTagPublisher(Node):
    def __init__(self):
        super().__init__('apriltag_publisher')

        self.pose_pub = self.create_publisher(PoseStamped, 'apriltag/pose', 10)
        self.id_pub = self.create_publisher(Int32, 'apriltag/id', 10)

        # UDP socket listening for data from Windows
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", 5005))
        self.sock.settimeout(1.0)

        self.get_logger().info('Waiting for AprilTag data from Windows bridge...')

        # Run socket listener in background thread
        self.running = True
        self.thread = threading.Thread(target=self.listen_loop, daemon=True)
        self.thread.start()

    def listen_loop(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(1024)
                print(f"[DEBUG] Raw packet from {addr}: {data.decode()}")  # add this
                payload = json.loads(data.decode())
                self.publish_detection(payload)
            except socket.timeout:
                continue
            except Exception as e:
                self.get_logger().warn(f'Socket error: {e}')


    def publish_detection(self, payload):
        # Publish ID
        id_msg = Int32()
        id_msg.data = payload["tag_id"]
        self.id_pub.publish(id_msg)

        # Publish pose
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'camera'
        pose_msg.pose.position.x = payload["x"]
        pose_msg.pose.position.y = payload["y"]
        pose_msg.pose.position.z = payload["z"]
        self.pose_pub.publish(pose_msg)

        self.get_logger().info(
            f'Tag {payload["tag_id"]} | '
            f'X:{payload["x"]:.3f} Y:{payload["y"]:.3f} Z:{payload["z"]:.3f}m'
        )

    def destroy_node(self):
        self.running = False
        self.sock.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = AprilTagPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()