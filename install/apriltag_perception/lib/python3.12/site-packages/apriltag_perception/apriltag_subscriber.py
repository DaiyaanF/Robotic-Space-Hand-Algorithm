import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32

class AprilTagSubscriber(Node):
    def __init__(self):
        super().__init__('apriltag_subscriber')

        self.pose_sub = self.create_subscription(
            PoseStamped,
            'apriltag/pose',
            self.pose_callback,
            10
        )

        self.id_sub = self.create_subscription(
            Int32,
            'apriltag/id',
            self.id_callback,
            10
        )

        self.get_logger().info('AprilTag subscriber node started')

    def pose_callback(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        self.get_logger().info(f'Received pose — X:{x:.3f} Y:{y:.3f} Z:{z:.3f}m')

    def id_callback(self, msg):
        self.get_logger().info(f'Received tag ID: {msg.data}')


def main(args=None):
    rclpy.init(args=args)
    node = AprilTagSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()