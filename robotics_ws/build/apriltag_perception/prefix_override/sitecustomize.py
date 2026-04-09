import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/yoheen/Robotic-Space-Hand-Algorithm/robotics_ws/install/apriltag_perception'
