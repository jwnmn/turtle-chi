import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import os
import time

from mlp_loader import load_mlp_model, extract_features


class TaiChiPoseNode(Node):
    def __init__(self):
        super().__init__('tai_chi_pose_node')
        
        self.declare_parameter('camera_topic', '/tb07/oakd/rgb/preview/image_raw') # i am #lazy
        self.declare_parameter('trigger_topic', '/trigger_capture')
        self.declare_parameter('result_topic', '/pose_result')
        self.declare_parameter('model_path', 'models/ver3/movement_1_mlp_scratch_ver3.npz')
        self.declare_parameter('scaler_path', 'models/ver3/movement_1_scaler_scratch_ver3.npz')
        self.declare_parameter('min_keypoint_confidence', 0.3)

        self.camera_topic = self.get_parameter('camera_topic').value
        self.trigger_topic = self.get_parameter('trigger_topic').value
        self.result_topic = self.get_parameter('result_topic').value
        self.model_path = self.get_parameter('model_path').value
        self.scaler_path = self.get_parameter('scaler_path').value
        self.min_keypoint_confidence = self.get_parameter('min_keypoint_confidence').value
        
        self.get_logger().info("="*60)
        self.get_logger().info("Tai Chi Pose Estimation Node Starting...")
        self.get_logger().info("="*60)
        
        
        self.bridge = CvBridge()
        
        
        self.get_logger().info("Loading MoveNet...")
        self.load_movenet()
        self.get_logger().info(f"MoveNet loaded (input size: {self.input_size})")
        
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.get_logger().info(f"Loading binary classifier...")
            self.model, self.feature_mean, self.feature_std = load_mlp_model(
                self.model_path,
                self.scaler_path,
                input_dim=43,
                hidden_dims=(64, 32)
            )
            self.get_logger().info(f"Binary Classifier loaded")
            self.get_logger().info(f"  Architecture: 43 to 64 to 32 to 1")
            self.get_logger().info(f"  Evaluates ANY pose as correct/incorrect")
        else:
            self.get_logger().error(f"Model files not found!")
            self.get_logger().error(f"  Model: {self.model_path}")
            self.get_logger().error(f"  Scaler: {self.scaler_path}")
            raise FileNotFoundError("Model or scaler file not found")
        
        self.latest_image = None
        self.capture_ready = False
        
        # Result publisher
        self.result_pub = self.create_publisher(String, self.result_topic, 10)
        
        # Camera subscriber
        self.camera_sub = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            10
        )
        
        # Trigger subscriber
        self.trigger_sub = self.create_subscription(
            Bool,
            self.trigger_topic,
            self.trigger_callback,
            10
        )
        
        self.get_logger().info("="*60)
        self.get_logger().info("Pose Estimation Node Ready!")
        self.get_logger().info(f"  Camera: {self.camera_topic}")
        self.get_logger().info(f"  Trigger: {self.trigger_topic}")
        self.get_logger().info(f"  Result: {self.result_topic}")
        self.get_logger().info("  Waiting for trigger...")
        self.get_logger().info("="*60)
    
    def load_movenet(self):
        module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
        self.input_size = 192
        
        def movenet(input_image):
            model = module.signatures['serving_default']
            input_image = tf.cast(input_image, dtype=tf.int32)
            outputs = model(input_image)
            return outputs['output_0'].numpy()
        
        self.movenet = movenet
    
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_image = cv_image
            self.capture_ready = True
        except Exception as e:
            self.get_logger().error(f"Error receiving image: {e}")
    
    def trigger_callback(self, msg):
        if not msg.data:
            return
        
        self.get_logger().info("TRIGGER RECEIVED - Capturing pose...")
        
        if not self.capture_ready or self.latest_image is None:
            self.get_logger().warn("No image available yet!")
            result_msg = String()
            result_msg.data = "no_person"
            self.result_pub.publish(result_msg)
            return
        
        try:
            cv_image = self.latest_image.copy()
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            input_image = tf.expand_dims(rgb_image, axis=0)
            input_image = tf.image.resize_with_pad(
                input_image,
                self.input_size,
                self.input_size
            )
            
            keypoints_with_scores = self.movenet(input_image)
            
            avg_conf = np.mean(keypoints_with_scores[0, 0, :, 2])
            if avg_conf < self.min_keypoint_confidence:
                self.get_logger().warn(f"No person detected (conf: {avg_conf:.2f})")
                result_msg = String()
                result_msg.data = "no_person"
                self.result_pub.publish(result_msg)
                return
            
            # Extract keypoints [17, 3] - (y, x, confidence)
            kpts = keypoints_with_scores[0, 0]
            
            # Convert to [17, 3] with (x, y, confidence) for feature extraction
            kpts_reordered = np.zeros_like(kpts)
            kpts_reordered[:, 0] = kpts[:, 1]  # x
            kpts_reordered[:, 1] = kpts[:, 0]  # y
            kpts_reordered[:, 2] = kpts[:, 2]  # confidence
            
            # Extract features (43 features)
            features = extract_features(kpts_reordered)
            
            # Normalize features
            features_norm = (features - self.feature_mean.flatten()) / self.feature_std.flatten()
            
            # Predict (binary: 1=correct, 0=incorrect)
            prediction = self.model.predict(features_norm.reshape(1, -1))[0]
            
            # Get probability
            prob = self.model.forward(features_norm.reshape(1, -1))[0, 0]
            
            # Publish result
            result_msg = String()
            if prediction == 1:
                result_msg.data = "correct"
                self.get_logger().info(f" CORRECT (confidence: {prob:.3f})")
            else:
                result_msg.data = "incorrect"
                self.get_logger().info(f"  INCORRECT (confidence: {1-prob:.3f})")
            
            self.result_pub.publish(result_msg)
            self.get_logger().info(f"  Keypoint avg confidence: {avg_conf:.3f}")
            
        except Exception as e:
            self.get_logger().error(f"Error classifying pose: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            
            result_msg = String()
            result_msg.data = "error"
            self.result_pub.publish(result_msg)


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = TaiChiPoseNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
