#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import time
import os
import subprocess


class TaiChiInteractionNode(Node):
    def __init__(self):
        super().__init__('tai_chi_interaction')
        
        self.declare_parameter('trigger_topic', '/trigger_capture')
        self.declare_parameter('result_topic', '/pose_result')
        self.declare_parameter('arm_topic', '/arm_controller/joint_trajectory')
        self.declare_parameter('audio_dir', 'audio')
        self.declare_parameter('use_audio', True)
        self.declare_parameter('teach_all', True) # was using this for testing each part seperately but can take it out now
        
        self.trigger_topic = self.get_parameter('trigger_topic').value
        self.result_topic = self.get_parameter('result_topic').value
        self.arm_topic = self.get_parameter('arm_topic').value
        self.audio_dir = self.get_parameter('audio_dir').value
        self.use_audio = self.get_parameter('use_audio').value
        self.teach_all = self.get_parameter('teach_all').value
        
        self.get_logger().info("="*60)
        self.get_logger().info("Tai Chi Interaction Node Starting...")
        self.get_logger().info("="*60)
        
        self.trigger_pub = self.create_publisher(Bool, self.trigger_topic, 10)
        self.arm_pub = self.create_publisher(JointTrajectory, self.arm_topic, 10)
        
        self.result_sub = self.create_subscription(
            String,
            self.result_topic,
            self.result_callback,
            10
        )
        
        
        self.latest_result = None
        self.waiting_for_result = False
        
        
        self.sequence_parts = [
            {
                "name": "Part 1: Opening : Raise Hands",
                "intro_text": "Let's begin with the opening. Watch my arms rise slowly.",
                "poses": [
                    [0.0, -0.023, 0.003, 0.0],  # Starting
                    [0.0, 1.499, 0.209, 0.0],   # Arms up
                    [0.0, 1.496, -0.407, 0.0],  # Arms forward 
                ],
                "audio": {
                    "intro": "intro_part1.wav",
                    "countdown": "countdown_1234.wav",
                    "user_turn": "your_turn.wav",
                    "hold": "hold_pose_3sec.wav",
                    "correct": "correct_next.wav",
                    "incorrect": "incorrect_repeat.wav"
                }
            },
            {
                "name": "Part 2: Downward Flow",
                "intro_text": "Now we'll lower the arms with control.",
                "poses": [
                    [0.0, 1.496, -0.407, 0.0],  # Continue from part 1
                    [0.0, 1.116, -0.699, 0.0],
                    [0.0, 0.545, -0.897, 0.0],
                    [0.0, 0.545, -1.197, 0.0], 
                ],
                "audio": {
                    "intro": "intro_part2.wav",
                    "countdown": "countdown_1234.wav",
                    "user_turn": "your_turn.wav",
                    "hold": "hold_pose_3sec.wav",
                    "correct": "correct_next.wav",
                    "incorrect": "incorrect_repeat.wav"
                }
            },
            {
                "name": "Part 3: Centering Movement",
                "intro_text": "This brings energy to your center.",
                "poses": [
                    [0.0, 0.545, -1.197, 0.0], 
                    [0.0, 0.003, -1.197, 0.0],
                    [0.0, -0.402, -1.197, 0.599],
                    [0.0, -0.402, -1.197, 0.0], 
                ],
                "audio": {
                    "intro": "intro_part3.wav",
                    "countdown": "countdown_1234.wav",
                    "user_turn": "your_turn.wav",
                    "hold": "hold_pose_3sec.wav",
                    "correct": "correct_done.wav",
                    "incorrect": "incorrect_repeat.wav"
                }
            }
        ]
        
        self.get_logger().info("="*60)
        self.get_logger().info("Interaction Node Ready!")
        self.get_logger().info(f"  Sequence has {len(self.sequence_parts)} parts")
        self.get_logger().info(f"  Audio enabled: {self.use_audio}")
        self.get_logger().info(f"  Teach all parts: {self.teach_all}")
        self.get_logger().info("="*60)
    
    def play_audio(self, filename, blocking=True):
        if not self.use_audio:
            self.get_logger().info(f"[Audio: {filename}]")
            return None
        
        audio_path = os.path.join(self.audio_dir, filename)
        
        if not os.path.exists(audio_path):
            self.get_logger().warn(f"Audio file not found: {audio_path}")
            self.get_logger().info(f"[Would play: {filename}]")
            return None
        
        try:
            self.get_logger().info(f"Playing: {filename}")
            
            if blocking:
                subprocess.run(['aplay', '-q', audio_path], check=True)
                return None
            else:
                process = subprocess.Popen(['aplay', '-q', audio_path])
                return process
                
        except Exception as e:
            self.get_logger().error(f"Error playing audio: {e}")
            return None
    
    def move_arm(self, joint_positions, duration_sec=1.0):
        msg = JointTrajectory()
        msg.joint_names = ['joint1', 'joint2', 'joint3', 'joint4']
        
        point = JointTrajectoryPoint()
        point.positions = joint_positions
        point.time_from_start = Duration(sec=int(duration_sec), nanosec=int((duration_sec % 1) * 1e9))
        
        msg.points = [point]
        
        self.arm_pub.publish(msg)
        self.get_logger().info(f"  Moving arm to: {joint_positions}")
    
    def result_callback(self, msg):
        self.latest_result = msg.data
        if self.waiting_for_result:
            self.get_logger().info(f"Received result: {msg.data}")
    
    def teach_part(self, part, part_num):
        audio = part["audio"]
        
        self.get_logger().info("\n" + "="*60)
        self.get_logger().info(f"{part['name']}")
        self.get_logger().info("="*60)
        
        # step 1 -- intro
        self.get_logger().info("\nStep 1: Robot Demonstrates")
        self.play_audio(audio["intro"])
        time.sleep(1.0)
        
        # step 2 -- robot demo with countdown
        self.get_logger().info("\nCounting through poses...")
        
        # starting countdown audio 
        self.play_audio(audio["countdown"], blocking=False)
        
        # execute poses while audio plays
        for i, pose in enumerate(part["poses"]):
            self.move_arm(pose, duration_sec=1.0)
            time.sleep(1.0)
        
        time.sleep(1.0)
        
        # step 3 -- user's turn
        self.get_logger().info("\nYour Turn")
        
        self.play_audio(audio["user_turn"])
        time.sleep(1.0)
        
        # user countdown
        self.get_logger().info("Follow along: 1, 2, 3, 4...")
        self.play_audio(audio["countdown"], blocking=False)
        time.sleep(len(part["poses"]))  # One second per pose
        
        # step 4 -- hold final pose & capture
        self.get_logger().info("\nEvaluating Your Final Pose")
        
        self.play_audio(audio["hold"])
        time.sleep(0.5)
        
        # Trigger capture
        self.get_logger().info("Triggering pose evaluation...")
        self.latest_result = None
        self.waiting_for_result = True
        
        trigger_msg = Bool()
        trigger_msg.data = True
        self.trigger_pub.publish(trigger_msg)
        
        # Wait for result
        self.get_logger().info("Waiting 3 seconds for evaluation...")
        wait_start = time.time()
        while (time.time() - wait_start) < 3.0:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        self.waiting_for_result = False
        
        # step 5 -- feedback
        self.get_logger().info("\nFeedback")
        
        if self.latest_result == "correct":
            self.get_logger().info("Your final pose is CORRECT!")
            self.play_audio(audio["correct"])
            success = True
        elif self.latest_result == "incorrect":
            self.get_logger().info("Your final pose needs work")
            self.play_audio(audio["incorrect"])
            success = False
        elif self.latest_result == "no_person":
            self.get_logger().warn("No person detected")
            success = False
        else:
            self.get_logger().warn(f"Unexpected result: {self.latest_result}")
            success = False
        
        time.sleep(2.0)
        
        return success
    
    def run_teaching_session(self):
        """Run the complete teaching session"""
        self.get_logger().info("\nStarting Tai Chi Teaching Session!\n")
        
        if self.teach_all:
            for i, part in enumerate(self.sequence_parts):
                self.get_logger().info(f"\n\n{'='*60}")
                self.get_logger().info(f"TEACHING PART {i+1} of {len(self.sequence_parts)}")
                self.get_logger().info(f"{'='*60}\n")
                
                self.teach_part(part, i+1)
                
                if i < len(self.sequence_parts) - 1:
                    self.get_logger().info("\nBrief pause before next part...")
                    time.sleep(3.0)
            
            self.get_logger().info("\n\n" + "="*60)
            self.get_logger().info("Complete Sequence Finished!")
            self.get_logger().info("="*60)
        else:
            pass


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = TaiChiInteractionNode()
        
        time.sleep(2.0)
        node.run_teaching_session()
        time.sleep(2.0)
        
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
