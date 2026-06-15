#!/usr/bin/env python3
import sys
import rclpy
from std_msgs.msg import String
import tkinter as tk

class LanguagePublisherGUI:
    def __init__(self):
        rclpy.init()
        self.node = rclpy.create_node('language_publisher_node')

        # Publish user prompt input
        self.pub = self.node.create_publisher(String, 'topic_prompt', 10)

        # Subscribe to the model answer
        self.sub = self.node.create_subscription(String, 'topic_vlm_answer', self.cb_answer, 10)

        # Create window
        self.root = tk.Tk()
        self.root.title('Language Input')

        # Text box
        self.entry = tk.Entry(self.root, width=40)
        self.entry.pack(pady=10)

        # Send button
        self.button = tk.Button(self.root, text='Send', command=self.send_text)
        self.button.pack(pady=5)

        # Message for the user
        self.label = tk.Label(self.root, text='Write the object you want to pick')
        self.label.pack(pady=10)

        # Print model's answer
        self.label_answer = tk.Label(self.root, text='Assistant answer:')
        self.label_answer.pack(pady=10)

        # Ensure proper shutdown when window is closed
        self.root.protocol('WM_DELETE_WINDOW', self.on_close)

    def send_text(self):
        text = self.entry.get()
        if text:
            # If the user input is not empty, publish it on the topic
            self.node.get_logger().info(f'Publishing: {text}')
            msg = String()
            msg.data = text
            self.pub.publish(msg)
            # Edit the label (what the user sees) to show the last message sent
            self.label.config(text=f'Selected object: {text}')
            self.entry.delete(0, tk.END)

    def cb_answer(self, msg):
        # Callback if the model has an answer
        self.node.get_logger().info(f'[GUI] Model answer received: {msg.data}')
        # Update Tkinter widget (callbacks run on the same thread because we poll with spin_once)
        self.label_answer.config(text=f'{msg.data}')

    def _poll_ros(self):
        # Poll rclpy to allow callbacks to run while Tkinter mainloop is active
        try:
            rclpy.spin_once(self.node, timeout_sec=0.01)
        except Exception:
            pass
        self.root.after(10, self._poll_ros)

    def on_close(self):
        try:
            self.node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass
        self.root.destroy()

    def run(self):
        self._poll_ros()
        self.root.mainloop()

if __name__ == '__main__':
    gui = LanguagePublisherGUI()
    try:
        gui.run()
    except KeyboardInterrupt:
        gui.on_close()
        sys.exit(0)
