#!/usr/bin/env python3
import sys
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import tkinter as tk


class LanguagePublisherGUI(Node):
    def __init__(self):
        super().__init__('language_publisher_node')

        # Publish user prompt input
        self.pub = self.create_publisher(String, 'topic_prompt', 10)

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
        self.label = tk.Label(self.root, text='Write your input prompt or command for the robot.')
        self.label.pack(pady=10)

        # Ensure proper shutdown when window is closed
        self.root.protocol('WM_DELETE_WINDOW', self.on_close)

    def send_text(self):
        text = self.entry.get()
        if text:
            self.get_logger().info(f'Publishing: {text}')
            msg = String()
            msg.data = text
            self.pub.publish(msg)

            self.label.config(text=f'User last input prompt: {text}')
            self.entry.delete(0, tk.END)

    def on_close(self):
        # stop tkinter loop
        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass

    def run(self):
        self.root.mainloop()


def main(args=None):
    rclpy.init(args=args)
    node = LanguagePublisherGUI()

    # Spin ROS in a background thread so tkinter mainloop can run
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        node.run()  # tkinter loop
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
