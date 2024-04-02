import cv2

class VideoFormats:
    def __init__(self,video_path):
        self.video_path = video_path

    def extract_frames(self, output_folder):
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            print(f"Error: Couldn't open the video file at: {self.video_path}")

        fps = 0