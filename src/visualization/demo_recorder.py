# Recorder that can capture the pygame surface to an AVI file (requires opencv)
import cv2
import numpy as np
import os


class Recorder:
    def __init__(self, out_path='logs/demo.avi', fps=30, size=(288,512)):
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.writer = cv2.VideoWriter(out_path, fourcc, fps, size)

    def write_frame(self, surface):
        # surface: pygame.Surface
        arr = __import__('pygame').surfarray.array3d(surface)
        # convert from (w,h,3) with x right, y down to image orientation
        frame = np.rot90(arr, k=3)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.writer.write(frame)

    def close(self):
        self.writer.release()
