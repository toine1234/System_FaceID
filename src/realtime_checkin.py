from collections import defaultdict, deque
import time
import numpy as np
from typing import List, Tuple

class RealtimeCheckIn:
    """
    Hỗ trợ điểm danh realtime dựa trên ArcFaceRecognizerONNX
    SỬA: Batch process, bbox center key cho track face, hysteresis cho status.
    """
    def __init__(self, recognizer, required_frames=2, reset_time=10):
        self.recognizer = recognizer
        self.required_frames = required_frames
        self.reset_time = reset_time
        self.threshold = recognizer.threshold  # Thống nhất

        # Buffer: {grid_key: deque(scores)} - dùng center rounded để track same face
        self.frame_buffers = defaultdict(lambda: deque(maxlen=self.required_frames))
        self.last_checked = {}  # label -> last_time
        self.checked_in = set()  # Đã check-in

        # Hysteresis cho status switch
        self.prev_status = defaultdict(lambda: "unknown")
        self.hysteresis_delta = 0.05  # Thresh thấp hơn nếu prev "pending"

    def _get_robust_key(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        x1, y1, x2, y2 = bbox
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        grid_size = 20  # Round jitter
        return (center_x // grid_size * grid_size, center_y // grid_size * grid_size)

    def process_frame(self, aligned_faces: List[Tuple[np.ndarray, Tuple[int, int, int, int]]]) -> List[Tuple[str, float, str]]:
        """
        aligned_faces: List[(face_bgr, bbox)]
        Trả về List[(label, score, status)] với status='new' nếu vừa check-in.
        """
        if not aligned_faces:
            return []

        face_bgr_list = [face for face, _ in aligned_faces]
        bboxes = [bbox for _, bbox in aligned_faces]

        # SỬA: Batch recognize
        results = self.recognizer.recognize_faces(aligned_faces)  # Giả sử recognize_faces hỗ trợ tuples
        batch_labels = [r[0] for r in results]
        batch_scores = [r[1] for r in results]

        current_time = time.time()
        output_results = []

        for i, (label, score) in enumerate(zip(batch_labels, batch_scores)):
            bbox = bboxes[i]
            grid_key = self._get_robust_key(bbox)

            # Nếu đã check-in gần đây, bỏ qua
            if label in self.checked_in:
                last_time = self.last_checked.get(label, 0)
                if current_time - last_time < self.reset_time:
                    output_results.append((label, score, 'already'))
                    continue
                else:
                    self.checked_in.remove(label)

            # Adaptive thresh với hysteresis
            prev_stat = self.prev_status[grid_key]
            thresh = self.threshold - self.hysteresis_delta if prev_stat == "pending" else self.threshold

            if score < thresh:
                output_results.append((label, score, 'unknown'))
                self.prev_status[grid_key] = 'unknown'
                continue

            # Cập nhật buffer cho key
            buffer = self.frame_buffers[grid_key]
            buffer.append(score)

            # Nếu đủ frames stable
            if len(buffer) == self.required_frames and all(s >= thresh for s in buffer):
                self.checked_in.add(label)
                self.last_checked[label] = current_time
                buffer.clear()  # Reset buffer
                status = 'new'
            else:
                status = 'pending'

            self.prev_status[grid_key] = status
            output_results.append((label, score, status))

        return output_results