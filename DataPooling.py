import torch
import cv2
import pickle
import random
import numpy as np


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

class DataPool:

    def __init__(self, name, stream_size):
        self.dict_obj = load_obj(name)
        self.length = self.get_dict_len()
        self.streams = []
        self.stream_size = stream_size
        self.round_robin_index = 0
        self.inilizalized = False

    def get_dict_len(self):
        length_count = 0
        for key, value in self.dict_obj.items():
            length_count += len(value)
        return length_count

    def get_one_video(self):
        current_len = self.get_dict_len()
        if not current_len:
            return None, None
        selected_index = random.randint(0,current_len-1)
        selected_video = None
        selected_video_label = None
        accum_len = 0
        for key, value in self.dict_obj.items():
            class_len = len(value)
            if (selected_index - accum_len) < class_len:
                selected_video = value[selected_index - accum_len]
                selected_video_label = key
                del value[selected_index - accum_len]
                break
            accum_len += class_len
        cap = cv2.VideoCapture(selected_video)
        if not selected_video_label or not cap.isOpened():
            print("Could not select a video from the index or open it")
            exit(0)
        return cap, selected_video_label;

    def initialize_stream(self):
        for i in range(self.stream_size):
            cap, label = self.get_one_video()
            if not cap:
                print("ERROR: Not enough videos to buid a stream")
                exit(0)
            self.streams.append((cap,label))
        self.inilizalized = True

    def total_videos(self):
        return self.length

    def nextImage(self):
        if not self.inilizalized:
            self.initialize_stream()
            # print(self.streams)
        ret = False
        while (not ret):
            if self.stream_size == 0:
                print("End of the stream")
                return None, None
            cap, label = self.streams[self.round_robin_index]
            ret, frame = cap.read()
            # if we reached the end of the video
            if not ret:
                cap.release()
                del self.streams[self.round_robin_index]
                new_cap, new_label = self.get_one_video()
                if new_cap is None:
                    self.stream_size -= 1
                else:
                    self.streams.append((new_cap,new_label))
            if self.stream_size == 0:
                return None, None
            self.round_robin_index += 1
            self.round_robin_index %= self.stream_size
        # Manual Resize
        tensor_frame = []
        tensor_frame.append(frame[:,:,0])
        tensor_frame.append(frame[:,:,1])
        tensor_frame.append(frame[:,:,2])
        tensor_frame = torch.from_numpy(np.array([tensor_frame]))
        tensor_label = torch.LongTensor(1)
        tensor_label[0] = label-2
        return tensor_frame.float(), tensor_label
