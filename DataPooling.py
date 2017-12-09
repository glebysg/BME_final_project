import torch
import cv2
import pickle
import random
import numpy as np


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

class DataPool:

    def __init__(self, name, stream_size, technique='CNN'):
        self.obj_name = name
        self.original_stream_size = stream_size
        self.dict_obj = load_obj(name)
        self.length = self.get_dict_len()
        self.streams = []
        self.stream_size = stream_size
        self.round_robin_index = 0
        self.initialized = False
        self.images_in_video = 20
        self.technique = technique

    def restart(self):
        self.dict_obj = load_obj(self.obj_name)
        self.length = self.get_dict_len()
        self.streams = []
        self.stream_size = self.original_stream_size
        self.round_robin_index = 0
        self.initialized = False

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
        if self.technique == 'CNN':
            for i in range(self.stream_size):
                cap, label = self.get_one_video()
                if not cap:
                    print("ERROR: Not enough videos to buid a stream")
                    exit(0)
                self.streams.append((cap,label))
            self.initialized = True
        elif self.technique == 'LSTM':
            cap, label = self.get_one_video()
            if not cap:
                print("ERROR: Not enough videos to buid a stream")
                exit(0)
            self.streams.append((cap,label))
            self.initialized = True


    def total_videos(self):
        return self.length

    def total_images(self):
        return self.length*self.images_in_video

    def nextImage(self):
        if self.technique == 'CNN':
            if not self.initialized:
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
            tensor_frame.append((frame[:,:,0] - np.tile(np.mean(frame[:,:,0]), (224,224)))/255.0)
            tensor_frame.append((frame[:,:,1] - np.tile(np.mean(frame[:,:,1]), (224,224)))/255.0)
            tensor_frame.append((frame[:,:,2] - np.tile(np.mean(frame[:,:,2]), (224,224)))/255.0)
            tensor_frame = torch.from_numpy(np.array([tensor_frame]))
            tensor_label = torch.LongTensor(1)
            tensor_label[0] = label-2
            return tensor_frame.float(), tensor_label
        elif self.technique == 'CNN+LSTM':
            if not self.initialized:
                self.initialize_stream()
                # print(self.streams)
            while (True):
                if len(self.streams) == 0:
                    new_cap, new_label = self.get_one_video()
                    if new_cap is None:
                        return None, None
                    else:
                        self.streams.append((new_cap,new_label))
                else:
                    cap, label = self.streams[0]
                    ret, frame = cap.read()
                    if not ret:
                        cap.release()
                        self.streams = []
                    else:
                        # Manual Resize
                        tensor_frame = []
                        tensor_frame.append((frame[:,:,0] - np.tile(np.mean(frame[:,:,0]), (224,224)))/255.0)
                        tensor_frame.append((frame[:,:,1] - np.tile(np.mean(frame[:,:,1]), (224,224)))/255.0)
                        tensor_frame.append((frame[:,:,2] - np.tile(np.mean(frame[:,:,2]), (224,224)))/255.0)
                        tensor_frame = torch.from_numpy(np.array([tensor_frame]))
                        tensor_label = torch.LongTensor(1)
                        tensor_label[0] = label-2
                        return tensor_frame.float(), tensor_label
        elif self.technique == 'LSTM':
            cap, label = self.get_one_video()
            if cap is None:
                return None, None
            tensor_frame = []
            tensor_label = torch.LongTensor(1)
            tensor_label[0] = label-2
            while (true):
                ret, frame = cap.read()
                if not ret:
                    break
                else:
                    # Manual Resize and append to the tensor ret
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    flat_gray = np.resize(gray,[1,-1])
                    tensor_frame.append((flat_gray - np.tile(np.mean(flat_gray), (224,224)))/255.0)
                    tensor_frame = torch.from_numpy(np.array([tensor_frame]))
            return tensor_frame.float(), tensor_label
