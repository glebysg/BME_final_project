import cv2
import csv
import argparse
import pickle
from random import shuffle

# Example usage:
# python split_videos.py -i annotated_test_data.csv -w 224 -y 224 -d data/ -l 10 -v data/test_data/

# Functions
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

# Create the parser
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file",
                    required=True, help=("full filepath including filename"
                    "to the .csv containing the annotated videos"))
parser.add_argument("-d", "--data_path",
                    required=True, help="path to the folder containing the annotated videos")
parser.add_argument("-v", "--output_vid_path",
                    required=True, help="path to the folder containing the new videos")
parser.add_argument("-w", "--width",
                    required=True, type=int,
                    help="desired width of the output video")
parser.add_argument("-y", "--height",
                    required=True, type=int,
                    help="desired height of the output video")
parser.add_argument("-l", "--length",
                    required=True, type=int,
                    help="desired length of the output video in secconds")
parser.add_argument("-o", "--output_file",
                    default='out', help=("full filepath including dictname"
                    "to the .pkl containing the produced dictionary"))
parser.add_argument("-p", "--percentage",
                    default=80, type=int,
                    help="percentage of the samples that will be used for training")
parser.add_argument("-f", "--desired_framerate",
                    default=30, type=int)
parser.add_argument("-n", "--desired_length",
                    default=30, type=int)
# init variables
framerate = 30
# parse the arguments
args = vars(parser.parse_args())
desired_framerate = args['desired_framerate']
desired_length = args['desired_length']
# create the dictionary that will have a
# key - value of: video path - label
out_dict = {}
train_dict = {}
test_dict = {}
# open the .csv file
with open(args['input_file']) as csvfile:
    filereader = csv.reader(csvfile, delimiter=',')
    video_count = 0
    sec_count = 0
    cap = None
    for row in filereader:
        # if the row has only one value, it's a video ID
        if len(row) == 1:
            video_id = int(row[0])
            if cap is not None:
                cap.release()
            input_video_name = args['data_path']+str(video_id)+'.mp4'
            cap = cv2.VideoCapture(input_video_name)
            print("Is the cap opened?", cap.isOpened(), "for: ", input_video_name)
            video_count = 0
            sec_count = 0
        elif len(row) == 3 and (cap is not None):
            init_time = int(row[0])
            end_time = int(row[1])
            label = int(row[2])
            # initialize the label key with an empty list
            # if we haven't added that list yet.
            try:
                out_dict[label]
            except KeyError:
                out_dict[label] = []
            # Read the frames that correspond to one
            # seccond until getting to the init_time
            print("sec_count: ", sec_count, " init_time: ", init_time)
            while sec_count < init_time and cap.isOpened():
                for i in range(framerate):
                    ret, frame = cap.read()
                sec_count += 1
                # print("Advancing",sec_count, "in: ", video_id)
            print("sec_count: ", sec_count, " end_time: ", end_time)
            while sec_count < end_time and cap.isOpened():
                # Create the new video
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out_video_name = args['output_vid_path'] + str(video_id)+'_'+str(video_count)+'.avi'
                out = cv2.VideoWriter(out_video_name,fourcc, desired_framerate, (args['width'],args['height']))
                individual_duration = 0
                img_count = 0
                for vid_secs in range(args['length']):
                    for i in range(framerate):
                        ret, frame = cap.read()
                        # resize the image to the required size by the videowriter
                        if (img_count%int(framerate*args['length']/desired_length)) == 0:
                            # print("added img", out_video_name)
                            res = cv2.resize(frame,(args['width'], args['height']))
                            out.write(res)
                        img_count += 1
                    sec_count += 1
                    individual_duration += 1
                out.release()
                video_count+= 1
                if individual_duration == args['length']:
                    out_dict[label].append(out_video_name)
                else:
                    print("video: ", out_video_name, "Was too short to be included")
        else:
            print(".csv file follows an invalid format for this program")
# save the dictionary in a file
# split the data into train and test
print(out_dict)
for key, value in out_dict.items():
    shuffle(value)
    percetage_len = int((args['percentage']/100.0)*len(value))
    train_dict[key] = value[:percetage_len]
    test_dict[key] = value[percetage_len:]
save_obj(out_dict,args['output_file'])
save_obj(train_dict,'train'+args['output_file'])
save_obj(test_dict,'test'+args['output_file'])
