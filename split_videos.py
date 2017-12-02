import cv2
import csv
import argparse
import pickle

# Example usage:
#

# Functions
def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

# Create the parser
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file",
                    required=True, help=("full filepath including filename"
                    "to the .csv containing the annotated videos"))
parser.add_argument("-d", "--data_path",
                    required=True, help="path to the folder containing the annotated videos")
parser.add_argument("-w", "--width",
                    required=True, type=int,
                    help="desired width of the output video")
parser.add_argument("-y", "--height",
                    required=True, type=int,
                    help="desired height of the output video")
parser.add_argument("-l", "--height",
                    required=True, type=int,
                    help="desired length of the output video")
parser.add_argument("-o", "--output_file",
                    default='out', help=("full filepath including dictname"
                    "to the .pkl containing the produced dictionary"))
# init variables
framerate = 30
# parse the arguments
args = vars(parser.parse_args())
# create the dictionary that will have a
# key - value of: video path - label
out_dict = {}
# open the .csv file
with open(args['input_file']) as csvfile:
    filereader = csv.reader(csvfile, delimiter=',')
    row = next(filereader)
    video_count = 0
    cap = None
    for row in filereader:
        # if the row has only one value, it's a video ID
        if len(row) == 1:
            video_id = int(row[0])
            if cap is not None:
                cap.release()
            cap = cv2.VideoCapture(str(video_id)+'.mp4')
            video_count = 0
            sec_count = 0
        elif len(row) == 3:
            init_time = int(row[0])
            end_time = int(row[1])
            label = int(row[2])
            # Read the frames that correspond to one
            # seccond until getting to the init_time
            while sec_count < init_time:
                for i in range(framerate) and cap.isOpened():
                    ret, frame = cap.read()
                sec_count += 1
            # Create the new video
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_name = str(video_id)+'_'+str(video_count)+'.avi'
            out = cv2.VideoWriter(,fourcc, 20.0, (640,480))
            while sec_count < end_time:
                for vid_secs in range(args['legth']):
                    for i in range(framerate) and cap.isOpened():
                        ret, frame = cap.read()
                        out.write(frame)
                    sec_count += 1
            out.release()
            out_dict[video_name] = label
        else:
            print(".csv file follows an invalid format for this program")

# save the dictionary in a file
save_obj(out_dict,args['output_file']



        # if the row has several values, it's a time - step with a label

