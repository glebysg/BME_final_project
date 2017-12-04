from DataPooling import DataPool

data = DataPool('trainout',20)
dict_obj = {2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
while(True):
    frame, label = data.nextImage()
    if frame is  None:
        print("GOT A NONE FRAME")
        break
    else:
        dict_obj[label] += 1
print(dict_obj)
