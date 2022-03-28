# A simple and lightweight package for state of the art face detection with GPU support
# pip install face-detection

################################### Social Distancing Analyzer and Mask Detector Model ###################################


# Import the necessary libraries
import cv2
import face_detection
import numpy as np
from sklearn.cluster import DBSCAN
from tensorflow.keras.models import load_model  , Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 , preprocess_input


# Get COCO class names
classes = []
with open("coco.names") as file:
    classes = [line.strip() for line in file.readlines()]


# Loading YoloV3 weights and configuration file using dnn module
modelWeights = 'yolov3.weights'
modelConfig = 'yolov3.cfg'
network = cv2.dnn.readNet(modelWeights , modelConfig)

# Get the name of all layers of the network
layer_names = network.getLayerNames()
# To run a forward pass using the cv2.dnn module, we need to pass in the names of layers for which the output is to be computed
output_layers = [layer_names[i[0] - 1] for i in network.getUnconnectedOutLayers()]


# State of the Art Face Detection in Pytorch with DSFD and RetinaFace
# Greater values of confidence threshold would detect only clear faces
detector = face_detection.build_detector("RetinaNetMobileNetV1", confidence_threshold = 0.5, nms_iou_threshold = 0.3)


# Load trained Face Mask Classfier (MobileNetV2 Model)
mask_classifier = load_model("#model.h5")
label2class = {0: 'with_mask', 1: 'without_mask'}


# path to input video file
path = './demo.mp4'
path1 = './Rohit Sharma With Full Secruty At Airport..mp4'
path2 = './Australian cricket team arrives at Johannesburg airport.mp4'


# Open video file or a capturing device or a IP video stream and fetch video properties
cap = cv2.VideoCapture(path)
if not cap.isOpened():
    print("Error opening video")

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)


# Initialize output video stream writer
output_stream = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'MP4V'), fps, (int(width),int(height)))


print("Starting Video Stream..")

while (cap.isOpened()):

    res, frame = cap.read()
    if res == False:
        break

    # Frame Dimensions
    height, width, ch = frame.shape

    # Detecting objects in the frame with YOLOv3
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0/255, size=(416,416), mean=(0,0,0), swapRB=True, crop=False)
    network.setInput(blob)
    outputs = network.forward(output_layers)


    class_ids = []
    bboxes = []
    confidences = []
    
    # Store label, bounding box and confidence score of different objects detected
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:               
                # center coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                # height and width of the box
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Compute top left co-ordinates of bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                bboxes.append([x, y, w, h])
                confidences.append(float(confidence))

    # Performs non maximum suppression given boxes and corresponding scores to eliminate overlapping boxes
    indexes = cv2.dnn.NMSBoxes(bboxes , confidences , 0.5 , 0.4)


    # Initialize empty lists for storing Bounding Boxes of People and their Faces
    persons = []
    masked_faces = []
    unmasked_faces = []

    for i in range(len(bboxes)):

        if i in indexes:

            box = np.array(bboxes[i])
            box = np.where(box<0,0,box)
            (x, y, w, h) = box

            label = str(classes[class_ids[i]])


            if label=='person':
                persons.append([x,y,w,h])              

                # Convert the image frame from BGR color (which OpenCV uses) to RGB color (which face_detection uses)
                person_rgb = frame[y:y+h,x:x+w,::-1]   
                # Find all the faces (in detected persons) in the current frame of video
                detections = detector.detect(person_rgb)

                # If a Face is Detected
                if detections.shape[0] > 0:

                  detection = np.array(detections[0])
                  detection = np.where(detection<0,0,detection)

                  # Calculating Co-ordinates of the Detected Face
                  x1 = x + int(detection[0])
                  x2 = x + int(detection[2])
                  y1 = y + int(detection[1])
                  y2 = y + int(detection[3])


                  try :
                    # Crop & BGR to RGB
                    face_rgb = frame[y1:y2,x1:x2,::-1]   

                    # Preprocess the Image
                    face_arr = cv2.resize(face_rgb, (224, 224), interpolation=cv2.INTER_NEAREST)
                    face_arr = np.expand_dims(face_arr, axis=0)
                    face_arr = preprocess_input(face_arr)

                    # Predict if the Face is Masked or Not and store the results
                    score = mask_classifier.predict(face_arr)

                    if score[0].argmax() == 1:
                      unmasked_faces.append([x1,y1,x2,y2])
                    else:
                      masked_faces.append([x1,y1,x2,y2])

                  except:
                    continue


    
    # Calculate Coordinates of People Detected and find Clusters using DBSCAN
    person_coordinates = []

    for p in range(len(persons)):
      person_coordinates.append((persons[p][0]+int(persons[p][2]/2),persons[p][1]+int(persons[p][3]/2)))

    if len(persons)>0:
      clustering = DBSCAN(eps=50 , min_samples=2).fit(person_coordinates)
      isSafe = clustering.labels_
    else:
      isSafe = []



    # Count no of persons detected with and without mask
    person_count = len(persons)
    masked_face_count = len(masked_faces)
    unmasked_face_count = len(unmasked_faces)
    safe_count = np.sum((isSafe==-1)*1)
    unsafe_count = person_count - safe_count



    # Show Clusters using Red Lines
    arg_sorted = np.argsort(isSafe)

    for i in range(1,person_count):

      if isSafe[arg_sorted[i]]!=-1 and isSafe[arg_sorted[i]]==isSafe[arg_sorted[i-1]]:
        cv2.line(frame , person_coordinates[arg_sorted[i]] , person_coordinates[arg_sorted[i-1]] , (0,0,255) , 2)



    # Put Bounding Boxes on People in the Frame
    for p in range(person_count):

      a,b,c,d = persons[p]

      # Green if Safe, Red if UnSafe
      if isSafe[p]==-1:
        cv2.rectangle(frame, (a, b), (a + c, b + d), (0,255,0), 2)
      else:
        cv2.rectangle(frame, (a, b), (a + c, b + d), (0,0,255), 2)



    # Put Bounding Boxes on Faces in the Frame  ;  Green if Safe, Red if UnSafe
    for f in range(masked_face_count):

      a,b,c,d = masked_faces[f]
      cv2.rectangle(frame, (a, b), (c,d), (0,255,0), 2)

    for f in range(unmasked_face_count):

      a,b,c,d = unmasked_faces[f]
      cv2.rectangle(frame, (a, b), (c,d), (0,0,255), 2)



    # Show Monitoring Status in a Black Box at the Top
    cv2.rectangle(frame , (0,0) , (width,50) , (0,0,0) , -1)
    cv2.rectangle(frame , (1,1) , (width-1,50) , (255,255,255) , 2)

    xpos = 15

    string = "Total People = " + str(person_count)
    cv2.putText(frame , string , (xpos,35) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255,255,255) , 2)
    xpos += cv2.getTextSize(string , cv2.FONT_HERSHEY_SIMPLEX , 1 , 2)[0][0]


    string = " ( "+str(safe_count) + " Safe "
    cv2.putText(frame , string , (xpos,35) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0,255,0) , 2)
    xpos += cv2.getTextSize(string , cv2.FONT_HERSHEY_SIMPLEX , 1 , 2)[0][0]

    string = str(unsafe_count)+ " Unsafe ) "
    cv2.putText(frame , string , (xpos,35) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0,0,255) , 2)
    xpos += cv2.getTextSize(string,cv2.FONT_HERSHEY_SIMPLEX,1,2)[0][0]

    
    string = "( " +str(masked_face_count)+" Masked "+str(unmasked_face_count)+" Unmasked "+\
             str(person_count-masked_face_count-unmasked_face_count)+" Unknown )"
    cv2.putText(frame , string , (xpos,35) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255,0,0) , 2)


    # Write Frame to the Output File
    output_stream.write(frame)

    cv2.imshow('OutPut_Frame', frame)


    # Exit stream when 'esc' key is pressed 
    if cv2.waitKey(1) == 27:
        print('Video Stream Ended')
        break



output_stream.release()
cap.release()
cv2.destroyAllWindows()
print('Done..!!')