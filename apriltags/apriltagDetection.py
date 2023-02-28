import cv2
import plotly.express as px
from pupil_apriltags import Detector
import pandas as pd
from io import BytesIO
import numpy as np
from PIL import Image
from numpy import asarray
import warnings

warnings.filterwarnings("ignore")

def radarChartImage(data, size=700):
    df = pd.DataFrame(dict(
    r=data[0],
    theta=data[1]))
    fig = px.line_polar(df, r='r', theta='theta', line_close=True, range_r=(0,10))
    image_bytes = fig.to_image(width=size, height=size)
    image = np.array(Image.open(BytesIO(image_bytes)))
    data = asarray(image)
    return data

def findFront(tag):
    p1 = tag.corners[0]
    p2 = tag.corners[1]
    return (int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2))

def detectApriltags(frame):
    #  initializing detector
    at_detector = Detector(
        # change this for more tags
        families="tag36h11",
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )  
    
    grey_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tags = at_detector.detect(
            grey_image,
            estimate_tag_pose=False,
            camera_params=None,
            tag_size=None,
        )

    # drawing on the markers
    for tag in tags:
        for idx in range(len(tag.corners)):
            cv2.line(
                frame,
                tuple(tag.corners[idx - 1, :].astype(int)),
                tuple(tag.corners[idx, :].astype(int)),
                (0, 255, 0),
                3
            )
            
        cv2.putText(
            frame,
            str(tag.tag_id),
            org=(
                tag.corners[0, 0].astype(int) + 10,
                tag.corners[0, 1].astype(int) + 10,
            ),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 255),
            thickness=2
        )
        
        cv2.arrowedLine(
            frame,
            (int(tag.center[0]), int(tag.center[1])),
            findFront(tag),
            (0,0,255),
            2
        )
    return tags

def getStats(tags):
    return ([5,5,5,5,5],['Wood', 'Earth', 'Metal', 'Fire', 'Water']) 

def combineFrames(videoframe, chartFrame):
    height1, width1 = chartFrame.shape[0],chartFrame.shape[1]
    height2, width2 = videoframe.shape[0],videoframe.shape[1]
    min_height = min(height1, height2)
    min_width = min(width1, width2)
 
    img1 = chartFrame[0:min_height, 0:min_width]
    img2 = videoframe[0:min_height, 0:min_width]
    img_add = cv2.add(img1, img2) 
    chartFrame[0:min_height,0:min_width] = img_add
    return chartFrame

def main():
    # define a video capture object
    vid = cv2.VideoCapture(0)
    
    while(True):
        # Capture the video frame by frame
        ret, videoframe = vid.read()
        
        # get tag data
        tags = detectApriltags(videoframe)
        
        # make radar chart
        fengShuiData = getStats(tags)
        chartFrame = radarChartImage(fengShuiData, 400)
        
        # shows the aruco detection
        cv2.imshow('Apriltag Detection', videoframe)
        # shows the chart. uncomment to see chart
        # cv2.imshow('radar chart / start chart / spider graph', chartFrame)
        
        # =================================================================================================
        # the stuff below was me trying to join the graph and the video together. doesn't work
        # combine the chart with the video
        # result_frame = combineFrames(videoframe, chartFrame)
        
        # Display the resulting frame
        # iresult_framemg = cv2.addWeighted(chartFrame, 0.3, videoframe, 0.7, 0)
        # cv2.imshow('frame', chartFrame)
        # cv2.imshow('frame', result_frame)

        # the 'q' button is set as the quitting button you may use any desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    vid.release()
    cv2.destroyAllWindows()   
    
main()
