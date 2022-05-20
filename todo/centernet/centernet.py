import cv2 as cv
import xarray as xr

##
fdir = r'C:\Users\82109\Videos\Captures/KakaoTalk_20210430_122103767.mp4'

cap = cv.VideoCapture(fdir)
fps = int(cap.get(cv.CAP_PROP_FPS))
delay = int(1000/fps)

def onChange():
    pass
cv_name = '1'
cv.namedWindow(cv_name)
cv.createTrackbar('ths',cv_name,0,255,onChange)
cv.setTrackbarPos('ths',cv_name,120)

#cap.get(cv.CAP_PROP_FPS)
if cap.isOpened():
    while True:
        ret,frame = cap.read()
        if not ret:
            break
        ths = cv.getTrackbarPos('ths', cv_name)
        frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        _,frame = cv.threshold(frame,ths,255,0)

        c,h = cv.findContours(frame,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
        for ci in c:
            cv.drawContours(frame,[ci],0,(0,0,255),1)

        cv.imshow(cv_name,frame)
        if cv.waitKey(10)==13:
            break
    cap.release()
    cv.destroyAllWindows()


##


