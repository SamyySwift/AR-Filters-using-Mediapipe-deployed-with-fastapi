from fastapi import FastAPI, Request, File, UploadFile, BackgroundTasks, Response
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse
import cv2
from apply_filter import *

app = FastAPI()

# Render html files
templates = Jinja2Templates(directory="templates")


def show_frames():

    # process input from webcam or video file
    cap = cv2.VideoCapture(0)

    # Some variables
    count = 0
    isFirstFrame = True
    sigma = 50

    iter_filter_keys = iter(filters_config.keys())
    filters, multi_filter_runtime = load_filter(next(iter_filter_keys))

    # The main loop
    while True:

        ret, frame = cap.read()
        if not ret:
            break
        else:

            points2 = getLandmarks(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # if face is partially detected
            if not points2 or (len(points2) != 75):
                continue

            ################ Optical Flow and Stabilization Code #####################
            img2Gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if isFirstFrame:
                points2Prev = np.array(points2, np.float32)
                img2GrayPrev = np.copy(img2Gray)
                isFirstFrame = False

            lk_params = dict(winSize=(101, 101), maxLevel=15,
                            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.001))
            points2Next, st, err = cv2.calcOpticalFlowPyrLK(img2GrayPrev, img2Gray, points2Prev,
                                                            np.array(points2, np.float32),
                                                            **lk_params)

            # Final landmark points are a weighted average of detected landmarks and tracked landmarks

            for k in range(0, len(points2)):
                d = cv2.norm(np.array(points2[k]) - points2Next[k])
                alpha = math.exp(-d * d / sigma)
                points2[k] = (1 - alpha) * np.array(points2[k]) + alpha * points2Next[k]
                points2[k] = fbc.constrainPoint(points2[k], frame.shape[1], frame.shape[0])
                points2[k] = (int(points2[k][0]), int(points2[k][1]))

            # Update variables for next pass
            points2Prev = np.array(points2, np.float32)
            img2GrayPrev = img2Gray
            ################ End of Optical Flow and Stabilization Code ###############

            if VISUALIZE_FACE_POINTS:
                for idx, point in enumerate(points2):
                    cv2.circle(frame, point, 2, (255, 0, 0), -1)
                    cv2.putText(frame, str(idx), point, cv2.FONT_HERSHEY_SIMPLEX, .3, (255, 255, 255), 1)
                cv2.imshow("landmarks", frame)

            for idx, filter in enumerate(filters):

                filter_runtime = multi_filter_runtime[idx]
                img1 = filter_runtime['img']
                points1 = filter_runtime['points']
                img1_alpha = filter_runtime['img_a']

                if filter['morph']:

                    hullIndex = filter_runtime['hullIndex']
                    dt = filter_runtime['dt']
                    hull1 = filter_runtime['hull']

                    # create copy of frame
                    warped_img = np.copy(frame)

                    # Find convex hull
                    hull2 = []
                    for i in range(0, len(hullIndex)):
                        hull2.append(points2[hullIndex[i][0]])

                    mask1 = np.zeros((warped_img.shape[0], warped_img.shape[1]), dtype=np.float32)
                    mask1 = cv2.merge((mask1, mask1, mask1))
                    img1_alpha_mask = cv2.merge((img1_alpha, img1_alpha, img1_alpha))

                    # Warp the triangles
                    for i in range(0, len(dt)):
                        t1 = []
                        t2 = []

                        for j in range(0, 3):
                            t1.append(hull1[dt[i][j]])
                            t2.append(hull2[dt[i][j]])

                        fbc.warpTriangle(img1, warped_img, t1, t2)
                        fbc.warpTriangle(img1_alpha_mask, mask1, t1, t2)

                    # Blur the mask before blending
                    mask1 = cv2.GaussianBlur(mask1, (3, 3), 10)

                    mask2 = (255.0, 255.0, 255.0) - mask1

                    # Perform alpha blending of the two images
                    temp1 = np.multiply(warped_img, (mask1 * (1.0 / 255)))
                    temp2 = np.multiply(frame, (mask2 * (1.0 / 255)))
                    output = temp1 + temp2
                else:
                    dst_points = [points2[int(list(points1.keys())[0])], points2[int(list(points1.keys())[1])]]
                    tform = fbc.similarityTransform(list(points1.values()), dst_points)
                    # Apply similarity transform to input image
                    trans_img = cv2.warpAffine(img1, tform, (frame.shape[1], frame.shape[0]))
                    trans_alpha = cv2.warpAffine(img1_alpha, tform, (frame.shape[1], frame.shape[0]))
                    mask1 = cv2.merge((trans_alpha, trans_alpha, trans_alpha))

                    # Blur the mask before blending
                    mask1 = cv2.GaussianBlur(mask1, (3, 3), 10)

                    mask2 = (255.0, 255.0, 255.0) - mask1

                    # Perform alpha blending of the two images
                    temp1 = np.multiply(trans_img, (mask1 * (1.0 / 255)))
                    temp2 = np.multiply(frame, (mask2 * (1.0 / 255)))
                    output = temp1 + temp2

                frame = output = np.uint8(output)

            # cv2.putText(frame, "Press F to change filters", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 1)

            # cv2.imshow("Face Filter", output)

            keypressed = cv2.waitKey(1) & 0xFF
            if keypressed == 27:
                break
            # Put next filter if 'f' is pressed
            elif keypressed == ord('f'):
                try:
                    filters, multi_filter_runtime = load_filter(next(iter_filter_keys))
                except:
                    iter_filter_keys = iter(filters_config.keys())
                    filters, multi_filter_runtime = load_filter(next(iter_filter_keys))

            count += 1
    
            ret, buffer = cv2.imencode('.jpg', frame)
            
            frame = buffer.tobytes()
            yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
           


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/video")
def video():
    return StreamingResponse(show_frames(), media_type="multipart/x-mixed-replace; boundary=frame")