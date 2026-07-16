import cv2
from ultralytics import YOLO

# load models
vehicle_model = YOLO("yolo11n.pt")
accident_model = YOLO("acc_best.pt")
severity_model = YOLO("car-damage.pt")

# traffic camera stream
CAMERA_URL = "https://video.dot.state.mn.us/public/C9181.stream/playlist.m3u8"

cap = cv2.VideoCapture(CAMERA_URL)

if not cap.isOpened():
    print("Error opening stream")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Stream ended")
        break

    frame = cv2.resize(frame,(960,540))

    # --------------------
    # VEHICLE DETECTION
    # --------------------
    vehicle_results = vehicle_model.track(
        frame,
        persist=True,
        classes=[2,3,5,7]
    )

    vehicles_detected = False

    for r in vehicle_results:
        if r.boxes is not None:

            vehicles_detected = True

            for box in r.boxes.xyxy:

                x1,y1,x2,y2 = map(int,box)

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame,"Vehicle",(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    # --------------------
    # ACCIDENT DETECTION
    # --------------------
    if vehicles_detected:

        accident_results = accident_model(frame)

        accident_found = False

        for r in accident_results:
            if r.boxes is not None:

                accident_found = True

                for box in r.boxes.xyxy:

                    x1,y1,x2,y2 = map(int,box)

                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),3)
                    cv2.putText(frame,"ACCIDENT",
                                (x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,(0,0,255),2)

        # --------------------
        # SEVERITY DETECTION
        # --------------------
        if accident_found:

            severity_results = severity_model(frame)

            for r in severity_results:

                if r.boxes is not None:

                    for box,cls,conf in zip(
                        r.boxes.xyxy,
                        r.boxes.cls,
                        r.boxes.conf):

                        if conf < 0.4:
                            continue

                        x1,y1,x2,y2 = map(int,box)

                        label = f"Severity {int(cls)}"

                        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)

                        cv2.putText(frame,label,
                                    (x1,y2+20),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,(255,0,0),2)

    cv2.imshow("Traffic Accident Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()