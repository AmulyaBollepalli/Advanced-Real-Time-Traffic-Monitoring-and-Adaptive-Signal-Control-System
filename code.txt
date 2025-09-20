import cv2
from ultralytics import YOLO
from collections import defaultdict

# Loading YOLOv8 model
model = YOLO("yolov8n.pt")  # You can use yolov8m.pt or yolov8s.pt if desired

# Defining video source
cap = cv2.VideoCapture("traffic.mp4")

# Defining the ROI for each lane manually (x1, y1, x2, y2)
# Adjusting these based on your video frame
lane_rois = {
    1: ((0, 200), (160, 480)),
    2: ((160, 200), (320, 480)),
    3: ((320, 200), (480, 480)),
    4: ((480, 200), (640, 480))
}

# Vehicle classes to detect
vehicle_classes = ['car', 'motorcycle', 'bus', 'truck', 'bicycle']

while True:
    ret, frame = cap.read()
    if not ret:
        break

    vehicle_counts = defaultdict(int)

    # Running YOLOv8 detection
    results = model(frame)[0]

    # Drawing lane ROIs
    for lane_id, ((x1, y1), (x2, y2)) in lane_rois.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(frame, f"Lane {lane_id}", (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        if class_name in vehicle_classes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Determining which lane this vehicle is in
            for lane_id, ((lx1, ly1), (lx2, ly2)) in lane_rois.items():
                if lx1 <= cx <= lx2 and ly1 <= cy <= ly2:
                    vehicle_counts[lane_id] += 1
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    break

    # Displaying counts on frame
    y_offset = 30
    for lane_id in sorted(lane_rois.keys()):
        count = vehicle_counts[lane_id]
        cv2.putText(frame, f"Lane {lane_id} Vehicles: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 25

    # Deciding which lane gets green signal
    if vehicle_counts:
        max_lane = max(vehicle_counts, key=vehicle_counts.get)
        cv2.putText(frame, f"Green Signal: Lane {max_lane}", (10, y_offset + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Green Signal: None", (10, y_offset + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Showing real-time frame
    cv2.imshow("Traffic Monitor", frame)

    # Exiting on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
