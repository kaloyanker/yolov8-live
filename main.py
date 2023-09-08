import cv2
import argparse

from ultralytics import YOLO
import supervision as sv
import numpy as np

colors = sv.ColorPalette.default()

polygons = [
    np.array([[0, 0], [0.5, 0], [0.5, 1], [0, 1]]),
    np.array([[0.61, 0.12], [0.6, 0.73], [0.85, 0.73], [0.85, 0.1], [0.6, 0.12]]),
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument("--webcam-resolution", default=[1280, 720], nargs=2, type=int)
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture("http://test:test@192.168.1.7:8080/video")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("yolov8l.pt")

    zone_polygons = [
        (polygon * np.array(args.webcam_resolution)).astype(int) for polygon in polygons
    ]

    zones = [
        sv.PolygonZone(
            polygon=zone_polygon, frame_resolution_wh=tuple(args.webcam_resolution)
        )
        for zone_polygon in zone_polygons
    ]

    zone_annotators = [
        sv.PolygonZoneAnnotator(
            zone=zone,
            color=colors.by_idx(index),
            thickness=2,
            text_thickness=1,
            text_scale=2,
        )
        for index, zone in enumerate(zones)
    ]

    box_annotators = [
        sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=1)
        for index in range(len(polygons))
    ]

    while True:
        ret, frame = cap.read()

        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)
        detections = detections[detections.class_id != 0]

        for zone, zone_annotator, box_annotator in zip(
            zones, zone_annotators, box_annotators
        ):
            mask = zone.trigger(detections=detections)
            detections_filtered = detections[mask]
            labels = [
                f"{model.model.names[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, _ in detections_filtered
            ]
            frame = box_annotator.annotate(
                scene=frame, detections=detections_filtered, labels=labels
            )
            print("Zone 1:")
            print(zone_annotators[0].zone.current_count)
            print("Zone 2:")
            print(zone_annotators[1].zone.current_count)
            frame = zone_annotator.annotate(scene=frame)

        # print[zone_annotators[0].zone.current_count]
        # print[zone_annotators[1].zone.current_count]

        cv2.imshow("yolov8", frame)

        if cv2.waitKey(30) == 27:
            break


if __name__ == "__main__":
    main()
