import cv2
import time
import json
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import paho.mqtt.client as mqtt
from vidgear.gears import WriteGear


def init_streamer(args):
    """
    
    """
    
    output_params = {
        "-f": "rtsp", 
        "-input_framerate": 15,
        "-preset": "ultrafast",
        "-vcodec": "libx264",
        "-tune": "fastdecode",
        "-crf": 18
    }

    streamer = WriteGear(output_filename="rtsp://{}:8554/{}".format(args.ip, args.service_uri), compression_mode=True, 
                         logging=False, **output_params)

    return streamer

def init_video_capture(args):
    """
    
    """

    cap = cv2.VideoCapture(args.video_path)

    return cap

def init_mqtt_client(client_name: str, mqtt_broker: str) -> mqtt.Client:
    """
        Initializes an MQTT client and connects it to the desired MQTT borker
    """

    client = mqtt.Client(client_name)
    client.connect(mqtt_broker)

    return client


def publish_frame_detections(client: mqtt.Client, topic: str, pred_info: np.array):
    """
        This function transforms the prediction info to a byte array and sends it to the desired MQTT 
        topic

        :param client: mqtt client used to broadcast the predictions info. The client is already connected
                       to the MQTT broker
        :param topic: topic where to broadcast the information
        :param pred_info: numpy array with the prediction info. Format:
                [0: ts x1 ][1-4: dect_xyxy x4 ][5: score x1 ][6: type_obj x1 ][7: id_obj x1 ][8-9: dect_middle x2 ][10: vel x1 ]
                [11-12: dect_middle_lonlat x2 ][13: frame_num x1 ][14: cam_id x1 ]
    """

    client.publish(topic, json.dumps(pred_info.tolist()))


def publish_alarm(client: mqtt.Client, topic: str, alarm_info: np.array):
    """
        This function transforms the prediction info to a byte array and sends it to the desired MQTT 
        topic

        :param client: mqtt client used to broadcast the predictions info. The client is already connected
                       to the MQTT broker
        :param topic: topic where to broadcast the information
        :param alarm_info: numpy array with the alarm info. Format:
                [0: ts x1 ][1: alarm  x1][2: distance x1]
    """

    client.publish(topic, json.dumps(alarm_info.tolist()))


def publish(client, tracking_data, alarm_data, video_capture, streamer):
    """
        Publishes the data from the pickle file at a rate of 15 FPS

        :param client: mqtt client already connected to a MQTT broker
        :param tracking_data: tracking data to publish to the TRACKING topic
        :param alarm_data: alarm data to publish to the ALARM topic
    """

    print("Starting to publish data")

    last_time = time.time_ns()/1E9
    time_between_publish = 1/15.0

    for i in tqdm(range(len(tracking_data))):
        while (time.time_ns()/1E9 - last_time) < time_between_publish:
            pass

        last_time = time.time_ns()/1E9
        if video_capture is not None:
            read_and_stream(video_capture, streamer, (args.width, args.height))
        publish_frame_detections(client, "TRACKING", tracking_data[i])
        publish_alarm(client, "ALARM", alarm_data[i])


def read_and_stream(video_capture, streamer, img_size):
    """
    
    """

    ret, frame = video_capture.read()

    if ret:
        frame = cv2.resize(frame, img_size)
        streamer.write(frame)


def main(args):
    # Load data from pickle file
    with open("tracking_data.pkl", "rb") as f:
        tracking_data = pickle.load(f)
    with open("alarm_data.pkl", "rb") as f:
        alarm_data = pickle.load(f)

    client = init_mqtt_client(args.client_name, args.broker)

    if args.video_path is not None:
        streamer = init_streamer(args)
        video_capture = init_video_capture(args)
    else:
        streamer = None
        video_capture = None

    if args.infinite_publication:
        while True:
            publish(client, tracking_data, alarm_data, video_capture, streamer)
            if video_capture is not None:
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    else:
        publish(client, tracking_data, alarm_data, video_capture, streamer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Video and stream parameters
    parser.add_argument("--video-path", type=str, default=None)
    parser.add_argument("--fps", default=20, help="If -1, use fps detected by opencv")  # On some videos with bad metadata Opencv fails to detect FPS
    parser.add_argument("--bitrate", type=int, default=600)
    parser.add_argument("--ip", type=str, default="localhost")
    parser.add_argument("--service-uri", type=str, default="dai_output")
    parser.add_argument("--height", type=int, default=450)
    parser.add_argument("--width", type=int, default=800)

    # MQTT parameters
    parser.add_argument("--broker", type=str, default="localhost")
    parser.add_argument("--client-name", type=str, default="peddect_client")
    parser.add_argument("--no-infinite-publication", dest="infinite_publication", action="store_false")
    parser.set_defaults(inifite_publication=True)

    args = parser.parse_args()
    main(args)
