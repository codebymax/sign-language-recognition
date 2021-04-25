from __future__ import division
import utils
import cv2
import time


if __name__ == '__main__':
    # open webcam (change index if your webcam isnt working)
    file_path = "images/test_video.mp4"
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise IOError("Cannot access video")

    result = cv2.VideoWriter(file_path[:-4] + "_output.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 30,
                             (int(cap.get(3)), int(cap.get(4))))

    # initialize hand pose net and try for CUDA optimization
    net = utils.init_hand_pose()
    if cv2.cuda.getCudaEnabledDeviceCount():
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # generate random forest classifier
    rf, columns = utils.generate_classifier('dataframe_centroid.pickle')

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_tuple = utils.img_preprocessing(frame)
        out = utils.process_frame(frame_tuple, net, rf, columns)
        frame, sign = utils.draw_text(frame, out)
        result.write(frame)

    result.release()
    cap.release()
    cv2.destroyAllWindows()
