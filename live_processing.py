from __future__ import division
import utils
import cv2
import time


if __name__ == '__main__':
    # open webcam (change index if your webcam isnt working)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot access webcam")

    # initialize hand pose net and try for CUDA optimization
    net = utils.init_hand_pose()
    if cv2.cuda.getCudaEnabledDeviceCount():
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # generate random forest classifier
    rf, columns = utils.generate_classifier('dataframe_centroid.pickle')

    count = 0
    while True:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        frame_tuple = utils.img_preprocessing(frame)
        out = utils.process_frame(frame_tuple, net, rf, columns)
        fps = round((1/(time.time() - start)), 1)
        frame, letter = utils.draw_text(frame, out, fps)
        cv2.imshow("win", frame)
        c = cv2.waitKey(1)
        # press space bar to capture an image and save
        if c == 32:
            cv2.imwrite("images/{0}{1}.png".format(letter, count), frame)
            count += 1
        if c == 27 or not ret:
            break

    cap.release()
    cv2.destroyAllWindows()
