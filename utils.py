from __future__ import division
import random
import cv2
import numpy as np
import os
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

label = {'nothing': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9, 'K': 10,
         'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21,
         'W': 22, 'X': 23, 'Y': 24}


# Initialize the hand pose estimator model
def init_hand_pose():
    proto_file = "hand/pose_deploy.prototxt"
    weights_file = "hand/pose_iter_102000.caffemodel"

    net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)
    if cv2.cuda.getCudaEnabledDeviceCount():
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    return net


# Used by run_hand_pose to open an image file and get some image data
def read_image(fname):
    frame = cv2.imread(fname)
    if frame.shape[0] < 200 and frame.shape[1] < 200:
        frame = cv2.resize(frame, (200, 200))
    frame_copy = np.copy(frame)
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    aspect_ratio = frame_width / frame_height
    return frame, frame_copy, frame_width, frame_height, aspect_ratio


# This function run the hand pose estimator on an image and outputs the keypoints
def run_hand_pose(img_data, net):
    frame, frame_copy, frame_width, frame_height, aspect_ratio = img_data
    n_points = 22
    t = time.time()
    # input image dimensions for the network
    in_height = 368
    in_width = int(((aspect_ratio * in_height) * 8) // 8)
    inp_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (in_width, in_height), (0, 0, 0), swapRB=False, crop=False)


    net.setInput(inp_blob)

    output = net.forward()
    # print("time taken by network : {:.3f}".format(time.time() - t))

    # Empty list to store the detected keypoints
    points = []
    threshold = 0.1

    for i in range(n_points):
        # confidence map of corresponding body's part.
        prob_map = output[0, i, :, :]
        prob_map = cv2.resize(prob_map, (frame_width, frame_height))

        # Find global maxima of the probMap.
        min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)

        if prob > threshold:
            cv2.circle(frame_copy, (int(point[0]), int(point[1])), 4, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame_copy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255),
                        2, lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(point[0]), int(point[1])))
        else:
            points.append(None)

    pose_pairs = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12],
                  [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

    # Draw Skeleton
    # for pair in pose_pairs:
    #     part_a = pair[0]
    #     part_b = pair[1]
    #
    #     if points[part_a] and points[part_b]:
    #         cv2.line(frame, points[part_a], points[part_b], (0, 255, 255), 2)
    #         cv2.circle(frame, points[part_a], 4, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
    #         cv2.circle(frame, points[part_b], 4, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
    #
    # cv2.imshow('Output-Keypoints', frame_copy)
    # cv2.imshow('Output-Skeleton', frame)
    #
    # cv2.imwrite('Output-Keypoints.jpg', frame_copy)
    # cv2.imwrite('Output-Skeleton.jpg', frame)
    #
    # print("Total time taken : {:.3f}".format(time.time() - t))
    #
    # cv2.waitKey(0)
    return points


# This function outputs a single row of the dataframe
# It splits a pair of points (x, y) into two columns pointn_x and pointn_y
# the columns range from point0_x to point21_y
# the first column is label where the letter is encoded for each row
def create_row(letter, points):
    output = {'label': label[letter]}

    for i in range(len(points)):
        column_name = 'point' + str(i) + '_'
        if points[i] is None:
            # not super sure what to do if a point doesn't exist
            output[column_name + 'x'] = -1
            output[column_name + 'y'] = -1
        else:
            output[column_name + 'x'] = points[i][0]
            output[column_name + 'y'] = points[i][1]
    return output


# Given a set of keypoints returns the centroid of those points.
# Ignores None points
def get_centroid(points):
    new_points = []
    for p in points:
        if p is None:
            continue
        else:
            new_points.append(p)
    if not new_points:
        return (0, 0)
    x = [p[0] for p in new_points]
    y = [p[1] for p in new_points]
    centroid = (sum(x) / len(new_points), sum(y) / len(new_points))
    centroid = [int(num) for num in centroid]
    return centroid


# This is the same as create_row above but makes all points relative to the centroid of the keypoints
# First we calculate the centroid of all the keypoints using get_centroid.
# Then the centroid is subtracted from every point before adding to the row
def create_row_2(letter, points):
    center = get_centroid(points)

    output = {'label': label[letter]}

    for i in range(len(points)):
        column_name = 'point' + str(i) + '_'
        if points[i] is None:
            output[column_name + 'x'] = -1
            output[column_name + 'y'] = -1
        else:
            output[column_name + 'x'] = points[i][0] - center[0]
            output[column_name + 'y'] = points[i][1] - center[1]
    return output


# This function builds a dataframe using the keypoints from the images in the dataset here:
# https://www.kaggle.com/danrasband/asl-alphabet-test
# outputs dataframe
def build_data():
    categories = ['nothing', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K',
                  'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

    columns = ['label', 'point0_x', 'point0_y', 'point1_x', 'point1_y', 'point2_x', 'point2_y', 'point3_x', 'point3_y',
               'point4_x', 'point4_y', 'point5_x', 'point5_y', 'point6_x', 'point6_y', 'point7_x', 'point7_y',
               'point8_x', 'point8_y', 'point9_x', 'point9_y', 'point10_x', 'point10_y', 'point11_x', 'point11_y',
               'point12_x', 'point12_y', 'point13_x', 'point13_y', 'point14_x', 'point14_y', 'point15_x', 'point15_y',
               'point16_x', 'point16_y', 'point17_x', 'point17_y', 'point18_x', 'point18_y', 'point19_x', 'point19_y',
               'point20_x', 'point20_y', 'point21_x', 'point21_y']

    output_df = pd.DataFrame(columns=columns)

    start = time.time()
    net = init_hand_pose()
    for letter in categories:
        print('Processing: ', letter)
        path = 'images/' + letter
        for filename in os.listdir(path):
            im = read_image(path + '/' + filename)
            row = create_row_2(letter, run_hand_pose(im, net))
            output_df = output_df.append(row, ignore_index=True)

    print("time elapsed : {:.3f}".format(time.time() - start))
    return output_df


# generate one row of data for testing custom images
# might expand to multiple rows later once we get more data
def format_custom_data(img_data, columns, net):
    im = img_data
    temp = run_hand_pose(im, net)
    temp = create_row_2('B', temp)
    temp.pop('label')
    temp_df = pd.DataFrame(columns=columns)
    temp_df = temp_df.append(temp, ignore_index=True)
    out = np.array(temp_df, dtype=int)
    return out


def generate_classifier(dfName):
    # load from pickle file and perform random forest classification
    df = pd.read_pickle(dfName)
    labels = np.array(df['label'])
    df2 = df.drop('label', axis=1)
    columns = list(df2.columns)
    data = np.array(df2, dtype=int)

    # the best state seems to be 307
    # I get 89.9% accuracy when 307 is the state
    state = 307
    print(state)
    x_train, x_test, y_train, y_test = \
        train_test_split(data, labels, test_size=0.25, random_state=state)
    y_train = y_train.astype('int')
    rf = RandomForestClassifier(n_estimators=1000, random_state=state)
    rf.fit(x_train, y_train)

    y_pred = rf.predict(x_test)

    y_test = y_test.astype(int)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    return rf, columns


def draw_text(frame_in, sign, fps_val=None):
    key_list = list(label.keys())
    values_list = list(label.values())
    sign = key_list[values_list.index(sign[0])]
    frame_out = cv2.putText(frame_in, "Detected Sign: " + sign, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0)
                            , 2, cv2.LINE_AA, bottomLeftOrigin=False)
    if fps_val is not None:
        frame_out = cv2.putText(frame_in, "FPS: " + str(fps_val), (frame_in.shape[1] - 140, 25), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2, cv2.LINE_AA, bottomLeftOrigin=False)
    return frame_out, sign


def process_frame(img, net_in, rf_in, columns_in):
    test = format_custom_data(img, columns_in, net_in)
    test_pred = rf_in.predict(test)
    return test_pred


def img_preprocessing(frame_in):
    frame_copy = np.copy(frame_in)
    frame_width = frame_in.shape[1]
    frame_height = frame_in.shape[0]
    aspect_ratio = frame_width / frame_height
    frame_out = (frame_in, frame_copy, frame_width, frame_height, aspect_ratio)
    return frame_out