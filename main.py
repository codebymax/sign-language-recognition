from __future__ import division

import cv2
import numpy as np
import os
import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


# Initialize the hand pose estimator model
def init():
    proto_file = "hand/pose_deploy.prototxt"
    weights_file = "hand/pose_iter_102000.caffemodel"

    net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)
    return net


# Used by run_hand_pose to open an image file and get some image data
def read_image(fname):
    frame = cv2.imread(fname)
    frame_copy = np.copy(frame)
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    aspect_ratio = frame_width / frame_height
    return frame, frame_copy, frame_width, frame_height, aspect_ratio


# This function run the hand pose estimator on an image and outputs the keypoints
def run_hand_pose(fname, net):
    frame, frame_copy, frame_width, frame_height, aspect_ratio = read_image(fname)
    n_points = 22
    t = time.time()
    # input image dimensions for the network
    in_height = 368
    in_width = int(((aspect_ratio * in_height) * 8) // 8)
    inp_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (in_width, in_height), (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inp_blob)

    output = net.forward()
    print("time taken by network : {:.3f}".format(time.time() - t))

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
    for pair in pose_pairs:
        part_a = pair[0]
        part_b = pair[1]

        if points[part_a] and points[part_b]:
            cv2.line(frame, points[part_a], points[part_b], (0, 255, 255), 2)
            cv2.circle(frame, points[part_a], 4, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, points[part_b], 4, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    # cv2.imshow('Output-Keypoints', frame_copy)
    # cv2.imshow('Output-Skeleton', frame)

    # cv2.imwrite('Output-Keypoints.jpg', frame_copy)
    # cv2.imwrite('Output-Skeleton.jpg', frame)

    print("Total time taken : {:.3f}".format(time.time() - t))

    return points


# This function outputs a single row of the dataframe
# It splits a pair of points (x, y) into two columns pointn_x and pointn_y
def create_row(letter, points):
    label = {'nothing': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9, 'K': 10,
             'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21,
             'W': 22, 'X': 23, 'Y': 24}

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

    df = pd.DataFrame(columns=columns)

    start = time.time()
    net = init()
    for letter in categories:
        path = 'images/' + letter
        for filename in os.listdir(path):
            row = create_row(letter, run_hand_pose(path + '/' + filename, net))
            df = df.append(row, ignore_index=True)
            # print(df)
    print("time elapsed : {:.3f}".format(time.time() - start))
    return df


if __name__ == '__main__':
    # code to build the dataframe instead of loading from pickle file
    # df = build_data()
    # df.to_pickle('dataframe.pickle')


    # load from pickle file and perform random forest classification
    df = pd.read_pickle('dataframe.pickle')

    labels = np.array(df['label'])

    df2 = df.drop('label', axis=1)

    columns = list(df2.columns)

    data = np.array(df2, dtype=int)

    X_train, X_test, y_train, y_test = \
        train_test_split(data, labels, test_size=0.2, random_state=42)

    y_train = y_train.astype('int')

    y_test = y_test.astype('int')

    rf = RandomForestClassifier(n_estimators=10000, random_state=42)

    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    net = init()
    test = run_hand_pose('test.jpg', net)
    test = create_row('B', test)
    df = pd.DataFrame(test[1:])
    print(df)
    #test_pred = rf.predict(test)

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
