from __future__ import division
import utils

if __name__ == '__main__':
    # code to build the dataframe instead of loading from pickle file
    df = utils.build_data()
    df.to_pickle('dataframe_centroid_laptop.pickle')

    # rf, columns = utils.generate_classifier()
    #
    # image = utils.read_image("images/right-frontal.jpg")
    # net = utils.init_hand_pose()
    # test = utils.format_custom_data(image, columns)
    # test_pred = rf.predict(test)
    #
    # print('Actual: ', utils.label['O'])
    # print('Predicted: ', test_pred[0])
