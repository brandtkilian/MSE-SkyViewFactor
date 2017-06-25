from cnn.cnn_main import classify_images
import argparse
import os
import time

default_weights = "cnn/weights/svf_2017-05-28_14:26:01.hdf5"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify pixels of every images in the given folder '
                                                 '(take care to rectify the optic first) and compute the views factors.'
                                                 'Outputs values in a csv files.')

    parser.add_argument('-i', '--input-path', type=str, required=True,
                        help='The path to the images folder')

    parser.add_argument('--csv-file', type=str, required="True",
                        help='The path to the csv results file. If the file exsits data are appended to the existing data in a new file name datasvf.csv')

    parser.add_argument('-o', '--output-path', type=str, nargs='?', default="outputs/predictions",
                        help='The path where to the output folder to store the classification results images')

    parser.add_argument('-w', '--weights-path', type=str, nargs='?', default=default_weights,
                        help='The path to the file that contains the weights of the classifier nodes')

    parser.add_argument('-v', '--visualize', action='store_true',
                        help="Stores the classification results in the specified output path")

    parser.add_argument('-l', '--overlay', action='store_true',
                        help="Stores the sources images with predictions colored overlays in the specified output path given with -o option")

    parser.add_argument('-m', '--magentize', action='store_true',
                        help="Use magenta color to fill the void class, if not magentize, then torify is applied")

    parser.add_argument('-t', '--torify', action='store_true',
                        help="Transform the hemispherics images into rectangles images by sampling radius, "
                             "samplig deltas depends on the width and height")

    parser.add_argument('-g', '--grav-angle', action='store_true',
                        help="Compute the angle from vertical where 50%% of the SVF is found")

    parser.add_argument('--width', type=int, nargs='?', default=480,
                       help='The width the images will be resized before going through'
                            ' the classifier (depends on which weights are loaded)')

    parser.add_argument('--height', type=int, nargs='?', default=480,
                        help='The height the images will be resized before going through'
                             ' the classifier (depends on which weights are loaded)')

    parser.add_argument('-n', '--normtype', type=int, nargs='?', default=2,
                        help='The normalization type Equalize = 1, EqualizeClahe : 2, StdMean = 4, SPHcl = 8, Nothing = 16'
                             'combinations are also possible, refer to the weights_table to know which normalization type'
                             ' is adapted to which weights')

    args = parser.parse_args()

    input_path = args.input_path
    csv_output = args.csv_file
    weight_filepath = args.weights_path
    output_path = args.output_path
    width = args.width
    height = args.height
    norm_type = args.normtype

    input_path_valid = len(input_path) > 0 and os.path.exists(input_path) if input_path is not None else False
    csv_file_valid = csv_output.lower().endswith(".csv") if csv_output is not None else False
    sizes_valid = width > 0 and height > 0 and height % 2 == 0 and width % 2 == 0
    weights_path_valid = os.path.exists(weight_filepath) and \
                         (weight_filepath.lower().endswith(".hdf5") or weight_filepath.lower().endswith(".hdf5.best"))

    magentize = args.magentize
    torify = args.torify
    visualize = args.visualize
    overlay = args.overlay
    grav_angle = args.grav_angle

    error_occured = False
    if not input_path_valid:
        print "The given input path was not found"
        error_occured = True

    if not sizes_valid:
        print "The desired target size of the inputs before feeding the classifier are not valid"
        error_occured = True

    if not csv_file_valid:
        print "Please double check the path to store the results csv file (it must include the filename and ending by .csv)"
        error_occured = True

    if not weights_path_valid:
        print "The given path to the classifier weights file (*.hdf5) is not valid"
        error_occured = True

    if error_occured:
        print "Stopping process, errors occured"
        parser.print_help()
    else:
        print "Going to classify the folder %s" % os.path.abspath(input_path)
        if visualize:
            print "Classification results images will be stored in %s" % os.path.abspath(output_path)

        if overlay:
            print "Sources with predictions overlay colors images will be stored in %s" % os.path.abspath(os.path.join(output_path, "overlays"))

        if magentize:
            print "The Void class will be filled with magenta"

        if grav_angle:
            print "The gravity angle (part angle from the vertical that contains 50% of the SVF) will also be computed"

        print "The results will be saved in %s" % csv_output

        print "The process can take up to few hours, be patient..."
        start_time = time.time()
        classify_images(input_path, weight_filepath, csv_output, visualize, overlay, output_path, width, height,
                        norm_type, magentize, torify, grav_angle)

        elapsed_time = time.time() - start_time

        print "Done in %d seconds" % elapsed_time

