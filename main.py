from cnn.cnn_main import classify_images
import argparse
import os

default_weights = "images/a.hdf5"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify pixels of every images in the given folder '
                                                 '(take care to rectify the optic first) and compute the views factors.'
                                                 'Outputs values in a csv files.')

    parser.add_argument('-i', '--input-path', type=str, required=True,
                        help='The path to the images folder')

    parser.add_argument('--csv-file', type=str, required="True",
                        help='The path to the csv results file')

    parser.add_argument('-o', '--output-path', type=str, nargs='?', default="outputs/predictions",
                        help='The path where to the output folder to store the classification results images')

    parser.add_argument('--weights-path', type=str, nargs='?', default=default_weights,
                        help='The path to the file that contains the weights of the classifier nodes')

    parser.add_argument('-v', '--visualize', action='store_true',
                        help="Stores the classification results in the specified output path")

    parser.add_argument('-m', '--magentize', action='store_true',
                        help="Use magenta color to fill the void class")

    parser.add_argument('--width', type=int, nargs='?', default=480,
                       help='The width the images will be resized before going through'
                            ' the classifier (depends on which weights are loaded)')

    parser.add_argument('--height', type=int, nargs='?', default=480,
                        help='The height the images will be resized before going through'
                             ' the classifier (depends on which weights are loaded)')

    parser.add_argument('-n', '--nblbl', type=int, nargs='?', default=4,
                        help='The number of labels (that impact the labels matrix given to'
                             ' the classifier) usually must be 4 (depends on which weights are loaded)')

    #parser.print_help()
    args = parser.parse_args()

    #print args

    input_path = args.input_path
    csv_output = args.csv_file
    weight_filepath = args.weights_path
    output_path = args.output_path
    width = args.width
    height = args.height
    nblbl = args.nblbl

    input_path_valid = len(input_path) > 0 and os.path.exists(input_path) if input_path is not None else False
    csv_file_valid = csv_output.lower().endswith(".csv") if csv_output is not None else False
    sizes_valid = width > 0 and height > 0 and height % 2 == 0 and width % 2 == 0
    nblbl_valid = nblbl > 0
    weights_path_valid = os.path.exists(weight_filepath) and weight_filepath.lower().endswith(".hdf5")

    magentize = args.magentize
    visualize = args.visualize > 0
    predictions_path = args.output_path

    error_occured = False
    if not input_path_valid:
        print "The given input path was not found"
        error_occured = True

    if not sizes_valid:
        print "The desired target size of the inputs before feeding the classifier are not valid"
        error_occured = True

    if not csv_file_valid:
        print "Please specify a path to store the results csv file"
        error_occured = True

    if not weights_path_valid:
        print "The given path to the classifier weights file (*.hdf5) is not valid"
        error_occured = True

    if error_occured:
        print "Stopping process, errors occured"
        parser.print_help()
    else:

        print "Going to classify the folder %s" % input_path
        if visualize:
            print "Classification results images will be stored in %s" % predictions_path

        if magentize:
            print "The Void class will be filled with magenta"

        print "The results will be saved in %s" % output_path

        print "The process can take up to few hours, be patient..."
        classify_images(input_path, weight_filepath, csv_output, visualize, output_path, width, height, nblbl, magentize)
        pass

