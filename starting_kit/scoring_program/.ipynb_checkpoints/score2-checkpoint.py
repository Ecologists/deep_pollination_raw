#!/usr/bin/env python

# Scoring program for the Raw Images challenge
# Isabelle Guyon and Arthur Pesah, ChaLearn, August 2014-November 2016

# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS". 
# ISABELLE GUYON, CHALEARN, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRINGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS. 
# IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL, 
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS, 
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE. 

# Some libraries and options
import os
from sys import argv,path

import libscores
import my_metric
import yaml
from libscores import *

# Default I/O directories:
root_dir = "../"
default_solution_dir = root_dir + "sample_data"
default_prediction_dir = root_dir + "sample_result_submission"
default_score_dir = root_dir + "scoring_output"
default_program_dir = root_dir + "ingestion_program"
default_data_name = "deep_pollination"

# Debug flag 0: no debug, 1: show all scores, 2: also show version amd listing of dir
debug_mode = 0

# Constant used for a missing score
missing_score = -0.999999

# Version number
scoring_version = 1.0

# =============================== MAIN ========================================

if __name__ == "__main__":
    

    #### INPUT/OUTPUT: Get input and output directory names
    if len(argv) == 2:  # Use the default data directories if no arguments are provided
        solution_dir = default_solution_dir
        prediction_dir = default_prediction_dir
        score_dir = default_score_dir
        program_dir= default_program_dir
        data_name = default_data_name
    elif len(argv) == 5: # The current default configuration of Codalab
        solution_dir = os.path.join(argv[1], 'ref')
        prediction_dir = os.path.join(argv[1], 'res')
        score_dir = argv[2]
        program_dir= argv[3]
        data_name = argv[4]
    elif len(argv) == 6:
        solution_dir = argv[1]
        prediction_dir = argv[2]
        score_dir = argv[3]
        program_dir= argv[4]
        data_name = argv[5]
    else: 
        swrite('\n*** WRONG NUMBER OF ARGUMENTS ***\n\n')
        exit(1)
        
    path.append (program_dir)
    import data_io       
    from data_io import read_solutions
        
    # Create the output directory, if it does not already exist and open output files
    mkdir(score_dir)
    score_file = open(os.path.join(score_dir, 'scores.txt'), 'w')
    html_file = open(os.path.join(score_dir, 'scores.html'), 'w')

    # Get the metric
    metric_name, scoring_function = get_metric()
    
    
    #Solution Arrays
    solutions = read_solutions(solution_dir)
 
    solution_names = ['train', 'valid', 'test']
    for i, solution_name in enumerate(solution_names):
        set_num = i + 1  # 1-indexed
        score_name = 'set%s_score' % set_num
        try:

            # Get the train prediction from the res subdirectory (must end with '.predict')
            predict_file = os.path.join(prediction_dir, data_name + '_'+solution_name+'.predict')
            if not os.path.isfile(predict_file):
                print("#--ERROR--# "+solution_name.capitalize()+" predict file NOT Found!")
                raise IOError("#--ERROR--# "+solution_name.capitalize()+" predict file NOT Found!")

            # Read the solution and prediction values into numpy arrays
            prediction = read_array(predict_file)
            solution = solutions[i]
            if (len(solution) != len(prediction)): 
                raise ValueError("Prediction length and Solution length={}".format(len(prediction), len(olution)))

            try:
                # Compute the score prescribed by the metric file 
                score = scoring_function(solution, prediction)
                print(
                    "======= Set %d" % set_num + " (" + data_name.capitalize() + "_" + solution_name + "): " + metric_name + "(" + score_name + ")=%0.12f =======" % score)
                html_file.write(
                    "<pre>======= Set %d" % set_num + " (" + data_name.capitalize() + "_" + solution_name + "): " + metric_name + "(" + score_name + ")=%0.12f =======\n" % score)
            except:
                raise Exception('Error in calculation of the specific score of the task')

            if debug_mode > 0:
                scores = compute_all_scores(solution, prediction)
                write_scores(html_file, scores)

        except Exception as inst:
            score = missing_score
            print(
                "======= Set %d" % set_num + " (" + data_name.capitalize() + "_" + solution_name + "): " + metric_name + "(" + score_name + ")=ERROR =======")
            html_file.write(
                "======= Set %d" % set_num + " (" + data_name.capitalize() + "_" + solution_name +  "): " + metric_name + "(" + score_name + ")=ERROR =======\n")
            print
            inst

    # Write score corresponding to selected task and metric to the output file
    score_file.write(score_name + ": %0.12f\n" % score)

    # End loop for solution_file in solution_names

    # Read the execution time and add it to the scores:
    try:
        metadata = yaml.load(open(os.path.join(input_dir, 'res', 'metadata'), 'r'))
        score_file.write("Duration: %0.6f\n" % metadata['elapsedTime'])
    except:
        score_file.write("Duration: 0\n")

        html_file.close()
    score_file.close()

    # Lots of debug stuff
    if debug_mode > 1:
        swrite('\n*** SCORING PROGRAM: PLATFORM SPECIFICATIONS ***\n\n')
        show_platform()
        show_io(prediction_dir, score_dir)
        show_version(scoring_version)
        
        
        
        
        
        
        
        
        