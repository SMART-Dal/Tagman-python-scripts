# This program generates required data and put the data into required form to apply machine/deep learning
# in a step by step way.
# Set the parameters first before running it.
# Steps are in general independent from another steps except dependence on data consumed by some steps
# are generated by previous steps.

# -- imports --
import sys

import tokenizer_runner
import java_designite_runner
import java_codeSplit_runner

# --

# -- Parameters --
DATA_BASE_PATH = "C:/Users/Himesh/Documents/thesis"

JAVA_REPO_SOURCE_FOLDER = DATA_BASE_PATH+'/output'
JAVA_SMELLS_RESULTS_FOLDER = DATA_BASE_PATH + '/designite_out_java'
DESIGNITE_JAVA_JAR_PATH = r'C:/Users/Himesh/OneDrive - Dalhousie University/Thesis/Designite/DesigniteJava.jar'

JAVA_CODE_SPLIT_OUT_FOLDER_CLASS = DATA_BASE_PATH + '/codesplit_java_class'
JAVA_CODE_SPLIT_OUT_FOLDER_METHOD = DATA_BASE_PATH + '/codesplit_java_method'
JAVA_CODE_SPLIT_MODE_CLASS = "class"
JAVA_CODE_SPLIT_MODE_METHOD = "method"
JAVA_CODE_SPLIT_EXE_PATH = 'C:/Users/Himesh/OneDrive - Dalhousie University/Thesis/CodeSplit_1_1_0_0/CodeSplitJava-master/target/CodeSplitJava.jar'

JAVA_LEARNING_DATA_FOLDER_BASE = DATA_BASE_PATH + '/smellML_data_java'

JAVA_TOKENIZER_OUT_PATH = DATA_BASE_PATH + '/tokenizer_out_java'
# --

if __name__ == "__main__":

    java_designite_runner.analyze_repositories(JAVA_REPO_SOURCE_FOLDER, JAVA_SMELLS_RESULTS_FOLDER, DESIGNITE_JAVA_JAR_PATH)

    java_codeSplit_runner.java_code_split(JAVA_REPO_SOURCE_FOLDER, JAVA_CODE_SPLIT_MODE_CLASS,JAVA_CODE_SPLIT_OUT_FOLDER_CLASS, JAVA_CODE_SPLIT_EXE_PATH)

    java_codeSplit_runner.java_code_split(JAVA_REPO_SOURCE_FOLDER, JAVA_CODE_SPLIT_MODE_METHOD,JAVA_CODE_SPLIT_OUT_FOLDER_METHOD, JAVA_CODE_SPLIT_EXE_PATH)
