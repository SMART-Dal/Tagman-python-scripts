# Tagman-python-scripts

Tagman Python scripts are a set of scripts to automate the repository downloading and splitting process. 

## Getting Started 

The scripts use the GitHub GraphAPI and QScored API to download and rate the repositories. As such, the following are needed to run the scripts:

### Prerequisites

Github Personal Access Token 
QScored API key
Designite Jar

### Configurations

The scripts can be configured to download scripts that match certain criteria. We have used the following criteria:

 * Lines Of Code : 10000 or more
 * QScored qaulity Score Threshold : 10
 * Language : Java
 * Number of Stars : 40,000 or more

These can be easily configured in each file with their corresponding constants. 

## Usage

1. Clone the repository 

``` console
git clone https://github.com/SMART-Dal/Tagman-python-scripts.git
```

2. Run the script

``` console
python download.py
python download-repo.py
python data_curation_main.py
```

The scripts rate repositories from QScored, download them from Github, run them through Designite to generate smells and split the code into classes and methods. 
