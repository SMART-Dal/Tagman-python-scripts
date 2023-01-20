# Tagman-python-scripts

Tagman Python scripts are a set of scripts to automate the repository downloading and splitting process. 

## Getting Started 

The scripts use the GitHub GraphAPI and QScored API to download and rate the repositories. As such, the following are needed to run the scripts:

### Prerequisites

- GitHub Personal Access Token 
- QScored API key
- [DesigniteJava](https://www.designite-tools.com/static/download/DJE/DesigniteJava.jar)
- [CodeSplitJava](https://github.com/tushartushar/CodeSplitJava)

### Configurations

The scripts can be configured to download scripts that match certain criteria. We have used the following criteria:

 * Lines of code: 10000 or more
 * QScored quality score threshold: 10
 * Language: Java
 * Number of stars: 40,000 or more

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

