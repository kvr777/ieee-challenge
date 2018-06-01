# IEEE Challenge solution

This repository contains a 4th place solution for [IEEE Investment Ranking Challenge](https://www.crowdai.org/challenges/ieee-investment-ranking-challenge).

## Getting Started

At the first step it is to clone this repository and keep the structure of files:
1. ieee folder - contains two scripts (core.py , technical_indicators.py) with the main logic of solution
2. docs folder  - this folder contain the paper with the brief description of research and solution
3. optimization folder - the results of optimization stage. Some optimization steps (e.g feature selection) can take 30+ hours and instead of starting from scratch this step it is possible just to load dataframe with the results
4. results folder - folder with the file for submission
5. final_pipeline.ipynb - notebook with the pipline of final solution.
6. requirements.txt - use this file to install the libraries needed to launch the solution


### Prerequisites

Python version 3.6 and above. I use windows 10 to create the solution. If you use linux based systems, the results could be different

### Installing

1. Copy the solution from this repository 
2. Create folder "data" in root directory of the solution. 
3. Download the files "full_dataset.csv" and "prediction_template.csv" from [ieee challenge page](https://www.crowdai.org/challenges/ieee-investment-ranking-challenge/dataset_files) and place them in data folder
4. Check that you have all libraries from requirements.txt file


## License

This project is licensed under the MIT License
