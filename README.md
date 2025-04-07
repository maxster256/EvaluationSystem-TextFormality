This is the solution for the test assignment for JetBrain AI.<br/>
To read the documentation and report, open the DocumentationAndReport.pdf file.<br/>
To read the content of the project, open the EvaluationSystem-TextFormality folder. This project consists of 2 python scripts:<br/>
- LocalDataset.py - module where the data is prepared. Run this script if you want to load the data to the local, although it's already loaded into 2 .csv files (osyvokon_test.csv and osyvokon_train.csv)
- main.py - the main module where the evaluation system is located

Instruction to run this project locally:
- download this repository
- unpack the downloaded zip file and choose "EvaluationSystem-TextFormality" folder as open new project in PyCharm IDE
- make sure that Python 3.12 is set as an interpreter in this project
- install necessary libraries using keyboard shortcut alt+shift+enter or manually through the terminal:
  * pip install -U scikit-learn
  * pip install transformers
  * pip install datasets
  * copy and paste a PyTorch installation command from this site https://pytorch.org/
