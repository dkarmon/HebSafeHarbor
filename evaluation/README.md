# NER Model Evaluation
The algorithm is based on [David Batista's](https://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/) Named Entity Evaluation as in SemEval 2013 task 9.1 [repository](https://github.com/davidsbatista/NER-Evaluation)

# Annotated Files Structure
The structure of the annotated files is assumed to be in the following structure:
```
|-- annotated_files
|   |-- doc1
|   |   |-- file_name1.ann
|   |   |-- file_name1.json
|   |   |-- file_name1.txt
|   |-- doc2
|   |   |-- file_name2.ann
|   |   |-- file_name2.json
|   |   |-- file_name2.txt
```
etc. the folders and file names could be anything but it is important to maintain this structure.

# To Run The Code:
1. Go to the `evaluation` folder
2. in `evaluate_model.py` set `FOLDERS_PATH` to point to the folder with the annotated files
3. run `python evaluate_model.py`