# NER Model Evaluation
The algorithm is based on [David Batista's](https://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/) Named Entity Evaluation as in SemEval 2013 task 9.1 [repository](https://github.com/davidsbatista/NER-Evaluation)

## Evaluation Metrics
We output the F2 score for all of the metrics defined in the [blog](https://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/):
* Strict: matching entity span and type
* Exact: matching entity span, ignores entity type
* Partial: partial entity span overlap, ignores type
* Type: matching entity types, there is some overlap between the entity spans

For the partial metrics, the true positives were calculated in two ways:
* First, as implemented in the NER-Evaluation [repository](https://github.com/davidsbatista/NER-Evaluation), we give partial matches a score of 0.5 (those with a span that ovelaps but doesn't match exactly), while exact and strict matches are given a score of 1
* Second, we give a weight for each entity pair based on their overlap and sum the weights to get the weighted true positive.
  
We then use these two measures to compute two F2-scores, one for each approach
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

# Saving the Results to a spreadsheet
There is an option to save the results, along with logs of all misclassified entities, into a spreadsheet by setting the `--save_to_folder` parameter to be `True`.
When enabled, a new spreadsheet will be added to the `evaluation/results` folder (if the folder doesn't exist, it will create it). The format of the file name will be `FP_and_FN_{ANNOTATION_DATE}_{VERSION_NUMBER}` where:
* `ANNOTATION_DATE`: the date when the annotations were made (can be configured in `evaluate_model.py`)
* `VERSION_NUMBER`: the script checks how many versions of the spreadsheet were saved and adds one to the count
# To Run The Code:
1. Go to the `evaluation` folder
2. run `pip install -r requirements-eval.txt`
3. in `evaluate_model.py` set `ANNOTATION_DATE` and `FOLDERS_PATH` to point to the folder with the annotated files
4. run `python evaluate_model.py --save_to_folder=True`
5. If `--save_to_folder=True`, a new spreadsheet will be created under evaluation/results

# Benchmark
The current master as a partial weighted F2 score of 0.919037