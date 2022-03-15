import sys,os 
#sys.path.insert(0, '../hebsafeharbor') #used in debugger mode
sys.path.append('..')

from hebsafeharbor import HebSafeHarbor
from hebsafeharbor_evaluator import HebSafeHarborEvaluator, compute_f2_metrics
from evaluator_utils import TAG_TYPES, ANNOT_TO_TAG_MAPPING, MODEL_TO_TAG_MAPPING,save_results_to_file,dict_to_dataframe,load_annotated_documents

from glob import glob
import logging
import pandas as pd
import click
from timeit import default_timer as timer



logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

ANNOTATION_DATE = "08-03-2022"
FOLDERS_PATH = f"../../phi_evaluation_set/phi_annotations_{ANNOTATION_DATE}"

@click.command()
@click.option('--save_to_file',default=True,help="save evaluation results to a spreadsheet")
def main(save_to_file):
    
    # Load the annotated files
    annotated_folders_list = glob(f"{FOLDERS_PATH}/*/", recursive = True)
    print(f'Number of annotated documents: {len(annotated_folders_list)}')
    idx_to_fname, annotations_list,txt_list = load_annotated_documents(annotated_folders_list)
    print(f'Loaded annotated files')

    # Run HebSafeHarbor on raw text
    print('Running HebSafeHarbor...')
    hebrew_phi = HebSafeHarbor()
    doc_list = [{"text": txt} for txt in txt_list]
    start = timer()
    predicted_entities = hebrew_phi(doc_list)
    end = timer()
    elapsed_time = round((end-start)/60,2)
    print(f'Model took {elapsed_time} minutes to complete')

    
    # Model Evaluation
    print('Evaluating the model')
    evaluator = HebSafeHarborEvaluator(annotations_list,predicted_entities,txt_list)
    metrics = evaluator.evaluate()
    entities_df = dict_to_dataframe(metrics[2],txt_list)
    f2_df = compute_f2_metrics(metrics,entities_df)
    print(f"Partial weighted f2-score: {f2_df['weighted_f2'].iloc[0]}")

    # Save results to file
    if save_to_file:
        print('Saving results to a file')
        fname_prefix =  f'results/FP_and_FN_{ANNOTATION_DATE}'
        save_results_to_file(entities_df,f2_df,txt_list,idx_to_fname,fname_prefix) 


if __name__ == "__main__":
    main(['--save_to_file',True])