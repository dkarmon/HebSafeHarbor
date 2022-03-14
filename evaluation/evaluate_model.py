import sys,os 
#sys.path.insert(0, '../hebsafeharbor') #used in debugger mode
sys.path.append('..')

from hebsafeharbor import HebSafeHarbor
from hebsafeharbor_evaluator import HebSafeHarborEvaluator

from glob import glob
import os
import logging
import numpy as np
import pandas as pd
from difflib import SequenceMatcher
from typing import List
import re


logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

ANNOTATION_DATE = "08-03-2022"
FOLDERS_PATH = f"../../phi_evaluation_set/phi_annotations_{ANNOTATION_DATE}"
SAVE_TO_FILE = True

def fb_score(precision:float,recall:float,beta:int=2)->float:
    '''
    Compute F beta score of a model
    :param precision: the model's precision score
    :param recall: the model's recall score
    :param beta: which metric to compute (1 for F1 score, 2 for F2 score etc.)
    '''
    return (1+(beta**2))*(precision*recall)/(((beta**2)*precision)+recall)

def dict_to_dataframe(res,txt_list:List[str]) -> pd.DataFrame:
    '''
    Takes a list of Evaluator outputs and inserts it into a pandas dataframe
    :param res: a list of Evaluator results
    :returns a pandas dataframe 
    '''
    entity_list = []
    for item in res:
        for match_type in item['results'].keys():
            
            for entities in item['results'][match_type]:
                item_dict = {}
                item_dict['idx'] = item['idx']
                item_dict['match_type'] = match_type
                if match_type =='spurious':
                    item_dict['pred_entity'] = entities.e_type
                    item_dict['pred_start'] = entities.start_offset
                    item_dict['pred_end'] = entities.end_offset
                elif match_type =='missed':
                    item_dict['true_entity'] = entities.e_type
                    item_dict['true_start'] = entities.start_offset
                    item_dict['true_end'] = entities.end_offset
                else:
                    item_dict['pred_entity'] = entities[1].e_type
                    item_dict['pred_start'] = entities[1].start_offset
                    item_dict['pred_end'] = entities[1].end_offset
                    item_dict['true_entity'] = entities[0].e_type
                    item_dict['true_start'] = entities[0].start_offset
                    item_dict['true_end'] = entities[0].end_offset
                entity_list.append(item_dict)
                
    entities_df = pd.DataFrame(entity_list)
    entities_df['pred_text'] = entities_df.apply(lambda x: txt_list[x['idx']][int(x['pred_start']):int(x['pred_end'])] if x['pred_entity'] is not np.nan else None,axis=1)
    entities_df['true_text'] = entities_df.apply(lambda x: txt_list[x['idx']][int(x['true_start']):int(x['true_end'])] if x['true_entity'] is not np.nan else None,axis=1)
    return entities_df


def load_annotated_documents(folders:List[str]):
    ''' 
    Loads annotations and the raw text associated with them from a list of folders.
    Annotations files must have a prefix of .ann and text files must have the prefix .txt
    :param folders: a list of folder names from which to extract the annotated files
    :return idx_to_fname: a dictionary that maps each index to the associated folder name
    :return annotations_list: a list of lists containing the annotated entities
    :return txt_list: a list containing the raw data
    '''
    annotations_list = []
    txt_list = []
    i=0
    idx_to_fname = {}
    for folder in folders:
        annotations_fname = glob(f"{folder}*.ann", recursive = True)[0]
        idx_to_fname[i] = '/'.join(annotations_fname.split('/')[-3:-1])
        txt_fname = glob(f"{folder}*.txt", recursive = True)[0]

        with open(annotations_fname) as f:
            annotations_list.append(f.readlines())
        
        with open(txt_fname) as f:
            txt_list.append(' '.join(f.readlines()))

        i+=1

    return idx_to_fname, annotations_list,txt_list

def weighted_f2_score(entities_df:pd.DataFrame):
    FP = entities_df[entities_df['match_type'].isin(['spurious'])].shape[0]
    FN = entities_df[entities_df['match_type'].isin(['missed'])].shape[0]
    tp_df = entities_df[~entities_df['match_type'].isin(['spurious','missed'])].copy()
    tp_df['weight'] = tp_df.apply(lambda x: SequenceMatcher(None,x['pred_text'],x['true_text']).ratio(),axis=1)
    weighted_TP = tp_df['weight'].sum()
    precision = weighted_TP/(weighted_TP+FP)
    recall = weighted_TP/(weighted_TP+FN)
    print(FP,FN,weighted_TP)
    return fb_score(precision=precision,recall=recall)

def add_context(df,txt_list,idx_to_fname,col_prefix:List[str],group_results=True):
    for prefix in col_prefix:
        col_name = f'{prefix}_context'
        df[col_name] = df.apply(lambda x: txt_list[x['idx']][(max(0,int(x[f'{prefix}_start'])-50)):(int(x[f'{prefix}_end'])+50)],axis=1)

    if group_results:
        if len(col_prefix)==1:
            group_cols = ['idx',f'{prefix}_entity']
            value_cols = [f'{prefix}_entity',f'{prefix}_context']
            df = df.groupby(group_cols)[value_cols].agg(list).reset_index() #.to_csv('spurious.csv',encoding = 'utf-8-sig')
        else:
            print("Can't group results, returning a non-aggregated dataframe")

    df['folder_name'] = df['idx'].apply(lambda x: '/'.join(idx_to_fname[x].split('/')[-3:]))
    return df


def get_output_version():
    files = glob(f"{os.getcwd()}/results/FP_and_FN_{ANNOTATION_DATE}*")
    if len(files)>0 and len(re.findall(r'_v(\d).xlsx',files[0]))>0:
        version = max(np.array([int(re.findall(r'_v(\d).xlsx',x)[0])+1 for x in files]))
    else:
        version = 1
    return version

def save_results_to_file(entities_df,f2_df,txt_list,idx_to_fname):

    try: 
        os.mkdir('./results')
    except OSError as error: 
        print(error) 
        pass

    #weighted_f2_score = weighted_f2_score(entities_df)
    #f2_df = pd.DataFrame([{'weighted f2 score':weighted_f2_score}])

    spurious_df = entities_df[entities_df['match_type']=='spurious'].copy()
    spurious_df = add_context(spurious_df,idx_to_fname,txt_list,col_prefix=['pred'])
    
    missed_df = entities_df[entities_df['match_type']=='missed'].copy()
    missed_df = add_context(missed_df,txt_list,idx_to_fname,col_prefix=['true'])

    missclass_df = entities_df[(entities_df['match_type']!='missed') & (entities_df['match_type']!='spurious') & (entities_df['pred_entity']!=entities_df['true_entity'])].copy()
    missclass_df = add_context(missclass_df,txt_list,idx_to_fname,col_prefix=['true','pred'],group_results=False)
    missclass_df = missclass_df.sort_values(['idx','match_type','true_entity','pred_entity'])[['idx','match_type','pred_entity','true_entity','pred_text','true_text','true_context','folder_name']]

    partial_df = entities_df[(entities_df['match_type']=='type')].copy()
    partial_df = add_context(partial_df,txt_list,idx_to_fname,col_prefix=['true','pred'],group_results=False)
    partial_df = partial_df.sort_values(['idx','match_type','true_entity','pred_entity'])[['idx','match_type','pred_entity','true_entity','pred_text','true_text','true_context','folder_name']]

    version = get_output_version()

    options = {}
    options['strings_to_formulas'] = False
    options['strings_to_urls'] = False

    with pd.ExcelWriter(f'./results/FP_and_FN_{ANNOTATION_DATE}_v{version}.xlsx',engine='xlsxwriter',engine_kwargs={'options':options}) as writer:
        f2_df.to_excel(writer,sheet_name='F2 score',index=True,encoding = 'utf-8-sig')
        spurious_df.to_excel(writer,sheet_name='FP',index=False,encoding = 'utf-8-sig')
        missed_df.to_excel(writer,sheet_name='FN',index=False,encoding = 'utf-8-sig')
        missclass_df.to_excel(writer,sheet_name='Misclassification',index=False,encoding = 'utf-8-sig')
        partial_df.to_excel(writer,sheet_name='Partial match',index=False,encoding = 'utf-8-sig')
    

def compute_f2_metrics(metrics,entities_df):
    f2_dict = {}
    for k in metrics[0].keys():
        #print(f'{k} F2 score: ',fb_score(metrics[0][k]['precision'],metrics[0][k]['recall']))
        f2_dict[k] = fb_score(metrics[0][k]['precision'],metrics[0][k]['recall'])
    f2_dict['weighted_f2'] = weighted_f2_score(entities_df)
    f2_df = pd.DataFrame([f2_dict])
    return f2_df

    
def main():
    annot_entity_mapping = {'ADDRESS':'LOC','ORGANIZATION':'ORG'}
    sh_entity_mapping = {'MEDICAL_DATE':'DATE','BIRTH_DATE':'DATE','CITY':'LOC','COUNTRY':'LOC',
                     'EMAIL_ADDRESS':'EMAIL','ISRAELI_ID_NUMBER':'ID','PER':'NAME','PERS':'NAME',
                     'PHONE_NUMBER':'PHONE_OR_FAX','FAC':'LOC','GPE':'LOC','MISC__AFF':'ETHNICITY'}

    # Entity types included in the analysis
    tags = ['LOC','EMAIL','ID','ORG','DATE','NAME','ETHNICITY','PHONE_OR_FAX','URL','IP_ADDRESS'] 


    
    folders = glob(f"{FOLDERS_PATH}/*/", recursive = True)

    hebrew_phi = HebSafeHarbor()
    idx_to_fname, annotations_list,txt_list = load_annotated_documents(folders)

    doc_list = [{"text": txt} for txt in txt_list]
    predicted_entities = hebrew_phi(doc_list)
    

    evaluator = HebSafeHarborEvaluator(annotations_list,predicted_entities,txt_list,tags,annot_entity_mapping,sh_entity_mapping)
    metrics = evaluator.evaluate()

    entities_df = dict_to_dataframe(metrics[2],txt_list)

    f2_df = compute_f2_metrics(metrics,entities_df)
    print(f"Partial weighted f2-score: {f2_df['weighted_f2'].iloc[0]}")

    if SAVE_TO_FILE:
        save_results_to_file(entities_df,f2_df,txt_list,idx_to_fname) 


if __name__ == "__main__":
    main()