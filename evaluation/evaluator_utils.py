import pandas as pd
import numpy as np
from typing import List,Tuple,Dict
from glob import glob
import os
import re

# Entity types included in the analysis
TAG_TYPES = ['LOC','EMAIL','ID','ORG','DATE','NAME','ETHNICITY','PHONE_OR_FAX','URL','IP_ADDRESS'] 

ANNOT_TO_TAG_MAPPING = {'ADDRESS':'LOC','ORGANIZATION':'ORG'}
MODEL_TO_TAG_MAPPING = {'MEDICAL_DATE':'DATE','BIRTH_DATE':'DATE','CITY':'LOC','COUNTRY':'LOC',
                     'EMAIL_ADDRESS':'EMAIL','ISRAELI_ID_NUMBER':'ID','PER':'NAME','PERS':'NAME',
                     'PHONE_NUMBER':'PHONE_OR_FAX','FAC':'LOC','GPE':'LOC','MISC__AFF':'ETHNICITY'}


def add_context(df:pd.DataFrame,txt_list:List[str],idx_to_fname:Dict[int,str],col_prefix:List[str],group_results:bool=True) -> pd.DataFrame:
    '''
    Adds context around the span of each token in the dataframe
    :param df: a dataframe that contains all of the predicted and annotated entities
    :param txt_list: a list of raw texts for the entities in the dataframe
    :param idx_do_dict: a dictionary that maps each text id to a folder name
    :param col_prefix: the source of the entities for which to extract context ('pred' for predicted entities or 'true' for annotated entities)
    :param group_results: group the entities by index and entity type? (True/False)
    :returns a dataframe with context for each token
    '''

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


def get_output_version(file_prefix:str) -> int:
    '''
    Checkes the latest version of the spreadsheets and returns the following version number
    '''

    files = glob(f"{os.getcwd()}/{file_prefix}*")
    if len(files)>0 and len(re.findall(r'_v(\d).xlsx',files[0]))>0:
        version = max(np.array([int(re.findall(r'_v(\d).xlsx',x)[0])+1 for x in files]))
    else:
        version = 1
    return version

def save_results_to_file(entities_df:pd.DataFrame,f2_df:pd.DataFrame,txt_list:List[str],idx_to_fname:Dict[str,str],fname_prefix:str):
    ''' 
    Saves the evaluation results along with a detailed list of misclassifications to a preadsheet
    :param entities_df: a dataframe that lists all of the entities in the predeicted and annotated sets
    :param f2_df: a dataframe that contains the f2 score of the model
    :param txt_list: a list of raw texts
    :param idx_to_fname: a dictionary that maps each document index to its folder's name
    :param fname_prefix: the prefix of the file name to be saved
    '''

    # create a directory to store the results (if it doesn't exist already)
    try: 
        os.mkdir('./results')
    except OSError as error: 
        print(error) 
        pass

    # create a data frame with all the spurious entities (FP)
    spurious_df = entities_df[entities_df['match_type']=='spurious'].copy()
    spurious_df = add_context(spurious_df,idx_to_fname,txt_list,col_prefix=['pred'])
    
    # create a data frame with all the missing entities (FN)
    missed_df = entities_df[entities_df['match_type']=='missed'].copy()
    missed_df = add_context(missed_df,txt_list,idx_to_fname,col_prefix=['true'])

    # create a dataframe with all misclassified entities
    missclass_df = entities_df[(entities_df['match_type']!='missed') & (entities_df['match_type']!='spurious') & (entities_df['pred_entity']!=entities_df['true_entity'])].copy()
    missclass_df = add_context(missclass_df,txt_list,idx_to_fname,col_prefix=['true','pred'],group_results=False)
    missclass_df = missclass_df.sort_values(['idx','match_type','true_entity','pred_entity'])[['idx','match_type','pred_entity','true_entity','pred_text','true_text','true_context','folder_name']]

    # create a dataframe with all partial matches
    partial_df = entities_df[(entities_df['match_type']=='type')].copy()
    partial_df = add_context(partial_df,txt_list,idx_to_fname,col_prefix=['true','pred'],group_results=False)
    partial_df = partial_df.sort_values(['idx','match_type','true_entity','pred_entity'])[['idx','match_type','pred_entity','true_entity','pred_text','true_text','true_context','folder_name']]

    version = get_output_version(fname_prefix)
    file_name = f'./{fname_prefix}_v{version}.xlsx'
    print(f'results are saved into {file_name}')

    options = {}
    options['strings_to_formulas'] = False
    options['strings_to_urls'] = False

    with pd.ExcelWriter(file_name,engine='xlsxwriter',engine_kwargs={'options':options}) as writer:
        f2_df.to_excel(writer,sheet_name='F2 score',index=True,encoding = 'utf-8-sig')
        spurious_df.to_excel(writer,sheet_name='FP',index=False,encoding = 'utf-8-sig')
        missed_df.to_excel(writer,sheet_name='FN',index=False,encoding = 'utf-8-sig')
        missclass_df.to_excel(writer,sheet_name='Misclassification',index=False,encoding = 'utf-8-sig')
        partial_df.to_excel(writer,sheet_name='Partial match',index=False,encoding = 'utf-8-sig')
    
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


def load_annotated_documents(folders:List[str]) -> Tuple[Dict[int,str],List[List[str]],List[str]]:
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
