from ner_evaluation.ner_eval import Evaluator, Entity
from difflib import SequenceMatcher
from evaluator_utils import TAG_TYPES,ANNOT_TO_TAG_MAPPING,MODEL_TO_TAG_MAPPING
from typing import List,Dict,Optional
import re
import pandas as pd


class HebSafeHarborEvaluator(Evaluator):
    '''
    The class compares predicted entities to annotated entities and computes evaluation metrics
    '''
    def __init__(self,annotations_list:List[List[str]],predicted_entities_list,txt_list:List[str]):
        '''
        :param annotations_list: a list of annotations for each annotated document
        :param predicted_entities_list: a list of predicted entities found in each document
        :param txt_list: a list of raw text
        '''
        self.tags = TAG_TYPES
        self.annot_to_tags_mapping = ANNOT_TO_TAG_MAPPING
        self.model_to_tags_mapping = MODEL_TO_TAG_MAPPING
        predicted_entities = self.extract_predicted_entities(predicted_entities_list)
        annotated_entities = self.extract_annotated_entities(annotations_list)
        super().__init__(annotated_entities,predicted_entities,self.tags)

    def extract_predicted_entities(self,predicted_entities)->List[List[str]]: 
        '''
        Extract the predicted entities into a list
        :param predicted_entities: a list of predicted entities (the output of the HebSafeHarbor model)
        :returns a list of lists of predicted entities (a separate list for each document)
        '''
        agg_predictions = []

        for prediction in predicted_entities:
            predicted_entity_list = []
            for entity in prediction.granular_analyzer_results:
                entity_dict = entity.__dict__
                entity_type = self.model_to_tags_mapping.get(entity_dict['entity_type'],entity_dict['entity_type'])
                predicted_entity_list.append(Entity(entity_type,entity_dict['start'],entity_dict['end']))
            agg_predictions.append(predicted_entity_list)
        return agg_predictions

    def extract_annotated_entities(self,annotations_list:List[List[str]])->List[List[str]]:
        '''
        Extracts and filters the list of annotated entities
        :param annotations_list: a list of all the annotated entities 
        :returns a filtered list of all annotated entities
        '''

        monitored_entities = self.tags
        if self.annot_to_tags_mapping:
            monitored_entities += list(self.annot_to_tags_mapping.keys())

        agg_true = []

        for annotations in annotations_list:
            entity_list = []
            for a in annotations:
                entity = re.split('\t|\n|\s',a) 
                if (len(entity[0])<1) or (entity[0][0]!='T') or not(entity[1] in monitored_entities):
                    continue
                entity_type = self.annot_to_tags_mapping.get(entity[1],entity[1])
                entity_list.append(Entity(entity_type,int(entity[2]),int(entity[3])))
            agg_true.append(entity_list)
        return agg_true

def fb_score(precision:float,recall:float,beta:int=2)->float:
    '''
    Compute F beta score of a model
    :param precision: the model's precision score
    :param recall: the model's recall score
    :param beta: which metric to compute (1 for F1 score, 2 for F2 score etc.)
    '''
    return (1+(beta**2))*(precision*recall)/(((beta**2)*precision)+recall)


def weighted_f2_score(entities_df:pd.DataFrame) -> float:
    '''
    Compute a weighted F2 score where each partial match gets a score based on the length of overlap
    :param entities_df: a data frame that contains the predicted and annotated entities 
    :returns a weighted F2 score
    '''

    FP = entities_df[entities_df['match_type'].isin(['spurious'])].shape[0]
    FN = entities_df[entities_df['match_type'].isin(['missed'])].shape[0]
    tp_df = entities_df[~entities_df['match_type'].isin(['spurious','missed'])].copy()
    tp_df['weight'] = tp_df.apply(lambda x: SequenceMatcher(None,x['pred_text'],x['true_text']).ratio(),axis=1)
    weighted_TP = tp_df['weight'].sum()
    precision = weighted_TP/(weighted_TP+FP)
    recall = weighted_TP/(weighted_TP+FN)
    return fb_score(precision=precision,recall=recall)


def compute_f2_metrics(metrics,entities_df:pd.DataFrame)->pd.DataFrame:
    '''
    Returns a dataframe with the different F2 scores (strict, exact, partial, entity and weighted partial)
    :param df: a dataframe with all the entities and their type of match
    :returns a dataframe containing the results
    '''
    f2_dict = {}
    for k in metrics[0].keys():
        #print(f'{k} F2 score: ',fb_score(metrics[0][k]['precision'],metrics[0][k]['recall']))
        f2_dict[k] = fb_score(metrics[0][k]['precision'],metrics[0][k]['recall'])
    f2_dict['weighted_f2'] = weighted_f2_score(entities_df)
    f2_df = pd.DataFrame([f2_dict])
    return f2_df
