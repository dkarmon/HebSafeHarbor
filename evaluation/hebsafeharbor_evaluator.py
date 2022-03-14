from ner_evaluation.ner_eval import Evaluator, Entity

from typing import List,Dict,Optional
import re


class HebSafeHarborEvaluator(Evaluator):
    def __init__(self,annotations_list,predicted_entities_list,txt_list,tags:List[str],annot_to_tags_mapping:Optional[Dict[str,str]],model_to_tags_mapping:Optional[Dict[str,str]]):
        
        self.tags = tags
        self.annot_to_tags_mapping = annot_to_tags_mapping
        self.model_to_tags_mapping = model_to_tags_mapping
        predicted_entities = self.extract_predicted_entities(predicted_entities_list)
        annotated_entities = self.extract_annotated_entities(annotations_list)
        super().__init__(annotated_entities,predicted_entities,tags)

    def extract_predicted_entities(self,predicted_entities): 
        agg_predictions = []

        for prediction in predicted_entities:
            predicted_entity_list = []
            for entity in prediction.granular_analyzer_results:
                entity_dict = entity.__dict__
                entity_type = self.model_to_tags_mapping.get(entity_dict['entity_type'],entity_dict['entity_type'])
                predicted_entity_list.append(Entity(entity_type,entity_dict['start'],entity_dict['end']))
            agg_predictions.append(predicted_entity_list)
        return agg_predictions

    def extract_annotated_entities(self,annotations_list):

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
