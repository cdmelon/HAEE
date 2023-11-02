import json
import codecs
import logging
import os
from typing import List
import torch

import tqdm

from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

UNK = "<unknown>"
DOC_TYPE_TRAIN = "train"
DOC_TYPE_TEST = "test"
DOC_TYPE_VALID = "valid"
DATA_TYPE_NEW = "new"
DATA_TYPE_ONTOED = "ontoed"


class InputExample(object):
    """A single training/test example for multiple choice"""

    def __init__(self, example_id, tokens, triggerL, triggerR, label=None, relation=None):
        """Constructs a InputExample.
        Args:
            example_id: str. unique id for the example.
            tokens: list of tokens.
            triggerL: int. beginning position of the trigger
            triggerR: int. endding position of the trigger
            label: (Optional) string. The label of the example. This should be specified for train and valid examples, but not for test examples.
        """
        self.example_id = example_id
        self.tokens = tokens
        self.triggerL = triggerL
        self.triggerR = triggerR
        self.label = label
        self.relation = relation


class InputFeatures(object):
    def __init__(self, example_id, input_ids, input_mask, segment_ids, label):
        self.example_id = example_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label = label


class DataProcessor(object):
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_valid_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the valid set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class NewModelProcessor(DataProcessor):
    """Processor for the OntoEvent data set."""

    def get_train_examples(self, data_dir):
        logger.info("LOOKING AT {} train".format(data_dir))
        # return self.create_examples(os.path.join(data_dir, 'maven_train.json'), DOC_TYPE_TRAIN, DATA_TYPE_NEW)
        return self.create_examples(os.path.join(data_dir, 'event_dict_train_data.json'), DOC_TYPE_TRAIN, DATA_TYPE_ONTOED)

        # return self.create_examples(os.path.join(data_dir, 'output_train_new.json'), DOC_TYPE_TRAIN, DATA_TYPE_NEW)+self.create_examples(os.path.join(data_dir, 'event_dict_train_data.json'), DOC_TYPE_TRAIN, DATA_TYPE_ONTOED)

    def get_valid_examples(self, data_dir):
        logger.info("LOOKING AT {} valid".format(data_dir))
        # return self.create_examples(os.path.join(data_dir, 'maven_test.json'), DOC_TYPE_VALID, DATA_TYPE_NEW)

        return self.create_examples(os.path.join(data_dir, 'event_dict_valid_data.json'), DOC_TYPE_VALID, DATA_TYPE_ONTOED) # self.create_examples(os.path.join(data_dir, 'output_test.json'), DOC_TYPE_VALID, DATA_TYPE_NEW)

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        # return self.create_examples(os.path.join(data_dir, 'maven_test.json'), DOC_TYPE_TEST, DATA_TYPE_NEW)

        return self.create_examples(os.path.join(data_dir, 'event_dict_test_data.json'), DOC_TYPE_TEST, DATA_TYPE_ONTOED)

    def str_process(self, st):
        if "." in st:
            st = st.split(".")[1]
        if "-" in st:
            st = st.replace("-", "_")
        st = st.lower()
        return st

    def get_labels(self):
        file_path = LABEL_PATH
        data = json2dicts(file_path)[0]
        list_label = [key for key in data.keys()]
        for i, _ in enumerate(list_label):
            list_label[i]=self.str_process(list_label[i])
        # data = json2dicts("ModelData/output_train_new.json")[0]
        # for i in data.keys():
        #     key = self.str_process(i)
        #     if key not in list_label:
        #         list_label.append(key)
        return list_label

    def create_examples(self, file_path, doc_type, data_type = DATA_TYPE_NEW):
        """Creates examples for the training and valid sets."""
        examples = []
        data = json2dicts(file_path)[0]
        turn = 0

        if data_type == DATA_TYPE_NEW:
            for event_type in data.keys():
                for event_instance in data[event_type]:

                    # turn += 1
                    # if wandb_config is not None and turn % wandb_config.rate != 1 and doc_type == DOC_TYPE_TRAIN:
                    #     continue

                    if (type(event_instance['offset']) == int):
                        triL = event_instance['offset']
                        triR = triL
                    else:
                        triL = event_instance['offset'][0]
                        triR = event_instance['offset'][1]
                    event_instance['relation'] = {}
                    if 'event_type' not in event_instance:
                        event_instance['event_type'] = self.str_process(event_type)
                    else:
                        event_instance['event_type'] = self.str_process(event_instance['event_type'])
                    examples.append(
                        InputExample(
                            example_id=event_instance['sentences_id'],
                            tokens=event_instance['sentences_token'],
                            triggerL=triL,
                            triggerR=triR,
                            label=event_instance['event_type'],
                            relation=event_instance['relation'],
                        )
                    )
        elif data_type == DATA_TYPE_ONTOED:
            rel = json2dicts("ModelData/event_relation.json")[0]
            for event_type in data.keys():
                for event_instance in data[event_type]:

                    # turn += 1
                    # if wandb_config is not None and turn % wandb_config.rate != 1 and doc_type == DOC_TYPE_TRAIN:
                    #     continue

                    sid = event_instance['sent_id']
                    if type(event_instance['sent_id'] != str):
                        sid = str(sid)
                        # e_id = "%s-+-%s-+-%s" % (set_type, event_instance['doc_id'], sid)
                    e_id = "%s-+-%s-+-%s" % (event_instance['event_type'], event_instance['doc_id'], sid)
                    if (type(event_instance['trigger_pos']) == int):
                        triL = event_instance['trigger_pos']
                        triR = triL
                    else:
                        triL = event_instance['trigger_pos'][0]
                        triR = event_instance['trigger_pos'][1]
                    event_instance['relation'] = {"causal_relations": [], "subevent_relations": []}
                    for i in rel["CAUSE"] + rel["CAUSEDBY"]:
                        h_id = "%s-+-%s-+-%s" % (i[0]['event_type'], i[0]['doc_id'], i[0]['sent_id'])
                        t_id = "%s-+-%s-+-%s" % (i[1]['event_type'], i[1]['doc_id'], i[1]['sent_id'])
                        if e_id in (h_id,t_id):
                            event_instance['relation']["causal_relations"].append([self.str_process(i[0]['event_type']), self.str_process(i[1]['event_type'])])
                    event_instance['event_type'] = self.str_process(event_instance['event_type'])
                    examples.append(
                        InputExample(
                            example_id=e_id,
                            tokens=event_instance['event_mention_tokens'],
                            triggerL=triL,
                            triggerR=triR,
                            label=event_instance['event_type'],
                            relation=event_instance['relation'],
                        )
                    )
        return examples


def json2dicts(jsonFile):
    data = []
    with codecs.open(jsonFile, "r", "utf-8") as f:
        for line in f:
            dic = json.loads(line)
            data.append(dic)
    return data


def convert_examples_to_features(
        examples: List[InputExample],
        label_list: List[str],
        max_length: int,
        tokenizer: PreTrainedTokenizer,
        pad_token_segment_id=0,
        pad_on_left=False,
        pad_token=0,
        mask_padding_with_zero=True,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """

    # 总的标签集合
    label_map = {label: i for i, label in enumerate(label_list)}

    list_event_id = set()
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        list_event_id.add(example.example_id)
    example_id_map = {example_id: i for i, example_id in enumerate(list_event_id)}  # eid counts from 0

    filename = "TestData/event_map"
    with open(filename, 'w') as file_obj:
        json.dump(label_map, file_obj)

    relation_map = {'CAUSE': 0, 'CAUSEDBY': 1, 'SUBEVENT': 2, 'SUPEREVENT': 3}
    max_rel_per_ins = 30

    dict_rel2events = json2dicts(RELATION_PATH)[0]
    for i in label_map.keys():
        rel_event_ids[label_map[i]] = {relation_map['CAUSE']: [], relation_map['CAUSEDBY']: [], relation_map['SUBEVENT']: [],
                    relation_map['SUPEREVENT']: []}
    for i in dict_rel2events['causal_relations']:
        rel_event_ids[label_map[i[0]]][relation_map['CAUSE']].append([label_map[i[0]],label_map[i[1]]])
        rel_event_ids[label_map[i[1]]][relation_map['CAUSEDBY']].append([label_map[i[0]], label_map[i[1]]])
    for i in dict_rel2events['subevent_relations']:
        rel_event_ids[label_map[i[0]]][relation_map['SUBEVENT']].append([label_map[i[0]],label_map[i[1]]])
        rel_event_ids[label_map[i[1]]][relation_map['SUPEREVENT']].append([label_map[i[0]], label_map[i[1]]])

    features = []

    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # the token inputs with trigger mark
        if example.triggerL != -1:
            textL = tokenizer.tokenize(" ".join(example.tokens[:example.triggerL]))
            textTrg = tokenizer.tokenize(" ".join(example.tokens[example.triggerL:example.triggerR]))
            textR = tokenizer.tokenize(" ".join(example.tokens[example.triggerR:]))
            text = textL + ['[<trigger>]'] + textTrg + ['[</trigger>]'] + textR
        else:
            text = tokenizer.tokenize(" ".join(example.tokens[:]))
        # text = textL + ['[<mask>]'] + textR
        # # the raw token inputs
        # text = tokenizer.tokenize(" ".join(example.tokens[:]))

        inputs = tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=max_length, return_token_type_ids=True
        )

        if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
            logger.info(
                "Attention! You are cropping tokens."
            )

        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length
        assert len(attention_mask) == max_length
        assert len(token_type_ids) == max_length

        example_id = example_id_map[example.example_id]  # 实例序号从0开始

        relation = {relation_map['CAUSE']: [], relation_map['CAUSEDBY']: [], relation_map['SUBEVENT']: [],
                    relation_map['SUPEREVENT']: []}

        if len(example.relation) != 0:
            if example.label in label_map:
                label = label_map[example.label]  # 类型序号从0开始
                for item in example.relation["causal_relations"]:
                    if example.label == item[0] and item[0] in label_map and item[1] in label_map:
                        relation[relation_map['CAUSE']].append([label_map[item[0]], label_map[item[1]]])
                    elif example.label == item[1] and item[0] in label_map and item[1] in label_map:
                        relation[relation_map['CAUSEDBY']].append([label_map[item[0]], label_map[item[1]]])
                for item in example.relation["subevent_relations"]:
                    if example.label == item[0] and item[0] in label_map and item[1] in label_map:
                        relation[relation_map['SUBEVENT']].append([label_map[item[0]], label_map[item[1]]])
                    elif example.label == item[1] and item[0] in label_map and item[1] in label_map:
                        relation[relation_map['SUPEREVENT']].append([label_map[item[0]], label_map[item[1]]])
            else:
                continue
        else:
            if example.label in label_map:
                label = label_map[example.label]
            else:
                continue

        rel_example_ids[example_id] = relation

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("example_id: {}".format(example.example_id))
            logger.info("input_ids: {}".format(" ".join(map(str, input_ids))))
            logger.info("attention_mask: {}".format(" ".join(map(str, attention_mask))))
            logger.info("token_type_ids: {}".format(" ".join(map(str, token_type_ids))))
            logger.info("label: {}".format(label))
            logger.info(rel_example_ids[example_id])
            logger.info(rel_event_ids[label])

        features.append(InputFeatures(example_id=example_id, input_ids=input_ids, input_mask=attention_mask,
                                      segment_ids=token_type_ids, label=label))

    filename = "TestData/rel_example_ids"
    with open(filename, 'w') as file_obj:
        json.dump(rel_example_ids, file_obj)

    filename = "TestData/rel_event_ids"
    with open(filename, 'w') as file_obj:
        json.dump(rel_event_ids, file_obj)

    return features


def pythonify(json_data):

    correctedDict = {}

    for key, value in json_data.items():
        if isinstance(value, list):
            value = [pythonify(item) if isinstance(item, dict) else item for item in value]
        elif isinstance(value, dict):
            value = pythonify(value)
        try:
            key = int(key)
        except Exception as ex:
            pass
        correctedDict[key] = value

    return correctedDict

def read_rel():
    global rel_example_ids
    global rel_event_ids
    rel_example_ids=pythonify(json2dicts("TestData/rel_example_ids")[0])
    rel_event_ids=pythonify(json2dicts("TestData/rel_event_ids")[0])

def get_event_rel():
    global rel_event_ids
    return rel_event_ids

def get_example_rel():
    global rel_example_ids
    return rel_example_ids


rel_example_ids = {}
rel_event_ids = {}

processors = {"newmodel": NewModelProcessor}  # other dataset can also be used here

LABEL_PATH = "ModelData/event_dict_test_data.json"
# # file path for the json data contains all labels, such as './event_dict_train_data.json'
RELATION_PATH = "ModelData/relation.json"
# # file path for the json data contains all relations, such as './event_relation.json'
