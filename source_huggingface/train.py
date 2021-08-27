import argparse

from dataclasses import dataclass, field
import json
import logging
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoConfig,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    set_seed
)
from transformers.training_args import TrainingArguments

from multimodal_transformers.data import load_data_from_folder
from multimodal_transformers.model import TabularConfig
from multimodal_transformers.model import AutoModelWithTabular

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.getLevelName("INFO"),
    handlers=[logging.StreamHandler(sys.stdout)],
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

os.environ['COMET_MODE'] = 'DISABLED'


if __name__ == '__main__':
        
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    logger.info(sys.argv)
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    parser.add_argument("--output_dir", type=str)
    
    # Data, model, and output directories
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    # args holds all passed-in arguments
    args, _ = parser.parse_known_args()
    
    # load datasets
    train_dataset = pd.read_csv(args.training_dir)
    test_dataset = pd.read_csv(args.test_dir)
    
    logger.info(f' loaded train_dataset length is: {len(train_dataset)}')
    logger.info(f' loaded test_dataset length is: {len(test_dataset)}')
    
    @dataclass
    class ModelArguments:
      """
      Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
      """

      model_name_or_path: str = field(
          metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
      )
      config_name: Optional[str] = field(
          default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
      )
      tokenizer_name: Optional[str] = field(
          default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
      )
      cache_dir: Optional[str] = field(
          default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
      )


    @dataclass
    class MultimodalDataTrainingArguments:
      """
      Arguments pertaining to how we combine tabular features
      Using `HfArgumentParser` we can turn this class
      into argparse arguments to be able to specify them on
      the command line.
      """

      data_path: str = field(metadata={
                                'help': 'the path to the csv file containing the dataset'
                            })
      column_info_path: str = field(
          default=None,
          metadata={
              'help': 'the path to the json file detailing which columns are text, categorical, numerical, and the label'
      })

      column_info: dict = field(
          default=None,
          metadata={
              'help': 'a dict referencing the text, categorical, numerical, and label columns'
                      'its keys are text_cols, num_cols, cat_cols, and label_col'
      })

      categorical_encode_type: str = field(default='ohe',
                                            metadata={
                                                'help': 'sklearn encoder to use for categorical data',
                                                'choices': ['ohe', 'binary', 'label', 'none']
                                            })
      numerical_transformer_method: str = field(default='yeo_johnson',
                                                metadata={
                                                    'help': 'sklearn numerical transformer to preprocess numerical data',
                                                    'choices': ['yeo_johnson', 'box_cox', 'quantile_normal', 'none']
                                                })
      task: str = field(default="classification",
                        metadata={
                            "help": "The downstream training task",
                            "choices": ["classification", "regression"]
                        })

      mlp_division: int = field(default=4,
                                metadata={
                                    'help': 'the ratio of the number of '
                                            'hidden dims in a current layer to the next MLP layer'
                                })
      combine_feat_method: str = field(default='individual_mlps_on_cat_and_numerical_feats_then_concat',
                                        metadata={
                                            'help': 'method to combine categorical and numerical features, '
                                                    'see README for all the method'
                                        })
      mlp_dropout: float = field(default=0.1,
                                  metadata={
                                    'help': 'dropout ratio used for MLP layers'
                                  })
      numerical_bn: bool = field(default=True,
                                  metadata={
                                      'help': 'whether to use batchnorm on numerical features'
                                  })
      use_simple_classifier: str = field(default=True,
                                          metadata={
                                              'help': 'whether to use single layer or MLP as final classifier'
                                          })
      mlp_act: str = field(default='relu',
                            metadata={
                                'help': 'the activation function to use for finetuning layers',
                                'choices': ['relu', 'prelu', 'sigmoid', 'tanh', 'linear']
                            })
      gating_beta: float = field(default=0.2,
                                  metadata={
                                      'help': "the beta hyperparameters used for gating tabular data "
                                              "see https://www.aclweb.org/anthology/2020.acl-main.214.pdf"
                                  })

      def __post_init__(self):
          assert self.column_info != self.column_info_path
          if self.column_info is None and self.column_info_path:
              with open(self.column_info_path, 'r') as f:
                  self.column_info = json.load(f)
    
    
    # data and training parameters
    text_cols = ['raw_Text']
    cat_cols = None
    numerical_cols = ['c_1', 'c_5', 'lcs_word']

    column_info_dict = {
        'text_cols': text_cols,
        'num_cols': numerical_cols,
        'cat_cols': cat_cols,
        'label_col': 'Plagiarism',
        'label_list': ['Not Plagiarism', 'Plagiarism']
    }

    # pre-trained model we want to use for fine-tuning
    model_args = ModelArguments(
        model_name_or_path='distilbert-base-cased'
    )

    # Training Args for our Multimodal model
    data_args = MultimodalDataTrainingArguments(
        data_path='.', # source_dir is multimodal_plagiarism_data
        combine_feat_method='gating_on_cat_and_num_feats_then_sum',
        column_info=column_info_dict,
        task='classification'
    )

    # define training args
    training_args = TrainingArguments(
        output_dir="./logs/model_name",
        logging_dir="./logs/runs",
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=32,
        num_train_epochs=1,
        evaluate_during_training=True,
        logging_steps=25,
        eval_steps=250
    )

    set_seed(training_args.seed)
    
    # setting up our tokenizer
    tokenizer_path_or_name = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    print('Specified tokenizer: ', tokenizer_path_or_name)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path_or_name,
        cache_dir=model_args.cache_dir,
    )
    
    
    # load dataset csvs to torch datasets
    # The function load_data_from_folder expects a path to a folder that contains 
    # train.csv, test.csv, and/or val.csv containing the respective split datasets. 
    train_dataset, test_dataset = load_data_from_folder(
        data_args.data_path,
        data_args.column_info['text_cols'],
        tokenizer,
        label_col=data_args.column_info['label_col'],
        label_list=data_args.column_info['label_list'],
        categorical_cols=data_args.column_info['cat_cols'],
        numerical_cols=data_args.column_info['num_cols'],
        sep_text_token_str=tokenizer.sep_token
    )
    
    # number of labels in dataset
    num_labels = len(np.unique(train_dataset.labels))

    
    # create config for multimodal model
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir
    )
    
    tabular_config = TabularConfig(
        num_labels=num_labels,
        cat_feat_dim=train_dataset.cat_feats.shape[1],
        numerical_feat_dim=train_dataset.numerical_feats.shape[1]
    )
    
    config.tabular_config = tabular_config
    
    # compute metrics function for binary classification 
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}
    
    # download model
    model = AutoModelWithTabular.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    
    # train model (from checkpoint if there is one)
    if get_last_checkpoint(args.output_dir) is not None:
        logger.info('***** continue training *****')
        trainer.train(resume_from_checkpoint=args.output_dir)
    else:
        trainer.train()
    
    # evaluate model
    eval_result = trainer.evaluate(eval_dataset=test_dataset)
    
    # writes eval result to file which can be accessed later in s3 output
    with open(os.path.join(args.output_data_dir, "eval_results.txt"), "w") as writer:
        print('***** Eval results *****')
        for key, value in sorted(eval_result.item()):
            writer.write(f'{key} = {value}\n')
            
    # saves the model to s3
    trainer.save_model(args.model_dir)