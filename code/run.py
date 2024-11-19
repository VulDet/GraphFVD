# coding=utf-8
from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import scipy.sparse as sp
import numpy as np
import torch
import csv
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score
import json
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
import multiprocessing
from model import *
cpu_cont = multiprocessing.cpu_count()
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, RobertaModel,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)

from typing import Optional
import string

logger = logging.getLogger(__name__)
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

class Lexer(object):
    def __init__(self, code: str):
        self.code = code

    def _is_identifier_char(self, c: str) -> bool:
        if c in string.ascii_letters + string.digits + "_":
            return True
        return False

    def _is_digit(self, c: str) -> bool:
        if c in string.digits:
            return True
        return False

    def __iter__(self):
        self.i = 0
        return self

    def _peek(self) -> Optional[str]:
        if self.i < len(self.code):
            return self.code[self.i]
        raise StopIteration

    def _get(self) -> Optional[str]:
        if self.i < len(self.code):
            val = self.code[self.i]
            self.i += 1
            return val
        raise StopIteration

    def _identifier(self) -> str:
        token = self._get()
        while self._is_identifier_char(self._peek()):
            token += self._get()
        return token

    def _number(self) -> str:
        token = self._get()
        while self._is_digit(self._peek()):
            token += self._get()
        return token

    def __next__(self) -> str:
        while True:
            c = self._peek()
            if c is None:
                break
            if not c.isspace():
                break
            self._get()

        c = self._peek()
        if c in string.ascii_letters:
            return self._identifier()
        elif c in string.digits:
            return self._number()
        elif c in "()[]{}<>=+-*/#.,:;'\"|":
            return self._get()
        else:
            return self._get()

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 all_ids,
                 edges,
                 edges_label,
                 num_nodes,
                 idx,
                 label,

    ):
        self.all_ids = all_ids,
        self.edges = edges
        self.edges_label = edges_label
        self.num_nodes = num_nodes
        self.idx = str(idx)
        self.label = label


def convert_graphs_to_features(js,tokenizer, args):
    codes = js['nodes_codes']
    edges = js['edges']
    edges_label = []
    edges_label_source = js['edges_label']
    for i, label in enumerate(edges_label_source):
        if len(label) != 0:
            edges_label.append(0)
        elif len(label) == 0:
            node0_code = codes[edges[i][0]]
            node1_code = codes[edges[i][1]]
            if (node0_code not in node1_code) and (node1_code not in node0_code):
                edges_label.append(1)
            else:
                edges_label.append(2)

    codes = codes[:args.block_size]
    all_ids = []
    codes = codes[:args.block_size]
    for code in codes:
        code_tokens = tokenizer.tokenize(code)
        source_tokens = [tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        all_ids.append(source_ids)
    num_nodes = len(codes)

    if num_nodes == 0:
        return None

    return InputFeatures(all_ids, edges, edges_label, num_nodes, js['idx'], js['target'])

def convert_codes_to_tokens(js, args):
    codes = js['nodes_codes']
    codes = codes[:args.block_size]
    all_tokens = []
    codes = codes[:args.block_size]
    for code in codes:
        code_tokens = code.split(' ')
        all_tokens.append(code_tokens)
    return all_tokens

# build graph function
def build_graph(node_idxs, nodes_edges, num_nodes, edges_label, w_embeddings, args):
    # print('using window size = ', window_size)
    node_idxs = node_idxs[0]
    edges = nodes_edges + [[y, x] for x, y in nodes_edges]
    edges_label = edges_label + edges_label
    adj_list = []
    for i in range(3):
        row, col, weight = [], [], []
        for j, edge in enumerate(edges):
            if edge[0] == edge[1]:
                continue
            if edge[0] < num_nodes and edge[1] < num_nodes and edges_label[j] == i:
                row.append(edge[0])
                col.append(edge[1])
                weight.append(1)
                # weight.append(edges_label[i])
        adj = sp.csr_matrix((weight, (row, col)), shape=(num_nodes, num_nodes))
        adj_list.append(adj)


    features_out = []
    for node_idx in node_idxs:
        emb_seq = []
        for token in node_idx:
            emb_seq.append(w_embeddings[token])
        emb_seq = torch.tensor(emb_seq)
        features_out.append(emb_seq)

    return adj_list, features_out

def build_adj(nodes_edges, num_nodes):
    # print('using window size = ', window_size)
    edges = nodes_edges + [[y, x] for x, y in nodes_edges]
    row, col, weight = [], [], []
    for edge in edges:
        if edge[0] == edge[1]:
            continue
        if edge[0] < num_nodes and edge[1] < num_nodes:
            row.append(edge[0])
            col.append(edge[1])
            weight.append(1)
    adj = sp.csr_matrix((weight, (row, col)), shape=(num_nodes, num_nodes))

    return adj

def convert_examples_to_features(js,tokenizer,args):
    code=' '.join(js['func'].split())

    code_tokens=tokenizer.tokenize(code)[:args.block_size-2]
    source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids+=[tokenizer.pad_token_id]*padding_length
    return InputFeatures(source_tokens,source_ids,js['idx'],js['target'])


class TextDataset(Dataset):
    def __init__(self, encoder, tokenizer, args, file_path=None, sample_percent=1.):
        self.examples = []
        self.encoder = encoder.to(args.device)
        self.args = args
        self.w_embeddings = encoder.embeddings.word_embeddings.weight.data.cpu().detach().clone().numpy()

        with open(file_path) as f:
            for line in f:
                js = json.loads(line.strip())
                result = convert_graphs_to_features(js, tokenizer, args)
                if result is None:
                    continue
                self.examples.append(result)
        total_len = len(self.examples)
        num_keep = int(sample_percent * total_len)

        if num_keep < total_len:
            np.random.seed(10)
            np.random.shuffle(self.examples)
            self.examples = self.examples[:num_keep]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        adj, x_feature = build_graph(self.examples[i].all_ids, self.examples[i].edges, self.examples[i].num_nodes, self.examples[i].edges_label, self.w_embeddings, self.args)
        adj, adj_mask = preprocess_adj(adj, self.args.block_size)
        adj_feature = preprocess_features(x_feature, self.args.block_size)

        label_ = self.examples[i].label

        return torch.tensor(adj), torch.tensor(adj_mask), torch.tensor(adj_feature), torch.tensor(label_)


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, train_dataset, eval_dataset, model, tokenizer):
    """ Train the model """ 
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
                                  batch_size=args.train_batch_size, num_workers=4, pin_memory=True)  # numworkers=4
    args.max_steps=args.epoch*len(train_dataloader)
    args.save_steps=len(train_dataloader)
    args.warmup_steps=len(train_dataloader)
    args.logging_steps=len(train_dataloader)
    args.num_train_epochs=args.epoch
    model.to(args.device)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps*0.1,
                                                num_training_steps=args.max_steps)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)
    
    global_step = args.start_step
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_mrr = 0.0
    best_f1 = 0.0
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()

    my_metrics = []
    for idx in range(args.start_epoch, int(args.num_train_epochs)):
        tr_num = 0
        train_loss = 0

        # for step, batch in enumerate(bar):
        for step, batch in enumerate(train_dataloader):
            adj = batch[0].to(args.device)
            adj_mask = batch[1].to(args.device)
            adj_feature = batch[2].to(args.device)
            labels = batch[3].to(args.device)
            model.train()
            loss, logits = model(adj, adj_mask, adj_feature, labels)


            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num+=1
            train_loss+=loss.item()
            if avg_loss==0:
                avg_loss=tr_loss
            avg_loss=round(train_loss/tr_num,5)
                
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                output_flag=True
                avg_loss=round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss
                    tr_nb=global_step

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, eval_dataset, model, tokenizer,eval_when_training=True)
                        my_metric = []
                        for key, value in results.items():
                            logger.info("  %s = %s", key, round(value,4))
                            my_metric.append(round(value,4))
                        my_metrics.append(my_metric)
                        # Save model checkpoint
                        
                    if results['eval_f1']>best_f1:
                        best_f1=results['eval_f1']
                        logger.info("  "+"*"*20)  
                        logger.info("  Best f1:%s",round(best_f1,4))
                        logger.info("  "+"*"*20)                          
                        
                        checkpoint_prefix = 'checkpoint-best-f1'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)                        
                        model_to_save = model.module if hasattr(model,'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)
        avg_loss = round(train_loss / tr_num, 5)
        logger.info("epoch {} loss {}".format(idx, avg_loss))


def p_r_f1(labels, preds):
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(len(labels)):
        if labels[i] == 1 and preds[i] == 1:
            TP += 1
        if labels[i] == 0 and preds[i] == 1:
            FP += 1
        if labels[i] == 1 and preds[i] == 0:
            FN += 1
        if labels[i] == 0 and preds[i] == 0:
            TN += 1
    precision = (TP+1) / (TP+FP+1)
    recall = (TP+1) / (TP+FN+1)
    f1 = 2 / (1/precision + 1/recall)
    FPR = (FP + 1) / (TN + FP + 1)
    FNR = (FN + 1) / (TP + FN + 1)
    return precision, recall, f1, FPR, FNR

def evaluate(args, eval_dataset, model, tokenizer, eval_when_training=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4, pin_memory=True)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []
    for batch in eval_dataloader:
        adj = batch[0].to(args.device)
        adj_mask = batch[1].to(args.device)
        adj_feature = batch[2].to(args.device)
        label = batch[3].to(args.device)
        with torch.no_grad():
            lm_loss, logit = model(adj, adj_mask, adj_feature, label)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    logits = np.concatenate(logits,0)
    labels = np.concatenate(labels,0)
    preds = logits[:,0]>0.5

    eval_acc = np.mean(labels==preds)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)

    eval_precision, eval_recall, eval_f1, eval_FPR, eval_FNR = p_r_f1(labels, preds)
    eval_auc = roc_auc_score(labels, preds)
    result = {
        "eval_loss": float(perplexity),
        "eval_acc": round(eval_acc,4),
        "eval_precision": round(eval_precision,4),
        "eval_recall": round(eval_recall,4),
        "eval_f1": round(eval_f1,4),
        "eval_auc": round(eval_auc,4),
        "eval_FPR":round(eval_FPR,4),
        "eval_FNR":round(eval_FNR,4)
    }
    return result

def test(args, eval_dataset, model, tokenizer):
    # Loop to handle MNLI double evaluation (matched, mis-matched)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]   
    labels=[]
    for batch in eval_dataloader:
        adj = batch[0].to(args.device)
        adj_mask = batch[1].to(args.device)
        adj_feature = batch[2].to(args.device)
        label = batch[3].to(args.device)
        with torch.no_grad():
            logit = model(adj, adj_mask, adj_feature)
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())

    logits=np.concatenate(logits,0)
    labels=np.concatenate(labels,0)
    preds=logits[:,0]>0.5

    test_acc=np.mean(labels==preds)
    # 计算精确率和召回率，AUC
    test_precision, test_recall, test_f1, test_FPR, test_FNR = p_r_f1(labels, preds)
    test_auc = roc_auc_score(labels, preds)
    with open(os.path.join(args.output_dir,"predictions.txt"),'w') as f:
        for example,pred in zip(eval_dataset.examples,preds):
            if pred:
                f.write(example.idx+'\t1\n')
            else:
                f.write(example.idx+'\t0\n')

    result = {
        "test_acc": round(test_acc, 4),
        "test_precision": round(test_precision, 4),
        "test_recall": round(test_recall, 4),
        "test_f1": round(test_f1, 4),
        "test_auc": round(test_auc, 4),
        "test_FPR": round(test_FPR, 4),
        "test_FNR": round(test_FNR, 4)
    }
    return result
                        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default="../dataset/my_train.jsonl", type=str,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default="./saved_models", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default="../dataset/my_valid.jsonl", type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default="../dataset/my_test.jsonl", type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
                    
    parser.add_argument("--model_type", default="roberta", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default="microsoft/codebert-base", type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="microsoft/codebert-base", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")    
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")


    parser.add_argument("--model", default="GNNs", type=str,help="")
    parser.add_argument("--hidden_size", default=256, type=int,
                        help="hidden size.")
    parser.add_argument("--feature_dim_size", default=768, type=int,
                        help="feature dim size.")
    parser.add_argument("--num_GNN_layers", default=2, type=int,
                        help="num GNN layers.")
    parser.add_argument("--num_classes", default=2, type=int,
                        help="num classes.")
    parser.add_argument("--gnn", default="ReGCN", type=str, help="ReGCN or ReGGNN")
    parser.add_argument("--remove_residual", default=False, action='store_true', help="remove_residual")
    parser.add_argument("--att_op", default='mul', type=str,
                        help="using attention operation for attention: mul, sum, concat")
    parser.add_argument("--training_percent", default=1., type=float, help="percet of training sample")
    parser.add_argument("--alpha_weight", default=1., type=float, help="percet of training sample")

    input_argument = [
        "--output_dir", "./saved_models",
        "--model_type", "roberta",
        "--tokenizer_name", "microsoft/graphcodebert-base",  
        "--model_name_or_path", "microsoft/graphcodebert-base",
        "--do_eval",
        "--do_test",
        "--do_train",
        "--train_data_file", "../dataset/NVD/my_train.jsonl",
        "--eval_data_file", "../dataset/NVD/my_valid.jsonl",
        "--test_data_file", "../dataset/NVD/my_test.jsonl",
        "--block_size", "400",
        "--train_batch_size", "32",
        "--eval_batch_size", "32",
        "--max_grad_norm", "1.0",
        "--evaluate_during_training",
        "--gnn", "ReGCN",
        "--learning_rate", "5e-4",
        "--epoch", "200",
        "--hidden_size", "128",
        "--num_GNN_layers", "2",
        "--seed", "123456",
    ]

    args = parser.parse_args(args=input_argument)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    args.per_gpu_train_batch_size=args.train_batch_size//max(args.n_gpu,1)
    args.per_gpu_eval_batch_size=args.eval_batch_size//max(args.n_gpu,1)
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)



    # Set seed
    set_seed(args.seed)
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.model_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels=1
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    if args.model_name_or_path:
        pretrained_model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)    
    else:
        pretrained_model = model_class(config)

    model = GNNReGVD(pretrained_model, config, tokenizer, args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    logger.info("Training/evaluation parameters %s", args)

    train_dataset = TextDataset(pretrained_model, tokenizer, args, args.train_data_file, args.training_percent)
    eval_dataset = TextDataset(pretrained_model, tokenizer, args, args.eval_data_file)
    test_dataset = TextDataset(pretrained_model, tokenizer, args, args.test_data_file)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        if args.local_rank == 0:
            torch.distributed.barrier()
        train(args, train_dataset, eval_dataset, model, tokenizer)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
            checkpoint_prefix = 'checkpoint-best-f1/model.bin'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
            model.load_state_dict(torch.load(output_dir))      
            model.to(args.device)
            result=evaluate(args, eval_dataset, model, tokenizer)
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(round(result[key],4)))
            
    if args.do_test and args.local_rank in [-1, 0]:
            checkpoint_prefix = 'checkpoint-best-f1/model.bin'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
            model.load_state_dict(torch.load(output_dir))                  
            model.to(args.device)
            test_result = test(args, test_dataset, model, tokenizer)

            logger.info("***** Test results *****")
            for key in sorted(test_result.keys()):
                logger.info("  %s = %s", key, str(round(test_result[key],4)))

    return results


if __name__ == "__main__":
    main()


