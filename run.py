# -*- coding: utf-8 -*-

#!pip install --quiet transformers
#!pip install --quiet datasets
#!pip install --quiet opacus==0.15.0

"""## Libaries"""

import time
import datetime
import scipy
import torch
import pickle
import numpy as np
import pandas as pd
from sklearn import metrics
import collections

import warnings
warnings.filterwarnings("ignore")

"""## Paramater"""

EPOCHS = 3
TRAIN_BATCH_SIZE = 24
VALID_BATCH_SIZE = 24
LEARNING_RATE = 5e-5
#BATCH_LIMIT = 12
INTERVAL = 347#1000#3333#208

TASK = 'imdb'
MODUS = 'contextual'
EPSILON = 25
TARGETS = 2

device = ('cuda' if torch.cuda.is_available() else 'cpu')

"""## Model"""

from transformers import BertTokenizer
from transformers import BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=TARGETS).to(device)

embedding = model.bert.embeddings.word_embeddings.weight.cpu().detach().numpy()

for p in model.bert.embeddings.parameters(): p.requires_grad = False
for p in model.bert.encoder.layer[0:-1].parameters(): p.requires_grad = True

print("Model Parameters (requires_grad: False): {}".format(
    sum(p.numel() for p in model.parameters())))
        
print("Model Parameters (requires_grad: True): {}".format(
    sum(p.numel() for p in model.parameters()
        if p.requires_grad))) # ~ 7M for BERT

"""### Data"""

import datasets
datasets.disable_caching() 

from datasets import load_from_disk

if EPSILON == None:
    valid_data = load_from_disk(f"valid_{TASK}_preprocessed")
else: train_data = load_from_disk(f"train_{TASK}_{MODUS}_{EPSILON}")
valid_data = load_from_disk(f"valid_{TASK}_preprocessed")

train_data = train_data.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=512), batched=True)
train_data = train_data.rename_column("label", "labels")
train_data.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
train_iterator = torch.utils.data.DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

valid_data = valid_data.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=512), batched=True)
valid_data = valid_data.rename_column("label", "labels")
valid_data.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
valid_iterator = torch.utils.data.DataLoader(valid_data, batch_size=VALID_BATCH_SIZE, shuffle=False)

"""### Training"""

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

STEPS = EPOCHS * len(train_data) / TRAIN_BATCH_SIZE
WARMUP = 0.1
CYCLES = 3

scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer, STEPS * WARMUP, STEPS, CYCLES
)

checkpoint = 0.0
history = collections.defaultdict(list)

t0 = time.time()

for epoch in range(EPOCHS):
    
  for index, batch in enumerate(train_iterator):
      
        if not isinstance(batch, list):
  
            model.train()
        
            batch = {k:v.to(device) for k,v in batch.items()}
        
            output = model(**batch)
            
            output.loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        if index % INTERVAL == 0:
            
          history['time'].append(
              time.time() - t0
          )
    
          history['train_loss'].append(
              output.loss.cpu().item()
          )
    
          logits = output.logits.detach().cpu()
          
          history['train_accuracy'].append(
             metrics.accuracy_score(
                batch['labels'].cpu(),
                torch.argmax(logits, dim=1),
            )
          )
          
          del output
          
          batch_loss = []
          batch_pred = []
          batch_true = []
    
          with torch.no_grad():
              
            model.eval() 
    
            for batch in valid_iterator:
                
                if not isinstance(batch, list):
    
                  batch = {k:v.to(device) for k,v in batch.items()}
        
                  output = model(**batch)
        
                  batch_loss.append(
                      output.loss.cpu().item()
                  )
            
                  batch_true.append(
                      batch['labels'].cpu()
                  )
        
                  batch_pred.append(
                      output.logits.detach().cpu()
                  )
              
            history['valid_loss'].append(
                np.mean(batch_loss)
            )
    
            history['valid_accuracy'].append(
                metrics.accuracy_score(
                    torch.cat(batch_true),
                    torch.argmax(torch.cat(batch_pred), dim=1),
                )
            )
            
            del batch_loss
            del batch_pred
            del batch_true
        
            #scheduler.step(np.mean(batch_loss))
    
            print("Epoch: {} -".format(epoch),
                  "Batch: {} |".format(index),
                  "Train Loss: {:.3f} |".format(history['train_loss'][-1]),
                  "Train Acc.: {:.3f} |".format(history['train_accuracy'][-1]),
                  "Valid Loss: {:.3f} |".format(history['valid_loss'][-1]),
                  "Valid Acc.: {:.3f} |".format(history['valid_accuracy'][-1]),
                  "Time: {} |".format(str(datetime.timedelta(seconds=history['time'][-1])).split(".")[0]),
                  (f"(ε = {EPSILON})" if EPSILON is not float('inf') else "(ε = ∞)")
            )
            
            if history['valid_accuracy'][-1] > checkpoint:
              
              torch.save(
                  model.state_dict(),
                  f'checkpoint_bert_{TASK}_{MODUS}_{EPSILON}.pt'
              )
                            
              checkpoint = history['valid_accuracy'][-1]
              
            with open(f'history_bert_{TASK}_{MODUS}_{EPSILON}.pk', 'wb') as f: pickle.dump(history, f)
            
print(max(history['valid_accuracy']))
                    
        
