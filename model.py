import numpy as np
from datasets import load_dataset
import torch

from transformers import TrainingArguments, EvalPrediction, XLMRobertaTokenizer, XLMRobertaConfig, TextClassificationPipeline, AutoModelForSequenceClassification

from adapters import AutoAdapterModel, AdapterTrainer, CompacterConfig

print('GPU: ', torch.cuda.is_available())

tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

def encode_batch(batch):
  """Encodes a batch of input data using the model tokenizer."""
  return tokenizer(batch["casename"],batch["facts"], max_length=256, truncation=True, padding="max_length")

def encode_batch_label(batch):
  return {'label':[0 if i == 'criminal' else 1 for i in batch["casetype"]]}


dataset = load_dataset("lbox/lbox_open", "casename_classification")

dataset = dataset.map(encode_batch, batched=True).map(encode_batch_label, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

config = XLMRobertaConfig.from_pretrained(
    "xlm-roberta-base",
    num_labels=2,
    id2label={ 0: "civil", 1: "criminal"}
)
model = AutoAdapterModel.from_pretrained(
    "xlm-roberta-base",
    config=config,
).to('cuda:0')

# model = AutoModelForSequenceClassification.from_pretrained(
#     "xlm-roberta-base",
#     config=config,
# ).to('cuda:0')

# adapters.init(model)

# Add a new adapter
model.add_adapter("dummy", config=CompacterConfig())
# Add a matching classification head
model.add_classification_head(
    "dummy",
    num_labels=2,
    id2label={ 0: "civil", 1: "criminal"}
  )
# Activate the adapter
model.train_adapter("dummy")

training_args = TrainingArguments(
    learning_rate=1e-4,
    num_train_epochs=6,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    logging_steps=200,
    output_dir="./training_output",
    overwrite_output_dir=True,
    # The next line is important to ensure the dataset labels are properly passed to the model
    remove_unused_columns=False,
)

def compute_accuracy(p: EvalPrediction):
  preds = np.argmax(p.predictions, axis=1)
  return {"acc": (preds == p.label_ids).mean()}

trainer = AdapterTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_accuracy,
)


trainer.train()
trainer.evaluate()


classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=training_args.device.index)

print(classifier("피고인은 2020. 1. 22. 02:00경 서울 서초구 B 소재 **** 주점 2번 테이블에서 술을 마시다가 화장실을 다녀오던 중, 갑자기 테이블 바깥쪽에 앉아있던 피해자 C(여, 39세)의 왼쪽 가슴을 3회 만져 피해자를 강제로 추행하였다."))

model.save_adapter("./final_adapter", "dummy")