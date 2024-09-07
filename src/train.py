import os
import torch
from dataset import LorasTask568DataSet
from transformers import Trainer, T5Tokenizer, TrainingArguments, T5ForConditionalGeneration, DataCollatorForSeq2Seq

print("Using Cuda" if torch.cuda.is_available() else  "Using CPU")

# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained(os.getcwd() + '/trained/t5')

# Load T5 model
model = T5ForConditionalGeneration.from_pretrained(os.getcwd() + '/trained/t5')

# Define the device
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Load dataset class
train_dataset = LorasTask568DataSet('train', tokenizer)
eval_dataset = LorasTask568DataSet('valid', tokenizer)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Prepare arguments
training_args = TrainingArguments(
    output_dir='./.out',
    eval_strategy='epoch',                           # Set strategy
    per_device_train_batch_size=8,                  # Test with different batch sizes, best 16
    per_device_eval_batch_size=8,
    num_train_epochs=3,                             # Change to bigger number for final train
    weight_decay=0.01,
    save_steps=10_000,
    warmup_steps=500,
    save_total_limit=2,
    logging_dir='./logs'
)

# Load trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,                           # Add evaluation data, use different set for final training
    data_collator=data_collator
)

# Train the model
trainer.train()
# Evaluate the model
trainer.evaluate()
# Save in folder
trainer.save_model("./trained/t5")
# Save pretrained
tokenizer.save_pretrained("./trained/t5")