import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the tokenizer and the model
tokenizer = T5Tokenizer.from_pretrained(os.getcwd() + '/trained/t5')
model = T5ForConditionalGeneration.from_pretrained(os.getcwd() + '/trained/t5')

# Ensure the model is on the right device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_question(text):
    # Prepare the input text with a prompt for question generation
    input_text = f"generate question about: {text}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    output_ids = model.generate(input_ids, max_length=64, num_beams=4, early_stopping=True)

    # Decode the generated question
    question = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return question

# Example text
context = 'At Growth Engineering Jusuf and Edi work together on a react component library.'

# Generate a question
generated_question = generate_question(context)

# Print the result
print(f"Context: {context}")
print(f"Generated Question: {generated_question}")