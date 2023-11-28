import transformers

tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
model = transformers.BertForSequenceClassification.from_pretrained('bert-base-uncased')
def predict(text):
   input_ids = tokenizer.encode(text, return_tensors='pt')
   attention_mask = tokenizer.create_attention_mask(input_ids)
   output = model(input_ids, attention_mask)
   return output.logits
