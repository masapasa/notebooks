from transformers import AutoTokenizer
model_checkpoint = "microsoft/deberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
def prepare_train_features