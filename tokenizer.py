from transformers import AutoTokenizer

class Tokenizer:
    def __init__(self, model_name="t5-small"):
        self.auto_tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize(self, text):
        """Stub: Tokenizes input text"""

        return self.auto_tokenizer.tokenize(text)
    
    def add_tokens(self, new_tokens):
        self.auto_tokenizer.add_tokens(new_tokens)

    def encode(self, text):
        """Stub: Encodes input text to token IDs."""

        return self.auto_tokenizer.encode(text, add_special_tokens=True)
    
    def decode(self, token_ids):
        """Stub: Decodes token IDs back to text."""

        return self.auto_tokenizer.decode(token_ids)
