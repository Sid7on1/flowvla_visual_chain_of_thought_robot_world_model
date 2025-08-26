import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from einops import rearrange
from config import Config
from utils import get_logger
from exceptions import ModelException

logger = get_logger(__name__)

class TokenEmbeddings(nn.Module):
    """Token Embeddings module."""
    
    def __init__(self, config: Config):
        super(TokenEmbeddings, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids):
        """Forward pass."""
        
        # Token embeddings
        token_embeddings = self.token_embedding(input_ids)
        
        # Position embeddings
        position_embeddings = self.position_embedding(torch.arange(input_ids.shape[1], device=input_ids.device))
        
        # Add token and position embeddings
        embeddings = token_embeddings + position_embeddings
        
        # Apply dropout
        embeddings = self.dropout(embeddings)
        
        return embeddings

class FlowVLAModel(nn.Module):
    """FlowVLA Model."""
    
    def __init__(self, config: Config):
        super(FlowVLAModel, self).__init__()
        self.config = config
        self.token_embeddings = TokenEmbeddings(config)
        self.unified_transformer = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=config.num_labels)
        self.action_head = nn.Linear(config.hidden_size, config.num_labels)
        self.flow_head = nn.Linear(config.hidden_size, config.num_labels)
        self.frame_head = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids, attention_mask, flow_input_ids, flow_attention_mask):
        """Forward pass."""
        
        # Token embeddings
        token_embeddings = self.token_embeddings(input_ids)
        
        # Unified transformer
        unified_transformer_output = self.unified_transformer(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=token_embeddings)
        
        # Action head
        action_output = self.action_head(unified_transformer_output.last_hidden_state[:, 0, :])
        
        # Flow head
        flow_output = self.flow_head(self.generate_flow_tokens(flow_input_ids, flow_attention_mask))
        
        # Frame head
        frame_output = self.frame_head(self.generate_frame_tokens(unified_transformer_output.last_hidden_state))
        
        return action_output, flow_output, frame_output

    def get_input_embeddings(self):
        """Get input embeddings."""
        
        return self.token_embeddings.token_embedding.weight

    def generate_action_tokens(self, input_ids, attention_mask):
        """Generate action tokens."""
        
        # Token embeddings
        token_embeddings = self.token_embeddings(input_ids)
        
        # Unified transformer
        unified_transformer_output = self.unified_transformer(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=token_embeddings)
        
        # Action head
        action_output = self.action_head(unified_transformer_output.last_hidden_state[:, 0, :])
        
        return action_output

    def generate_flow_tokens(self, flow_input_ids, flow_attention_mask):
        """Generate flow tokens."""
        
        # Token embeddings
        token_embeddings = self.token_embeddings(flow_input_ids)
        
        # Unified transformer
        unified_transformer_output = self.unified_transformer(input_ids=flow_input_ids, attention_mask=flow_attention_mask, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=token_embeddings)
        
        # Flow head
        flow_output = self.flow_head(unified_transformer_output.last_hidden_state[:, 0, :])
        
        return flow_output

    def generate_frame_tokens(self, unified_transformer_output):
        """Generate frame tokens."""
        
        # Frame head
        frame_output = self.frame_head(unified_transformer_output[:, 0, :])
        
        return frame_output

def get_model(config: Config):
    """Get model."""
    
    try:
        model = FlowVLAModel(config)
        logger.info("Model created successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to create model: {str(e)}")
        raise ModelException("Failed to create model.")

if __name__ == "__main__":
    config = Config()
    model = get_model(config)
    print(model)