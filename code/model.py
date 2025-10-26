import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalClassifier(nn.Module):
    """
    This class implements a multi-instance learning approach for emotion recognition
    using both EEG signals and facial images.
    """
    def __init__(self, input_size = 768, num_classes = 4, 
                 num_heads = 12, dim_feedforward = 2048, num_encoder_layers = 2, device = device, 
                 eeg_size = 384, transformer_dropout_rate = 0.2, cls_dropout_rate = 0.1,
                 fusion_type = 'cross_attention',  # options: 'none', 'cross_attention', 'mlp'
                 instance_selection_method = 'attention_weighted_topk',  # options: 'none', 'softmax', 'amil', 'attention_topk', 'attention_weighted_topk'
                 num_select = 3, num_instances = 10):
        """
        Args:
            input_size (int): Hidden dimension size for transformer layers
            num_classes (int): Number of output classes for classification
            num_heads (int): Number of attention heads in transformer
            dim_feedforward (int): Feedforward dimension in transformer layers
            num_encoder_layers (int): Number of transformer encoder layers
            device: Device to run the model on
            eeg_size (int): Input dimension of EEG data
            transformer_dropout_rate (float): Dropout rate for transformer layers
            cls_dropout_rate (float): Dropout rate for classification head
            fusion_type (str): Type of fusion strategy for image features
            instance_selection_method (str): Method for selecting instances in MIL
            num_select (int): Number of instances to select
            num_instances (int): Total number of instances available
        """
        super().__init__()
        
        # Core hyperparameters and options
        self.transformer_dropout_rate = transformer_dropout_rate
        self.cls_dropout_rate = cls_dropout_rate
        self.fusion_type = fusion_type
        self.instance_selection_method = instance_selection_method
        # Swin image processor and backbone (fine-tuned)
        self.img_processor = swin_processor
        self.swin_model = swin_model
        for param in self.swin_model.parameters():
            # Enable fine-tuning
            param.requires_grad = True

        # Token type embeddings: 0 for image tokens, 1 for EEG tokens
        self.token_type_embeddings = nn.Embedding(2, input_size)
        
        # Transformer encoder over concatenated image and EEG tokens
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model = input_size, 
                nhead = num_heads, 
                dim_feedforward = dim_feedforward, 
                dropout = transformer_dropout_rate, 
                batch_first = True
            ),
            num_layers = num_encoder_layers
        )

        # EEG projection and normalization
        self.eeg_proj = nn.Linear(eeg_size, input_size)
        self.activation = nn.ReLU()
        self.layernorm = nn.LayerNorm(eeg_size)
        
        # Classification head
        self.cls_token = nn.Parameter(torch.zeros(1, 1, input_size)).to(device)
        self.dropout = nn.Dropout(cls_dropout_rate)
        self.classifier = nn.Linear(input_size, num_classes)  # Final classifier

        # Initialize cross-attention components only when fusion_type == 'cross_attention'
        if fusion_type == 'cross_attention':
            self.num_queries = 147
            self.query_tokens = nn.Parameter(torch.zeros(1, self.num_queries, input_size))
            nn.init.normal_(self.query_tokens, std = 0.02)
            self.cross_attention = nn.MultiheadAttention(
                embed_dim = input_size,
                num_heads = num_heads,
                dropout = transformer_dropout_rate,
                batch_first = True
            )
        # MLP up/down projection components
        elif fusion_type == 'mlp':
            self.mlp_up = nn.Linear(input_size, 4 * input_size)  # 768 -> 3072
            self.mlp_act = nn.GELU()
            self.mlp_down = nn.Linear(4 * input_size, input_size)  # 3072 -> 768

        self.num_instances = num_instances
        self.num_select = num_select

        # Global learnable instance weights (used by 'softmax' method)
        self.instance_weights = nn.Parameter(torch.ones(1, num_instances))

        # AMIL attention projection layers
        self.amil_value_proj = nn.Linear(input_size, input_size)
        self.amil_weight_proj = nn.Linear(input_size, 1)

        # Top-K attention projection layers
        self.topk_value_proj = nn.Linear(input_size * 49, input_size * 49)
        self.topk_weight_proj = nn.Linear(input_size * 49, 1)
        self.topk_dimension_recover = nn.Linear(input_size * 49, self.num_instances)

    def select_instances(self, images_embedding):
        """
        Select instances from multiple image embeddings using different MIL methods.
        """
        if self.instance_selection_method == 'none':
            # Return the embedding of the first image
            return images_embedding[:, 0, :, :].unsqueeze(1)
        
        elif self.instance_selection_method == 'softmax':
            # Compute weight for each instance
            weights = F.softmax(self.instance_weights, dim = 1)
            # Select top-k instances
            _, indices = torch.topk(weights, self.num_select, dim = 1)
            selected_embeddings = []
            batch_size = images_embedding.size(0)
            
            for i in range(batch_size):
                # Select the k instances with the highest weights
                selected = images_embedding[i, indices[0], :]
                selected_embeddings.append(selected)
                
            return torch.stack(selected_embeddings)
        
        elif self.instance_selection_method == 'amil':
            batch_size = images_embedding.size(0)
            
            # Compute per-instance representation
            instance_features = images_embedding.mean(dim = 2)
            
            # AMIL attention scores
            hidden = torch.tanh(self.amil_value_proj(instance_features))
            weights = self.amil_weight_proj(hidden)
            weights = F.softmax(weights, dim = 1)
            
            # Weighted sum
            weighted_features = (instance_features * weights).sum(dim = 1)
            
            return weighted_features
        
        elif self.instance_selection_method == 'attention_topk':
            batch_size = images_embedding.size(0)
            
            # Compute per-instance representation
            instance_features = images_embedding.view(batch_size, self.num_instances, -1)
            
            # Attention scores
            hidden = torch.tanh(self.topk_value_proj(instance_features))
            weights = self.topk_weight_proj(hidden)
            weights = F.softmax(weights, dim = 1)

            # Weighted sum
            weighted_features = (instance_features * weights).sum(dim = 1)
            # Recover per-instance weights
            recovered_weights = self.topk_dimension_recover(weighted_features).view(batch_size, self.num_instances, 1)
            
            _, indices = torch.topk(recovered_weights, self.num_select, dim = 1)
            selected_embeddings = []
            
            for i in range(batch_size):
                # Select top-k instances by recovered weights
                selected = images_embedding[i, indices[i], :, :]
                selected_embeddings.append(selected)
            
            return torch.stack(selected_embeddings)
        
        elif self.instance_selection_method == 'attention_weighted_topk':
            batch_size = images_embedding.size(0)
            
            # Compute per-instance representation
            instance_features = images_embedding.view(batch_size, self.num_instances, -1)
            
            # Attention scores
            hidden = torch.tanh(self.topk_value_proj(instance_features))
            weights = self.topk_weight_proj(hidden)
            weights = F.softmax(weights, dim = 1)

            # Select top-k instances
            _, indices = torch.topk(weights, self.num_select, dim = 1)
            selected_embeddings = []
            
            for i in range(batch_size):
                # Select the k instances with the highest weights
                selected = images_embedding[i, indices[i].squeeze(), :, :]
                selected_embeddings.append(selected)
            
            return torch.stack(selected_embeddings)

    def forward(self, eeg_data, images_data):
        batch_size = images_data.size(0)
        
        # Process multiple images
        images_embedding = []
        for i in range(self.num_instances):
            image = images_data[:, i]
            vision_inputs = self.img_processor(image, return_tensors = "pt").to(device)
            embedding = self.swin_model(**vision_inputs).last_hidden_state
            images_embedding.append(embedding)
        
        images_embedding = torch.stack(images_embedding, dim = 1)

        # Select instances
        selected_embeddings = self.select_instances(images_embedding)

        selected_embeddings = selected_embeddings.view(batch_size, -1, 768)
        
        # Process according to fusion_type
        if self.fusion_type == 'cross_attention':
            query_tokens = self.query_tokens.expand(batch_size, -1, -1)
            image_features, _ = self.cross_attention(
                query = query_tokens,
                key = selected_embeddings,
                value = selected_embeddings
            )
            images_embedding = image_features
        
        elif self.fusion_type == 'mlp':
            # MLP up/down projection
            x = self.mlp_up(selected_embeddings)
            x = self.mlp_act(x)
            images_embedding = self.mlp_down(x)
        
        else:
            images_embedding = selected_embeddings

        eeg_data = self.layernorm(eeg_data)
        eeg_embedding = self.eeg_proj(eeg_data)
        eeg_embedding = self.activation(eeg_embedding)

        # Add token-type embeddings to distinguish modalities
        images_embedding, eeg_embedding = (
            images_embedding + self.token_type_embeddings(torch.zeros(images_embedding.shape[0], 1, dtype = torch.long, device = device)),
            eeg_embedding + self.token_type_embeddings(torch.ones(eeg_embedding.shape[0], 1, dtype = torch.long, device = device))
        )

        # Concatenate image and EEG tokens, prepend CLS, and encode
        multi_embedding = torch.cat((images_embedding, eeg_embedding), dim = 1)
        multi_embedding = torch.cat((self.cls_token.expand(multi_embedding.size(0), -1, -1), multi_embedding), dim = 1)
        multi_embedding = self.transformer_encoder(multi_embedding)

        # Take the output of the CLS token
        cls_token_output = multi_embedding[:, 0, :]
        cls_token_output = self.dropout(cls_token_output)
        x = self.classifier(cls_token_output)

        return x