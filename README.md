# UNLP-2025-Shared-Task-Span-Identification
## 2nd Place Solution for the UNLP 2025 Shared Task: Span Identification

### Overview of the Approach

Our solution treats the span identification problem as a token classification task. We fine-tuned a large pre-trained transformer model to predict labels for each token, which were then converted back into character-level spans.

1. **Model:** We used FacebookAI/xlm-roberta-large, a powerful multilingual transformer model, adapting it for token classification.
    
2. **Data Preprocessing & Feature Engineering:** Character-level spans were mapped to token-level BIO labels (Beginning, Inside, Outside) using the tokenizer's offset_mapping.
    
3. **Validation Strategy:** A 15% validation split was used with early stopping based on the span-level F1 score.
    

### Details of the Submission

Several key components and decisions contributed to the final performance:

1. **Model Choice (XLM-RoBERTa-Large):** Given the task involved Ukrainian text, a strong multilingual model was essential. XLM-RoBERTa-Large provides robust representations learned from a massive multilingual corpus. Its large size generally offers higher performance, which we prioritized. We loaded it using Hugging Face's AutoModelForTokenClassification.
    
    ```
    from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoConfig
    
    MODEL_NAME = "FacebookAI/xlm-roberta-large"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load model config specifying 3 labels (O, B, I)
    config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=3)
    
    # Load model for token classification
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, config=config)
    model.to(device)
    ```
    
2. **Token Classification Formulation & Alignment:** Framing the problem as token classification allowed leveraging powerful pre-trained models. The custom align_tokens_and_spans logic was crucial for correctly translating character-level ground truth to token-level BIO labels (0=Outside, 1=Beginning, 2=Inside), using the tokenizer's offset_mapping and handling subword tokenization carefully. Special tokens were assigned -100 to be ignored. (The implementation detail of align_tokens_and_spans is provided in the full code).
    
3. **Weighted Focal Loss:** To handle the inherent class imbalance (many 'O' tokens vs. fewer 'B'/'I' tokens) and focus training on harder examples, we implemented and used a WeightedFocalLoss. The alpha parameter directly weighted the classes, and the gamma parameter modulated the loss based on prediction confidence.
    
    ```
    import torch
    from torch import nn
    
    class WeightedFocalLoss(nn.Module):
        """
        Weighted Focal Loss implementation.
        alpha: Weights for each class (e.g., [O_weight, B_weight, I_weight])
        gamma: Focusing parameter (>= 0). Higher gamma focuses more on hard examples.
        """
        def __init__(self, alpha=[0.1, 0.45, 0.45], gamma=2.0, ignore_index=-100):
            super(WeightedFocalLoss, self).__init__()
            self.alpha = torch.tensor(alpha).float()
            self.gamma = gamma
            self.ignore_index = ignore_index
            self.log_softmax = nn.LogSoftmax(dim=-1)
    
        def forward(self, inputs, targets):
            mask = targets != self.ignore_index
            valid_inputs = inputs[mask]
            valid_targets = targets[mask]
    
            if valid_targets.numel() == 0:
                 return torch.tensor(0.0, device=inputs.device, requires_grad=True)
    
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
    
            log_probs = self.log_softmax(valid_inputs)
            gathered_log_probs = log_probs.gather(1, valid_targets.unsqueeze(1)).squeeze(1)
            probs = torch.exp(gathered_log_probs)
            alpha_t = self.alpha[valid_targets]
            focal_loss = alpha_t * torch.pow(1 - probs, self.gamma) * (-gathered_log_probs)
    
            return focal_loss.mean()
    
    # Instantiate the loss function for training
    criterion = WeightedFocalLoss(alpha=[0.1, 0.45, 0.45], gamma=2.0, ignore_index=-100).to(device)
    ```

    
4. **Layer-wise Learning Rate Decay (LLRD):** We applied LLRD (LLRD_RATE = 0.9) to assign smaller learning rates to the lower, more general layers of XLM-R and higher rates to the top layers and classification head. This encourages more stable fine-tuning. The get_optimizer_grouped_parameters function handled the parameter grouping.
    
    ```
    def get_optimizer_grouped_parameters(
        model, learning_rate, weight_decay, layerwise_lr_decay_rate
    ):
        no_decay = ["bias", "LayerNorm.weight"]
        model_prefix = model.base_model_prefix
        layers = getattr(model, model_prefix).encoder.layer
        num_layers = len(layers)
        optimizer_grouped_parameters = [ # Classifier/Pooler layers get base LR
            {"params": [p for n, p in model.named_parameters() if "classifier" in n or "pooler" in n],
             "weight_decay": 0.0, "lr": learning_rate},
        ]
        # Assign layer-wise decayed learning rates
        for i, layer in enumerate(layers):
            lr_scale = layerwise_lr_decay_rate ** (num_layers - 1 - i)
            layer_lr = learning_rate * lr_scale
            # Add parameters with and without weight decay for the current layer
            optimizer_grouped_parameters += [
                {"params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                 "weight_decay": weight_decay, "lr": layer_lr},
                {"params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                 "weight_decay": 0.0, "lr": layer_lr},
            ]
        # Assign decayed learning rate for embeddings
        embeddings = getattr(model, model_prefix).embeddings
        embeddings_lr = learning_rate * (layerwise_lr_decay_rate ** num_layers)
        optimizer_grouped_parameters += [
            {"params": [p for n, p in embeddings.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": weight_decay, "lr": embeddings_lr},
            {"params": [p for n, p in embeddings.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, "lr": embeddings_lr},
        ]
        return optimizer_grouped_parameters
    
    # Setup the optimizer with grouped parameters
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    LLRD_RATE = 0.9
    optimizer_parameters = get_optimizer_grouped_parameters(
        model, LEARNING_RATE, WEIGHT_DECAY, LLRD_RATE
    )
    optimizer = torch.optim.AdamW(optimizer_parameters, lr=LEARNING_RATE, eps=1e-8)
    ```
    
5. **Training Optimizations:**
    
    - **AdamW Optimizer:** Used for its effectiveness with transformers (eps=1e-8).
        
    - **Linear Warmup Scheduler:** Stabilized early training by gradually increasing the LR over 10% of steps before linear decay.
        
        ```
        from transformers import get_linear_schedule_with_warmup
        
        EPOCHS = 8
        ACCUMULATION_STEPS = 4
        num_update_steps_per_epoch = (len(train_dataloader) + ACCUMULATION_STEPS - 1) // ACCUMULATION_STEPS
        total_steps = num_update_steps_per_epoch * EPOCHS
        num_warmup_steps = int(0.1 * total_steps)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps
        )
        ```
        
    - **Gradient Accumulation (ACCUMULATION_STEPS = 4):** Allowed an effective batch size of 8 while using a per-step batch size of 2, improving stability. The loss was scaled down before backward().
        
        ```
        # Inside training loop for each step:
        # ... forward pass ...
        loss = criterion(logits.view(-1, model.config.num_labels), labels.view(-1))
        loss = loss / ACCUMULATION_STEPS # Scale loss
        
        loss.backward() # Accumulate gradients
        
        if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == len(train_dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clip gradients
            optimizer.step() # Update weights
            scheduler.step() # Update learning rate
            optimizer.zero_grad() # Reset gradients
        ```
 
        
6. **Post-processing: Span Conversion and Merging:** After inference, token predictions (0, 1, 2) were converted back to character spans using offset_mapping. Crucially, spans separated by a small gap (SPAN_MERGE_DISTANCE = 1) were merged. This helps connect parts of a manipulative phrase that might be broken by punctuation or tokenization artifacts.
    
    ```
    def tokens_to_char_spans(tokenizer, text, token_preds, offset_mapping, merge_distance=1):
        char_preds = []
        current_span = None
        # Iterate through tokens and their predictions (0=O, 1=B, 2=I)
        for i, (start, end) in enumerate(offset_mapping):
            if start == end == 0 or i >= len(token_preds): continue # Skip special/padding tokens
    
            pred = token_preds[i]
            if pred == 1: # Begin span
                if current_span: char_preds.append(tuple(current_span))
                current_span = [start, end]
            elif pred == 2: # Inside span
                if current_span: current_span[1] = max(current_span[1], end) # Extend span
                else: current_span = [start, end] # Start span if 'I' is first tag encountered
            else: # Outside span
                if current_span: char_preds.append(tuple(current_span))
                current_span = None
        if current_span: char_preds.append(tuple(current_span)) # Add last span
    
        # Filter empty spans and sort
        char_preds = [span for span in char_preds if span[0] < span[1]]
        if not char_preds: return []
        char_preds.sort(key=lambda x: x[0])
    
        # Merge nearby spans
        if len(char_preds) > 1:
            merged_spans = [char_preds[0]]
            for span in char_preds[1:]:
                prev_span = merged_spans[-1]
                # Check distance: start of current - end of previous
                if span[0] - prev_span[1] <= merge_distance:
                    # Merge: update end of the previous span
                    merged_spans[-1] = (prev_span[0], max(prev_span[1], span[1]))
                else:
                    # No merge: add the current span as a new one
                    merged_spans.append(span)
            char_preds = merged_spans
    
        return char_preds
    ```
    
### Sources

- **Hugging Face Transformers Library:** Heavily utilized for models, tokenizers, and training utilities. [https://huggingface.co/docs/transformers/index](https://www.google.com/url?sa=E&q=https%3A%2F%2Fhuggingface.co%2Fdocs%2Ftransformers%2Findex)
    
- **PyTorch:** The deep learning framework used for model definition and training. [https://pytorch.org/](https://www.google.com/url?sa=E&q=https%3A%2F%2Fpytorch.org%2F)
    
- **XLM-RoBERTa Paper:** Conneau, Alexis, et al. "Unsupervised cross-lingual representation learning at scale." arXiv preprint arXiv:1911.02116 (2019). [https://arxiv.org/abs/1911.02116](https://www.google.com/url?sa=E&q=https%3A%2F%2Farxiv.org%2Fabs%2F1911.02116)
    
- **Focal Loss Paper (Inspiration for WeightedFocalLoss):** Lin, Tsung-Yi, et al. "Focal loss for dense object detection." Proceedings of the IEEE international conference on computer vision. 2017. [https://arxiv.org/abs/1708.02002](https://www.google.com/url?sa=E&q=https%3A%2F%2Farxiv.org%2Fabs%2F1708.02002)
    
- **AdamW Optimizer Paper:** Loshchilov, Ilya, and Frank Hutter. "Decoupled weight decay regularization." arXiv preprint arXiv:1711.05101 (2017). [https://arxiv.org/abs/1711.05101](https://www.google.com/url?sa=E&q=https%3A%2F%2Farxiv.org%2Fabs%2F1711.05101)
    
- **LLRD Concept (Often discussed in forums/blogs, e.g., related to ULMFiT):** Howard, Jeremy, and Sebastian Ruder. "Universal language model fine-tuning for text classification." arXiv preprint arXiv:1801.06146 (2018). [https://arxiv.org/abs/1801.06146](https://www.google.com/url?sa=E&q=https%3A%2F%2Farxiv.org%2Fabs%2F1801.06146) (While ULMFiT introduced gradual unfreezing, the concept of differential learning rates is related).
