from dataclasses import dataclass
import torch, einops, os
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from ..utils.model_utils import (
    gather_residual_activations, 
)
from ..utils.data_utils import *
from pyvene import (
    IntervenableModel,
)
from transformers import set_seed
import transformers, datasets
from typing import Dict, Optional, Sequence, Union, List, Any
from ..scripts.inference import prepare_df

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)


class BaseModel(object):
    """Base class for all models."""
    def __init__(self, **kwargs):
        pass

    def __str__(self):
        pass

    def make_model(self, **kwargs):
        pass

    def make_dataloader(self, examples, **kwargs):
        pass

    def train(self, examples, **kwargs):
        pass

    def save(self, dump_dir, **kwargs):
        pass

    def load(self, dump_dir, **kwargs):
        pass

    def predict_latent(self, examples, **kwargs):
        pass    

    def predict_steer(self, examples, **kwargs):
        pass

    def get_logits(self, concept_id, k=10):
        pass

    def pre_compute_mean_activations(self, dump_dir, **kwargs):
        pass

    def to(self, device):
        pass


class Model(BaseModel):

    def __init__(self, model, tokenizer, layer, training_args=None, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        # abstracting layer
        self.layer = layer
        self.training_args = training_args
        self.max_activations = {}
        # Set default device to GPU if available, otherwise CPU
        self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.seed = kwargs.get("seed", 42)
        self.steering_layers = kwargs.get("steering_layers", None)
        self.num_of_layers = len(self.steering_layers) if self.steering_layers else 1
        self.dump_dir = kwargs.get("dump_dir", None)

    def make_model(self, **kwargs):
        pass

    def make_dataloader(self, examples, **kwargs):
        data_module = make_data_module(self.tokenizer, examples, **kwargs)
        g = torch.Generator()
        g.manual_seed(self.seed)
        train_dataloader = DataLoader(
            data_module["train_dataset"], shuffle=True, # we shuffle for examples.
            batch_size=self.training_args.batch_size, 
            collate_fn=data_module["data_collator"],
            generator=g)
        return train_dataloader
    
    def train(self, examples, **kwargs):
        pass
        
    def save(self, dump_dir, **kwargs):
        model_name = kwargs.get("model_name", self.__str__())
        weight_file = dump_dir / f"{model_name}_weight.pt"
        weight = self.ax.proj.weight.data.cpu()
        if weight_file.exists():
            weight = torch.cat([torch.load(weight_file), weight], dim=0)
        torch.save(weight, weight_file)
        
        bias_file = dump_dir / f"{model_name}_bias.pt"
        bias = self.ax.proj.bias.data.cpu()
        if bias_file.exists():
            bias = torch.cat([torch.load(bias_file), bias], dim=0)
        torch.save(bias, bias_file)

    def load(self, dump_dir=None, **kwargs):
        model_name = kwargs.get("model_name", self.__str__())
        weight = torch.load(
            f"{dump_dir}/{model_name}_weight.pt"
        )
        bias = torch.load(
            f"{dump_dir}/{model_name}_bias.pt"
        )
        self.make_model(low_rank_dimension=weight.shape[0], **kwargs)
        self.ax.proj.weight.data = weight.to(self.device)
        self.ax.proj.bias.data = bias.to(self.device)
    
    @torch.no_grad()
    def predict_latent(self, examples, **kwargs):
        self.ax.eval()
        batch_size = kwargs.get('batch_size', 32)
        return_max_act_only = kwargs.get("return_max_act_only", False)
        is_chat_model = kwargs.get("is_chat_model", False)
        eager_prepare_df = kwargs.get("eager_prepare_df", False)
        overwrite_concept_id = kwargs.get("overwrite_concept_id", None)
        
        all_acts = []
        all_max_act = []
        all_max_act_idx = []
        all_max_token = []
        all_tokens = []
        # Process in batches
        progress_bar = tqdm(range(0, len(examples), batch_size), desc="Processing batches")
        for i in progress_bar:
            batch = examples.iloc[i:i + batch_size]
            if eager_prepare_df:
                batch = prepare_df(batch, self.tokenizer, is_chat_model)
            
            # Batch encode all inputs and send to model's device
            inputs = self.tokenizer(
                batch["input"].tolist(),
                return_tensors="pt",
                padding=True,
                add_special_tokens=True
            ).to(self.device)  # Use model's device

            act_in = gather_residual_activations(
                self.model, self.layer, inputs)
            
            ax_acts_batch = self.ax(act_in[:, kwargs["prefix_length"]:])  # no bos token
            # Process each sequence in the batch
            seq_lens = inputs["attention_mask"].sum(dim=1) - kwargs["prefix_length"] # no bos token
            for seq_idx, row in enumerate(batch.itertuples()):
                # select acts with attention mask
                acts = ax_acts_batch[
                    seq_idx, :seq_lens[seq_idx], overwrite_concept_id if overwrite_concept_id is not None else row.concept_id].flatten().float().cpu().numpy().tolist()
                acts = [round(x, 3) for x in acts]
                max_act = max(acts)
                all_max_act.append(max_act)
                if not return_max_act_only:
                    max_act_indices = [i for i, x in enumerate(acts) if x == max_act]
                    max_act_idx = max_act_indices[0]
                    # Get tokens for this specific sequence
                    tokens = self.tokenizer.tokenize(row.input)[kwargs["prefix_length"]-1:] # -1 is because it does not prepend BOS token
                    max_token = tokens[max_act_idx]
                    all_acts.append(acts)
                    all_max_act_idx.append(max_act_idx)
                    all_max_token.append(max_token)
                    all_tokens.append(tokens)
            # clear memory and cache
            del ax_acts_batch
            del act_in
            torch.cuda.empty_cache()

        if return_max_act_only:
            return {
                "max_act": all_max_act
            }
        return {
            "acts": all_acts,
            "max_act": all_max_act,
            "max_act_idx": all_max_act_idx,
            "max_token": all_max_token,
            "tokens": all_tokens
        }

    @torch.no_grad()
    def predict_latents(self, examples, **kwargs):
        self.ax.eval()
        batch_size = kwargs.get('batch_size', 32)

        all_acts = []
        all_max_act = []
        all_max_act_idx = []
        all_max_token = []
        all_tokens = []
        # Process in batches
        for i in range(0, len(examples), batch_size):
            batch = examples.iloc[i:i + batch_size]
            
            # Batch encode all inputs and send to model's device
            inputs = self.tokenizer(
                batch["input"].tolist(),
                return_tensors="pt",
                padding=True,
                add_special_tokens=True
            ).to(self.device)  # Use model's device
            
            act_in = gather_residual_activations(
                self.model, self.layer, inputs)
            
            ax_acts_batch = self.ax(act_in[:, kwargs["prefix_length"]:]).float().cpu().numpy()  # no bos token
            # Process each sequence in the batch
            seq_lens = inputs["attention_mask"].sum(dim=1) - kwargs["prefix_length"] # no bos token
            for seq_idx, row in enumerate(batch.itertuples()):
                # select acts with attention mask
                acts_batch = ax_acts_batch[
                    seq_idx, :seq_lens[seq_idx]]
                
                concept_acts = []
                concept_max_act = []
                concept_max_act_idx = []
                concept_max_token = []
                concept_tokens = []
                for row_idx in range(ax_acts_batch.shape[-1]):
                    # row_idx here is the concept id
                    acts = acts_batch[:, row_idx].flatten().tolist()
                    acts = [round(x, 3) for x in acts]
                    max_act = max(acts)
                    max_act_indices = [i for i, x in enumerate(acts) if x == max_act]
                    max_act_idx = max_act_indices[0]
                    # Get tokens for this specific sequence
                    tokens = self.tokenizer.tokenize(row.input)[kwargs["prefix_length"]-1:] # -1 is because it does not prepend BOS token
                    max_token = tokens[max_act_idx]
                    concept_acts.append(acts)
                    concept_max_act.append(max_act)
                    concept_max_act_idx.append(max_act_idx)
                    concept_max_token.append(max_token)
                    concept_tokens.append(tokens)
                all_acts.append(concept_acts)
                all_max_act.append(concept_max_act)
                all_max_act_idx.append(concept_max_act_idx)
                all_max_token.append(concept_max_token)
                all_tokens.append(concept_tokens)

        return {
            # "acts": all_acts,
            "max_act": all_max_act,
            # "max_act_idx": all_max_act_idx,
            # "max_token": all_max_token,
            # "tokens": all_tokens
        }

    @torch.no_grad()
    def predict_steer(self, examples, **kwargs):
        self.ax.eval()
        # set tokenizer padding to left
        self.tokenizer.padding_side = "left"
        # depending on the model, we use different concept id columns
        concept_id_col = "sae_id" if "sae" in self.__str__().lower() and not kwargs.get("disable_neuronpedia_max_act", False) else "concept_id"
        use_synergy = kwargs.get("use_synergy", False)

        # iterate rows in batch
        batch_size = kwargs.get("batch_size", 64)
        eval_output_length = kwargs.get("eval_output_length", 128)
        temperature = kwargs.get("temperature", 1.0)
        all_generations = []
        all_perplexities = []
        all_strenghts = []
        # Main training loop.
        rank = torch.distributed.get_rank()
        progress_bar = tqdm(range(0, len(examples), batch_size), position=rank, leave=True)
        for i in range(0, len(examples), batch_size):
            batch_examples = examples.iloc[i:i+batch_size]
            if use_synergy:
                # print("Using steered prompt to evaluate synergy of prompt and lsreft.")
                input_strings = batch_examples['steered_input'].tolist()
            else:
                input_strings = batch_examples['input'].tolist()
            mag = torch.tensor(batch_examples['factor'].tolist()).to(self.device)
            idx = torch.tensor(batch_examples["concept_id"].tolist()).to(self.device)
            max_acts = torch.tensor([
                self.max_activations.get(id, 1.0) 
                for id in batch_examples[concept_id_col].tolist()]).to(self.device)
            # logger.warning(f"Using max activations: {max_acts}")
            # tokenize input_strings
            inputs = self.tokenizer(
                input_strings, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            _, generations = self.ax_model.generate(
                inputs, 
                unit_locations=None, intervene_on_prompt=True, 
                subspaces=[{"idx": idx, "mag": mag, "max_act": max_acts, 
                            "prefix_length": kwargs["prefix_length"]}]*self.num_of_layers,
                max_new_tokens=eval_output_length, do_sample=True, 
                temperature=temperature,
            )

            # Decode and print only the generated text without prompt tokens
            input_lengths = [len(input_ids) for input_ids in inputs.input_ids]
            generated_texts = [
                self.tokenizer.decode(generation[input_length:], skip_special_tokens=True)
                for generation, input_length in zip(generations, input_lengths)
            ]
            all_generations += generated_texts

            # Calculate perplexity for each sequence
            unpruned_generated_texts = [
                self.tokenizer.decode(generation, skip_special_tokens=True)
                for generation in generations
            ]
            batch_input_ids = self.tokenizer(
                unpruned_generated_texts, return_tensors="pt", padding=True, truncation=True).input_ids.to(self.device)
            batch_attention_mask = (batch_input_ids != self.tokenizer.pad_token_id).float()
            
            # Forward pass without labels to get logits
            outputs = self.model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            
            logits = outputs.logits[:, :-1, :].contiguous()  # Remove last token prediction
            target_ids = batch_input_ids[:, 1:].contiguous()  # Shift right by 1
            
            # Calculate loss for each token
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            token_losses = loss_fct(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            
            # Reshape losses and mask
            token_losses = token_losses.view(batch_input_ids.size(0), -1)
            mask = batch_attention_mask[:, 1:].contiguous()
            
            # Calculate perplexity for each sequence
            seq_lengths = mask.sum(dim=1)
            seq_losses = (token_losses * mask).sum(dim=1) / seq_lengths
            seq_perplexities = torch.exp(seq_losses).tolist()
            all_perplexities.extend(seq_perplexities)
            all_strenghts.extend((mag*max_acts).tolist())
            progress_bar.update(1)

        return {
            "steered_generation": all_generations,
            "perplexity": all_perplexities,
            "strength": all_strenghts,
        }

    def get_logits(self, concept_id, k=10):
        top_logits, neg_logits = [None], [None]
        if concept_id is not None:
            W_U = self.model.lm_head.weight.T
            W_U = W_U * (self.model.model.norm.weight +
                        torch.ones_like(self.model.model.norm.weight))[:, None]
            W_U -= einops.reduce(
                W_U, "d_model d_vocab -> 1 d_vocab", "mean"
            )

            vocab_logits = self.ax.proj.weight.data[concept_id] @ W_U
            top_values, top_indices = vocab_logits.topk(k=k, sorted=True)
            top_tokens = self.tokenizer.batch_decode(top_indices.unsqueeze(dim=-1))
            top_logits = [list(zip(top_tokens, top_values.tolist()))]
            
            neg_values, neg_indices = vocab_logits.topk(k=k, largest=False, sorted=True)
            neg_tokens = self.tokenizer.batch_decode(neg_indices.unsqueeze(dim=-1))
            neg_logits = [list(zip(neg_tokens, neg_values.tolist()))]
        return top_logits, neg_logits
    
    def pre_compute_mean_activations(self, dump_dir, **kwargs):
        max_activations = {} # sae_id to max_activation
        # Loop over saved latent files in dump_dir.
        for file in os.listdir(dump_dir):
            if file.startswith("latent_") and file.endswith(".parquet"):
                latent_path = os.path.join(dump_dir, file)
                latent = pd.read_parquet(latent_path)
                # loop through unique sorted concept_id
                for concept_id in sorted(latent["concept_id"].unique()):
                    concept_latent = latent[latent["concept_id"] == concept_id]
                    max_act = concept_latent[f"{self.__str__()}_max_act"].max()
                    max_activations[concept_id] = max_act if max_act > 0 else 50
        self.max_activations = max_activations
        return max_activations  

    def to(self, device):
        """Move model to specified device"""
        self.device = device
        if hasattr(self, 'ax'):
            self.ax = self.ax.to(device)
            if hasattr(self, 'ax_model'):
                if isinstance(self.ax_model, IntervenableModel):
                    self.ax_model.set_device(device)
                else:
                    self.ax_model = self.ax_model.to(device)
        return self
