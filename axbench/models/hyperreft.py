from .model import Model
import torch, einops
import torch.nn as nn
import torch.nn.functional as F

from tqdm.auto import tqdm
import os
import pandas as pd
from pyvene import (
    IntervenableConfig,
    IntervenableModel
)
from .interventions import (
    HyperTopKReLUIntervention,
    TopKReLUSubspaceIntervention,
    HyperAdditionIntervention,
    SubspaceIntervention,
    SamplingAdditionIntervention,
    ThresholdingIntervention
)
from ..utils.data_utils import make_data_module
from ..utils.constants import EXAMPLE_TAG
from torch.utils.data import DataLoader
from ..utils.model_utils import (
    set_decoder_norm_to_unit_norm,
    remove_gradient_parallel_to_decoder_directions,
    gather_residual_activations, 
    get_lr,
    calculate_l1_losses
)
from transformers import get_scheduler, AutoTokenizer, AutoModelForCausalLM
from transformers import set_seed
from ..scripts.inference import prepare_df
import json


class RegressionWrapper(nn.Module):
    def __init__(self, base_model, hidden_size, output_dim):
        super().__init__()
        self.base_model = base_model
        self.regression_head = nn.Linear(hidden_size, output_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model.model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        last_hiddens = outputs.hidden_states[-1]
        last_token_representations = last_hiddens[:, -1]
        preds = self.regression_head(last_token_representations)
        preds = F.normalize(preds, p=2, dim=-1)
        return preds
    
    
class HyperReFT(Model):
    """HyperReFT"""
    def __str__(self):
        return 'HyperReFT'

    def make_model(self, **kwargs):
        mode = kwargs.get("mode", "latent")
        if mode == "steering":
            intervention_type = kwargs.get("intervention_type", "addition")
            if intervention_type == "thresholding":
                raise NotImplementedError("ThresholdingIntervention not implemented for HyperReFT")
                ax = ThresholdingIntervention(
                    embed_dim=self.model.config.hidden_size, 
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),
                )
            elif intervention_type == "sampling":
                raise NotImplementedError("SamplingAdditionIntervention not implemented for HyperReFT")
                ax = SamplingAdditionIntervention(
                    embed_dim=self.model.config.hidden_size, 
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),
                )
            elif intervention_type == "addition":
                ax = HyperAdditionIntervention(
                    embed_dim=self.model.config.hidden_size, 
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),
                )
            elif intervention_type == "clamping":
                raise NotImplementedError("SubspaceIntervention not implemented for HyperReFT")
                ax = SubspaceIntervention(
                    embed_dim=self.model.config.hidden_size, 
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),
                )
        else:
            intervention_type = kwargs.get("intervention_type", "addition")
            if intervention_type == "addition":
                ax = HyperTopKReLUIntervention(
                    embed_dim=self.model.config.hidden_size, 
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),
                )
            elif intervention_type == "clamping":
                raise NotImplementedError("TopKReLUSubspaceIntervention not implemented for HyperReFT")
                ax = TopKReLUSubspaceIntervention(
                    embed_dim=self.model.config.hidden_size, 
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),
                )
                
        layers = self.steering_layers if self.steering_layers else [self.layer]
        self.ax = ax.to(self.device)
        self.ax.train()
        ax_config = IntervenableConfig(representations=[{
            "layer": l,
            "component": f"model.layers[{l}].output",
            "low_rank_dimension": kwargs.get("low_rank_dimension", 1),
            "intervention": self.ax} for l in layers])
        ax_model = IntervenableModel(ax_config, self.model)
        ax_model.set_device(self.device)
        self.ax_model = ax_model
        
        reconstruction_dict_path = kwargs.get("reconstruction_dict_path", None)
        if reconstruction_dict_path is not None:
            self.reconstruction_dict = torch.load(os.path.join(reconstruction_dict_path, "LsReFT_weight.pt"))
        else:
            self.reconstruction_dict = None
            
        self.reconstruction_loss_ceoff = 0
        self.steering_loss_ceoff = 1
        
        base_model_name = kwargs.get("base_model_name", "google/gemma-2-2b")
        
        # Load the interpreting model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.bfloat16)
        
        self.base_model_tokenizer = AutoTokenizer.from_pretrained(base_model_name, model_max_length=512)
        self.base_model_tokenizer.padding_side = "left"
        
        if self.base_model_tokenizer.unk_token == None and self.base_model_tokenizer.pad_token == None:
            # raw llama3
            print("adding a special padding token...")
            self.base_model_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            need_resize = True
        else:
            need_resize = False
        if need_resize:
            self.base_model_tokenizer.resize_token_embeddings(len(self.base_model_tokenizer))
        
        self.concept_embedding = RegressionWrapper(
            base_model=base_model,
            hidden_size=base_model.config.hidden_size,
            output_dim=self.model.config.hidden_size
        )
        
        self.concept_embedding = self.concept_embedding.to(self.device, dtype=torch.bfloat16)
        
        # To easily test concept embedding through logit lens
        meta_data = kwargs.get("metadata", None)
        if meta_data is not None:
            self.concept_id_to_text = {}
            for d in meta_data:
                self.concept_id_to_text[d['concept_id']] = d['concept']

    def train(self, examples, **kwargs):
        train_dataloader = self.make_dataloader(examples, **kwargs)
        torch.cuda.empty_cache()
        # Optimizer and lr
        optimizer = torch.optim.AdamW(
            self.concept_embedding.parameters(), 
            lr=self.training_args.lr, weight_decay=self.training_args.weight_decay)
        num_training_steps = self.training_args.n_epochs * (len(train_dataloader) // self.training_args.gradient_accumulation_steps)
        lr_scheduler = get_scheduler(
            "linear", optimizer=optimizer,
            num_warmup_steps=0, num_training_steps=num_training_steps)
        norm_loss_fn = torch.nn.MSELoss()
        
        mse_criterion = nn.MSELoss(reduction="none")
        
        # Main training loop.
        # rank = torch.distributed.get_rank()
        rank = 0
        progress_bar, curr_step = tqdm(range(num_training_steps), position=rank, leave=True), 0
        
        if self.reconstruction_dict is not None:
            self.reconstruction_dict = self.reconstruction_dict.to(self.device)
        
        self.concept_embedding.train()
        
        for epoch in range(self.training_args.n_epochs):
            for step, batch in enumerate(train_dataloader):
                
                # prepare input
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                
                unit_locations={"sources->base": (
                    None,
                    inputs["intervention_locations"].permute(1, 0, 2).tolist()
                )}
                subspaces = [{
                    "k": self.training_args.topk
                }]
                
                #print(self.concept_embedding.regression_head.weight.grad)
                
                v = self.concept_embedding(
                    inputs["concept_input_ids"],
                    inputs["concept_attention_mask"],
                )
                                
                self.ax._update_v(v)
                
                # forward
                _, cf_outputs = self.ax_model(
                    base={
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"]
                    }, unit_locations=unit_locations, labels=inputs["labels"],
                    subspaces=subspaces, use_cache=False)

                # loss
                if self.reconstruction_dict is not None:
                    target_vector = self.reconstruction_dict[inputs["concept_ids"]]
                
                    mse_loss = mse_criterion(v, target_vector) 
                    mse_loss = mse_loss.sum(dim=-1).mean()
                    similarity = F.cosine_similarity(v, target_vector, dim=-1)  # shape: [batch_size]
                    cos_loss = 1.0 - similarity.mean()
                    recon_loss = mse_loss + cos_loss
                
                steering_loss = cf_outputs.loss
                latent, non_topk_latent = self.ax_model.full_intervention_outputs[0].latent
                l1_loss = calculate_l1_losses(
                    latent, non_topk_latent,
                    mask=inputs["intervention_masks"],
                )
                
                coeff = curr_step/num_training_steps
                steering_loss += coeff * self.training_args.coeff_latent_l1_loss * l1_loss
                steering_loss = steering_loss.mean()
                steering_loss /= self.training_args.gradient_accumulation_steps
                # grads
                
                loss = steering_loss * self.steering_loss_ceoff 
                
                if self.reconstruction_dict is not None:
                    loss += recon_loss * self.reconstruction_loss_ceoff

                    
                loss.backward()
                
                # clear the steering vector generated for this batch
                self.ax._reset_v()

                # Perform optimization step every gradient_accumulation_steps
                if (step + 1) % self.training_args.gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                    torch.nn.utils.clip_grad_norm_(self.concept_embedding.parameters(), 1.0)
                    # set_decoder_norm_to_unit_norm(self.ax)
                    
                    # TODO: need to be implimented for concept_embedding
                    # remove_gradient_parallel_to_decoder_directions(self.ax)
                    
                    curr_step += 1
                    curr_lr = get_lr(optimizer)
                    # optim
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    progress_bar.set_description(
                        "lr %.6f || loss %.6f || l1 loss %.6f" % (
                            curr_lr, loss, l1_loss))
        progress_bar.close()
        
    def make_dataloader(self, examples, **kwargs):
        data_module = make_data_module(self.tokenizer, examples, include_concept=True, concept_tokenizer=self.base_model_tokenizer, **kwargs)
        g = torch.Generator()
        g.manual_seed(self.seed)
        train_dataloader = DataLoader(
            data_module["train_dataset"], shuffle=True, # we shuffle for examples.
            batch_size=self.training_args.batch_size, 
            collate_fn=data_module["data_collator"],
            generator=g)
        return train_dataloader
    
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

            # Batch encode all inputs
            inputs = self.tokenizer(
                batch["input"].tolist(), return_tensors="pt", 
                add_special_tokens=True, padding=True, truncation=True).to(self.device)
            
            
            concept_inputs = self.base_model_tokenizer(
                batch["output_concept"].tolist(), return_tensors="pt",
                add_special_tokens=True, padding=True, truncation=True).to(self.device)
            
            v = self.concept_embedding(
                concept_inputs["input_ids"],
                concept_inputs["attention_mask"],
            )
            self.ax._update_v(v)

            gather_acts = gather_residual_activations(
                self.model, self.layer, inputs)
            outputs = self.ax(
                gather_acts[:, kwargs["prefix_length"]:],  # no bos token
                subspaces={"k": 1}
            )
            ax_acts = outputs.latent[0].float().detach().cpu()

            seq_lens = inputs["attention_mask"].sum(dim=1) - kwargs["prefix_length"] # no bos token
            # Process each sequence in the batch
            for seq_idx, ax_seq in enumerate(ax_acts):
                acts = ax_seq[:seq_lens[seq_idx]].flatten().data.numpy().tolist()
                acts = [round(x, 3) for x in acts]
                max_act = max(acts)
                all_max_act.append(max_act)
                if not return_max_act_only:
                    max_act_indices = [i for i, x in enumerate(acts) if x == max_act]
                    max_act_idx = max_act_indices[0]
                    # Get tokens for this specific sequence
                    tokens = self.tokenizer.tokenize(batch.iloc[seq_idx]["input"])[kwargs["prefix_length"]-1:] # -1 is because it does not prepend BOS token
                    max_token = tokens[max_act_idx]
                    all_acts.append(acts)
                    all_max_act_idx.append(max_act_idx)
                    all_max_token.append(max_token)
                    all_tokens.append(tokens)
            # clear memory and cache
            del ax_acts
            del gather_acts
            self.ax._reset_v()
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
            # Batch encode all inputs
            inputs = self.tokenizer(
                batch["input"].tolist(), return_tensors="pt", 
                add_special_tokens=True, padding=True, truncation=True).to(self.device)
            
            gather_acts = gather_residual_activations(
                self.model, self.layer, inputs)
            
            ax_acts_batch = torch.relu(torch.matmul(
                gather_acts[:, kwargs["prefix_length"]:], # bs, s, h
                self.ax.proj.weight.permute(1, 0) # h, d
            )).float().cpu().numpy()
            
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
    
    def save(self, dump_dir, **kwargs):
        model_name = kwargs.get("model_name", self.__str__())
        layer = self.steering_layers if self.steering_layers else self.layer
        weight_file = os.path.join(dump_dir, f"ckpt_{model_name}_{str(layer)}.pt")
        weight = self.concept_embedding.cpu()
        torch.save(weight, weight_file)

    def load(self, dump_dir=None, **kwargs):
        model_name = kwargs.get("model_name", self.__str__())
        layer = self.steering_layers if self.steering_layers else self.layer
        weight_file = os.path.join(dump_dir, f"ckpt_{model_name}_{str(layer)}.pt")        
        self.make_model(**kwargs)
        
        del self.concept_embedding
        self.concept_embedding = torch.load(weight_file, weights_only=False)
        self.concept_embedding.to(self.device)
        
    def get_logits(self, concept_id, k=10):
        top_logits, neg_logits = [None], [None]

        W_U = self.model.lm_head.weight.T
        W_U = W_U * (self.model.model.norm.weight +
                    torch.ones_like(self.model.model.norm.weight))[:, None]
        W_U -= einops.reduce(
            W_U, "d_model d_vocab -> 1 d_vocab", "mean"
        )
            
        concept_text = self.concept_id_to_text[concept_id]
        
        concept_input = self.base_model_tokenizer(
            concept_text, return_tensors="pt", 
            add_special_tokens=True, padding=True, truncation=True).to(self.device)
        
        concept_subspace = self.concept_embedding(
            concept_input["input_ids"],
            concept_input["attention_mask"],
        )
        
        vocab_logits = concept_subspace @ W_U
        top_values, top_indices = vocab_logits.topk(k=k, sorted=True)
        top_tokens = self.tokenizer.batch_decode(top_indices)

        top_logits = [list(zip(top_tokens, top_values.tolist()))]

        neg_values, neg_indices = vocab_logits.topk(k=k, largest=False, sorted=True)
        neg_tokens = self.tokenizer.batch_decode(neg_indices)
        neg_logits = [list(zip(neg_tokens, neg_values.tolist()))]

        return top_logits, neg_logits

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
        progress_bar = tqdm(range(0, len(examples), batch_size), leave=True)
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
            
            concept_inputs = self.base_model_tokenizer(
                batch_examples["input_concept"].tolist(), return_tensors="pt",
                add_special_tokens=True, padding=True, truncation=True).to(self.device)
            
            v = self.concept_embedding(
                concept_inputs["input_ids"],
                concept_inputs["attention_mask"],
            )

            self.ax._update_v(v)
            
            _, generations = self.ax_model.generate(
                inputs, 
                unit_locations=None, intervene_on_prompt=True, 
                subspaces=[{"idx": idx, "mag": mag, "max_act": max_acts, 
                            "prefix_length": kwargs["prefix_length"]}] * self.num_of_layers,
                max_new_tokens=eval_output_length, do_sample=True, 
                temperature=temperature,
            )
            
            self.ax._reset_v()
            
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