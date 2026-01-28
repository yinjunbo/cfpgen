
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


from typing import Optional
from typing import List
import torch
import torch.nn as nn
from byprot.models import register_model
from torch.nn import functional as F
from typing import List, Optional, Tuple, Union
from tqdm import tqdm
from transformers.models.esm.modeling_esm import *
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from copy import deepcopy
import random


class RCFEBlock(nn.Module):
    def __init__(self, base_block=None, block_index=None, hidden_size=None):
        super().__init__()

        self.copied_block = deepcopy(base_block)
        self.block_index = block_index
        self.hidden_size = hidden_size

        if self.block_index == 0:
            self.before_proj = nn.Linear(hidden_size, hidden_size)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)

        self.after_proj = nn.Linear(hidden_size, hidden_size)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)


    def forward(self, x, mask=None, c=None, y=None):

        if self.block_index == 0:
            c = self.before_proj(c)
            c = self.copied_block(x + c, mask, cond_input=y)[0]
            c_skip = self.after_proj(c)
        else:
            c = self.copied_block(c, mask, cond_input=y)[0]
            c_skip = self.after_proj(c)

        return c, c_skip


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AGFMSelfOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor, gate=None):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if gate is not None:
            hidden_states = gate.unsqueeze(1) * hidden_states + input_tensor
        else:
            hidden_states = hidden_states + input_tensor
        return hidden_states


class AGFMOutput(EsmOutput):
    def forward(self, hidden_states, input_tensor, gate=None):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if gate is not None:
            hidden_states = gate.unsqueeze(1) * hidden_states + input_tensor
        else:
            hidden_states = hidden_states + input_tensor
        return hidden_states


class AGFMAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self = EsmSelfAttention(config)
        self.output = AGFMSelfOutput(config)
        self.pruned_heads = set()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
            shift_msa=None,
            scale_msa=None,
            gate_msa=None,
    ):
        hidden_states_ln = self.LayerNorm(hidden_states)

        if shift_msa is not None:
            hidden_states_ln = modulate(hidden_states_ln, shift_msa, scale_msa)

        self_outputs = self.self(
            hidden_states_ln,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )

        if gate_msa is not None:
            attention_output = self.output(self_outputs[0], hidden_states, gate_msa)
        else:
            attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class AGFMLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_len_dim = 1
        self.attention = AGFMAttention(config)
        self.intermediate = EsmIntermediate(config)
        self.output = AGFMOutput(config)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_size, 4 * config.hidden_size, bias=True),  # use gate_msa
        )


    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
            cond_input=None,
    ):

        if cond_input is not None:
            shift_msa, scale_msa, shift_mlp, scale_mlp = self.adaLN_modulation(cond_input).chunk(4, dim=1)
            gate_msa = gate_mlp = None
        else:
            shift_msa = scale_msa = shift_mlp = scale_mlp = gate_msa = gate_mlp = None

        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            shift_msa=shift_msa,
            scale_msa=scale_msa,
            gate_msa=gate_msa,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        # feed_forward_chunk: layer_norm->linear(5120)->linear(1280)
        attention_output_ln = self.LayerNorm(attention_output)
        if cond_input is not None:
            attention_output_ln = modulate(attention_output_ln, shift_mlp, scale_mlp)
            intermediate_output = self.intermediate(attention_output_ln)
            layer_output = self.output(intermediate_output, attention_output, gate_mlp)
        else:
            intermediate_output = self.intermediate(attention_output_ln)
            layer_output = self.output(intermediate_output, attention_output)

        outputs = (layer_output,) + outputs
        return outputs


class ModifiedEsmSelfAttention(EsmSelfAttention):
    def __init__(self, config, position_embedding_type=None,
                 kdim=None,
                 vdim=None):
        super().__init__(config, position_embedding_type)
        if kdim is not None:
            self.key = nn.Linear(kdim, self.all_head_size)
        if vdim is not None:
            self.value = nn.Linear(vdim, self.all_head_size)


class ModifiedEsmAttention(EsmAttention):
    def __init__(self, config, kdim=None, vdim=None):
        super().__init__(config)
        self.self = ModifiedEsmSelfAttention(config, kdim=kdim, vdim=vdim)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
            cond_input=None,
    ):
        hidden_states_ln = self.LayerNorm(hidden_states)
        self_outputs = self.self(
            hidden_states_ln,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs



class ModifiedEsmLayer(EsmLayer):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ModifiedEsmAttention(config)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            print(self.is_decoder)
        self.intermediate = EsmIntermediate(config)
        self.output = EsmOutput(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
            cond_input=None
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            cond_input=cond_input,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise AttributeError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated"
                    " with cross-attention layers by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
                cond_input=cond_input
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value
        # layer_norm->linear(5120)->linear(1280)
        layer_output = self.feed_forward_chunk(attention_output)

        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)
        return outputs


class FuncTagEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        use_cfg_embedding = True
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


class CFPGenEncoder(EsmEncoder):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.emb_layer_norm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False

        self.use_go, self.use_ipr, self.use_ec = config.use_go, config.use_ipr, config.use_ec

        if self.use_go:
            self.go_class_num = config.go_num
            self.go_cls_dropout_all = config.go_drop
            self.go_cls_dropout_each = 0.1
            self.go_embedder = FuncTagEmbedder(config.go_num, config.hidden_size)

        if self.use_ipr:
            self.ipr_class_num = config.ipr_num
            self.ipr_cls_dropout_all = config.ipr_drop
            self.ipr_cls_dropout_each = 0.1
            self.ipr_embedder = FuncTagEmbedder(config.ipr_num, config.hidden_size)

        if self.use_ec:
            self.ec_class_num = config.ec_num
            self.ec_cls_dropout_all = config.ec_drop
            self.ec_cls_dropout_each = 0
            self.ec_embedder = FuncTagEmbedder(config.ec_num, config.hidden_size)


        self.layer = nn.ModuleList([AGFMLayer(deepcopy(config)) for _ in range(config.num_hidden_layers)])

        if config.use_seq_motif:
            self.copy_blocks_num = config.num_hidden_layers//2
            self.anno_dropout = 0.5
            self.seq_controlnet = nn.ModuleList(
                [RCFEBlock(ModifiedEsmLayer(deepcopy(config)), i, config.hidden_size) for i in range(self.copy_blocks_num)]
            )
        else:
            self.seq_controlnet = None


    def drop_anno_ids(self, class_tensor, embedder, class_num, training, drop_all_prob, drop_each_prob):
        """
        Drop annotation class IDs either at sample level or element level, then compute embeddings.
        """
        if training:
            # Drop all class IDs in a row with drop_all_prob
            drop_all = torch.rand(class_tensor.size(0), device=class_tensor.device) < drop_all_prob
            full_replacement = torch.full_like(class_tensor, class_num)
            class_tensor = torch.where(drop_all.unsqueeze(1), full_replacement, class_tensor)

            # Drop individual elements in class_tensor with drop_each_prob
            drop_each = torch.rand_like(class_tensor, dtype=torch.float) < drop_each_prob
            class_tensor = torch.where(drop_each, full_replacement, class_tensor)

        class_embeds = []
        for i, class_split in enumerate(class_tensor.split(1, dim=-1)):
            class_ids = class_split.squeeze(-1)
            class_embed = embedder(class_ids)
            # Zero-out embeddings where class_id == class_num (i.e., dropped)
            mask = (class_ids == class_num).unsqueeze(-1)
            class_embed = torch.where(mask, torch.zeros_like(class_embed), class_embed)
            class_embeds.append(class_embed)

        # Combine class embeddings by summation
        return torch.sum(torch.stack(class_embeds, dim=0), dim=0)


    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            **kwargs
    ):


        '''
        Annotation-Guided Feature Modulation (AGFM)
        '''

        anno_tag = kwargs.get('anno_tag')
        anno_embed = None

        if anno_tag is not None:

            go_class = anno_tag.get('go')
            ipr_class = anno_tag.get('ipr')
            ec_class = anno_tag.get('ec')

            seq_num = hidden_states.size(0)

            def prepare_class(cls, class_num):
                """Replace -1 with class_num and broadcast if needed."""
                if not self.training and cls.dim() == 1:
                    cls = cls.unsqueeze(0).repeat(seq_num, 1)
                return torch.where(cls == -1, torch.full_like(cls, class_num), cls)

            if self.use_go and go_class is not None:
                go_class = prepare_class(go_class, self.go_embedder.num_classes)
                anno_embed = self.drop_anno_ids(go_class, self.go_embedder, self.go_class_num,
                                                self.training, self.go_cls_dropout_all, self.go_cls_dropout_each)

            if self.use_ipr and ipr_class is not None:
                ipr_class = prepare_class(ipr_class, self.ipr_embedder.num_classes)
                ipr_embed = self.drop_anno_ids(ipr_class, self.ipr_embedder, self.ipr_class_num,
                                               self.training, self.ipr_cls_dropout_all, self.ipr_cls_dropout_each)
                anno_embed = anno_embed + ipr_embed if anno_embed is not None else ipr_embed

            if self.use_ec and ec_class is not None:
                ec_class = prepare_class(ec_class, self.ec_embedder.num_classes)
                ec_embed = self.drop_anno_ids(ec_class, self.ec_embedder, self.ec_class_num,
                                              self.training, self.ec_cls_dropout_all, self.ec_cls_dropout_each)
                anno_embed = anno_embed + ec_embed if anno_embed is not None else ec_embed


        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                    "`use_cache=False`..."
                )
                use_cache = False
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None

        '''
        Residue-Controlled Functional Encoding (RCFE)
        '''
        if self.seq_controlnet and anno_tag['seq_cond'] is not None and anno_tag['seq_cond'].numel() > 0:

            motif = anno_tag['seq_cond']
          
            random_go_embed = anno_embed if (not self.training or random.random() > self.anno_dropout) else None  # motif embedding 多大程度参考 global condition

            for index in range(1, self.copy_blocks_num + 1):
                motif, motif_skip = self.seq_controlnet[index - 1](hidden_states, attention_mask, motif, None)
                hidden_states = self.layer[index](hidden_states+motif_skip, attention_mask, cond_input=random_go_embed)[0]

            for index in range(self.copy_blocks_num + 1, len(self.layer)):
                hidden_states = self.layer[index](hidden_states, attention_mask, cond_input=random_go_embed)[0]

        else:
            for i, layer_module in enumerate(self.layer):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_head_mask = head_mask[i] if head_mask is not None else None
                past_key_value = past_key_values[i] if past_key_values is not None else None

                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        layer_module.__call__,
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        past_key_value,
                        output_attentions,
                    )
                else:
                    layer_outputs = layer_module(
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        past_key_value,
                        output_attentions,
                        anno_embed,
                    )

                hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache = next_decoder_cache + (layer_outputs[-1],)
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)
                    if self.config.add_cross_attention:
                        all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if self.emb_layer_norm_after:
            hidden_states = self.emb_layer_norm_after(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class ModifiedEsmModel(EsmModel):
    def __init__(self, config, add_pooling_layer=True):
        EsmPreTrainedModel.__init__(self, config)
        self.config = config

        self.embeddings = EsmEmbeddings(config)
        self.encoder = CFPGenEncoder(config)

        self.pooler = EsmPooler(config) if add_pooling_layer else None

        self.contact_head = EsmContactPredictionHead(
            in_features=config.num_hidden_layers * config.num_attention_heads, bias=True
        )

        self.hidden_size = config.hidden_size

        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                module.weight.data.fill_(1.0)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, AGFMLayer):
            # set shift and scale to 0
            nn.init.constant_(module.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(module.adaLN_modulation[-1].bias[:2 * self.hidden_size], 0)
            nn.init.constant_(module.adaLN_modulation[-1].bias[3 * self.hidden_size:5 * self.hidden_size], 0)
            # set gate to 1
            nn.init.constant_(module.adaLN_modulation[-1].bias[2 * self.hidden_size:3 * self.hidden_size], 1)
            nn.init.constant_(module.adaLN_modulation[-1].bias[5 * self.hidden_size:6 * self.hidden_size], 1)


    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            seq_cond_attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            try:
                input_shape = input_ids['x_t'].size()
            except (KeyError, TypeError, AttributeError, IndexError):
                input_shape = input_ids.size() if torch.is_tensor(input_ids) else None
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        if not torch.is_tensor(input_ids):
            device = input_ids['x_t'].device if input_ids is not None else inputs_embeds.device
        else:
            device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            # encoder_extended_attention_mask = None
            encoder_extended_attention_mask = encoder_attention_mask

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids if torch.is_tensor(input_ids) else input_ids['x_t'],
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        if input_ids['seq_cond'] is not None:
            seq_cond_embedding = self.embeddings(
                input_ids=input_ids['seq_cond'],
                position_ids=position_ids,
                attention_mask=seq_cond_attention_mask,
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_key_values_length,
            )
            input_ids['seq_cond'] = seq_cond_embedding
            input_ids['seq_cond_att_mask'] = seq_cond_attention_mask

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            anno_tag=input_ids,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


@register_model('func_mlm_esm')
class EsmForCFPGEN(EsmForMaskedLM):
    def __init__(self, config, dropout=0.1):
        tokenizer = AutoTokenizer.from_pretrained(config._name_or_path)
        config.hidden_dropout_prob = dropout
        
        EsmPreTrainedModel.__init__(self, config)
        self.esm = ModifiedEsmModel(config, add_pooling_layer=False)
        self.lm_head = EsmLMHead(config)

        self.init_weights()
        
        self.mask_id = tokenizer.mask_token_id
        self.pad_id = tokenizer.pad_token_id
        self.bos_id = tokenizer.cls_token_id
        self.eos_id = tokenizer.eos_token_id
        self.x_id = tokenizer._token_to_id['X']
        
        self.contact_head = None
        self.tokenizer = tokenizer
    
    def forward(self,
                input_ids,
                attention_mask=None,
                inputs_embeds=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                decoder_inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
            ):


        assert isinstance(input_ids, dict)
        attention_mask = input_ids['x_t'].ne(self.pad_id)

        seq_cond_attention_mask = input_ids['seq_cond'].ne(self.pad_id) if input_ids['seq_cond'] is not None else None

        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            seq_cond_attention_mask=seq_cond_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        sequence_output = outputs[0]
        logits = self.lm_head(sequence_output)
        
        result = {
            "logits": logits,
            "last_hidden_state": sequence_output,
        }
        return result


def sample_from_categorical(logits=None, temperature=1.0):
    if temperature:
        dist = torch.distributions.Categorical(logits=logits.div(temperature))
        tokens = dist.sample()
        scores = dist.log_prob(tokens)
    else:
        scores, tokens = logits.log_softmax(dim=-1).max(dim=-1)
    return tokens, scores