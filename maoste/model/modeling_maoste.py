import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch.nn import CrossEntropyLoss
from torch.nn import KLDivLoss
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Config
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_outputs import \
    BaseModelOutputWithPastAndCrossAttentions
from transformers.modeling_outputs import ModelOutput
from transformers.models.t5.modeling_t5 import T5Block
from transformers.models.t5.modeling_t5 import T5LayerNorm
from transformers.models.t5.modeling_t5 import T5Stack

from maoste.model.vision_t5.generation_utils_vision_t5 import \
    GenerationMixin_VisionT5

class TransformerCrossModalAttention(nn.Module):

  def __init__(self, text_dim, visual_dim, num_heads):
    super(TransformerCrossModalAttention, self).__init__()
    self.num_heads = num_heads
    self.text_dim = text_dim
    self.visual_dim = visual_dim

    # Ensure the text and visual dimensions are compatible with the number of heads
    assert text_dim % num_heads == 0
    assert visual_dim % num_heads == 0

    self.text_to_query = nn.Linear(text_dim, text_dim)
    self.visual_to_key = nn.Linear(visual_dim, text_dim)
    self.visual_to_value = nn.Linear(visual_dim, visual_dim)

    self.final_linear = nn.Linear(visual_dim, visual_dim)

  def forward(self,
              text_features,
              visual_features,
              text_mask=None,
              visual_mask=None):

    # Prepare query, key, value
    query = self.text_to_query(
        text_features)  # (batch_size, num_text_tokens, text_dim)
    key = self.visual_to_key(
        visual_features)  # (batch_size, num_visual_features, text_dim)
    value = self.visual_to_value(
        visual_features)  # (batch_size, num_visual_features, visual_dim)

    # Reshape for multi-head attention
    batch_size = query.size(0)
    query = query.view(batch_size, -1, self.num_heads,
                       self.text_dim // self.num_heads).transpose(1, 2)
    key = key.view(batch_size, -1, self.num_heads,
                   self.text_dim // self.num_heads).transpose(1, 2)
    value = value.view(batch_size, -1, self.num_heads,
                       self.visual_dim // self.num_heads).transpose(1, 2)

    # Scaled dot-product attention
    scores = torch.matmul(query, key.transpose(-2, -1)) / (self.text_dim**0.5)
    if visual_mask is not None:
      scores = scores.masked_fill(
          visual_mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
    attention_weights = F.softmax(scores, dim=-1)

    # Apply attention
    attended_visuals = torch.matmul(attention_weights, value)
    attended_visuals = attended_visuals.transpose(1, 2).contiguous().view(
        batch_size, -1, self.visual_dim)

    # Final linear layer
    attended_visuals = self.final_linear(attended_visuals)

    return attended_visuals


class Encoder(T5Stack):

  def __init__(self, config, embed_tokens=None):
    super().__init__(config)
    self.embed_tokens = embed_tokens
    self.is_decoder = config.is_decoder

    # ---- Modified ----#
    # add visual features (without position features)
    self.visual_feat_embedding = nn.Linear(config.feat_dim, config.d_model)
    self.cross_attention_vis = TransformerCrossModalAttention(config.d_model,
                                                              config.d_model,
                                                              num_heads=8)
    # ------------------#

    self.block = nn.ModuleList([
        T5Block(config, has_relative_attention_bias=bool(i == 0))
        for i in range(config.num_layers)
    ])
    self.final_layer_norm = T5LayerNorm(config.d_model,
                                        eps=config.layer_norm_epsilon)
    self.dropout = nn.Dropout(config.dropout_rate)

    # Initialize weights and apply final processing
    self.init_weights()
    # Model parallel
    self.model_parallel = False
    self.device_map = None
    self.gradient_checkpointing = False

    self.vis_embeds = None

  def forward(
      self,
      input_ids=None,
      attention_mask=None,
      vis_feats=None,
      vis_attention_mask=None,
      inputs_embeds=None,
      head_mask=None,
      past_key_values=None,
      use_cache=None,
      output_attentions=None,
      output_hidden_states=None,
      return_dict=None,
      **kwargs,
  ):
    if inputs_embeds is None:
      assert self.embed_tokens is not None, 'You have to initialize the model with valid token embeddings'
      inputs_embeds = self.embed_tokens(input_ids)
    # add
    vis_embeds = self.visual_feat_embedding(vis_feats)  # B, vvl_region_num, D

    if vis_attention_mask is not None and self.config.enable_global_img_feat:
      global_vis_embeds = vis_embeds.mean(1, keepdim=True)
      vis_embeds = torch.cat([global_vis_embeds, vis_embeds], dim=1)
      vis_attention_mask = torch.cat([
          vis_attention_mask.new_ones(vis_embeds.shape[0], 1),
          vis_attention_mask
      ],
                                     dim=1)
    # add: cross attention
    text_vis_embeds = self.cross_attention_vis(inputs_embeds, vis_embeds,
                                               attention_mask,
                                               vis_attention_mask)
    inputs_embeds = inputs_embeds + text_vis_embeds
    inputs_embeds = torch.cat([inputs_embeds, vis_embeds], dim=1)

    batch_size, text_seq_length = inputs_embeds.size()[:-1]
    self.vis_embeds = vis_embeds

    vis_seq_length = vis_embeds.size(1)
    seq_length = text_seq_length + vis_seq_length
    input_shape = (batch_size, seq_length)

    # required mask seq length can be calculated via length of past
    mask_text_seq_length = past_key_values[0][0].shape[2] + \
                           text_seq_length if past_key_values is not None else text_seq_length

    if use_cache is True:
      assert self.is_decoder, f'`use_cache` can only be set to `True` if {self} is used as a decoder'

    if attention_mask is None:
      attention_mask = torch.ones(batch_size,
                                  mask_text_seq_length).to(inputs_embeds.device)

    # add vis_attention_mask
    if vis_attention_mask is None:
      # new_ones returns same tensor.dtype and device
      vis_attention_mask = attention_mask.new_ones(batch_size, vis_seq_length)

    attention_mask = torch.cat([attention_mask, vis_attention_mask], dim=1)
    # ------------------#

    # initialize past_key_values with `None` if past does not exist
    if past_key_values is None:
      past_key_values = [None] * len(self.block)

    # ------------------#
    extended_attention_mask = self.get_extended_attention_mask(
        attention_mask, input_shape, inputs_embeds.device)

    # If a 2D or 3D attention mask is provided for the cross-attention
    # we need to make broadcastable to [batch_size, num_heads, text_seq_length, text_seq_length]
    encoder_extended_attention_mask = None

    # Prepare head mask if needed
    head_mask = self.get_head_mask(head_mask, self.config.num_layers)
    present_key_value_states = () if use_cache else None
    all_hidden_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None
    all_cross_attentions = () if (output_attentions and
                                  self.is_decoder) else None
    position_bias = None
    encoder_decoder_position_bias = None

    hidden_states = self.dropout(inputs_embeds)

    for i, (layer_module,
            past_key_value) in enumerate(zip(self.block, past_key_values)):
      layer_head_mask = head_mask[i]
      # Model parallel
      layer_outputs = layer_module(
          hidden_states,
          attention_mask=extended_attention_mask,
          position_bias=position_bias,
          encoder_hidden_states=None,
          encoder_attention_mask=encoder_extended_attention_mask,
          encoder_decoder_position_bias=encoder_decoder_position_bias,
          layer_head_mask=layer_head_mask,
          past_key_value=past_key_value,
          use_cache=use_cache,
          output_attentions=output_attentions,
      )

      # layer_outputs is a tuple with: hidden-states, key-value-states, (self-attention position bias),
      # (self-attention weights), (cross-attention position bias), (cross-attention weights)
      if not use_cache:
        layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

      hidden_states, present_key_value_state = layer_outputs[:2]

      # We share the position biases between the layers - the first layer store them
      # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
      # (cross-attention position bias), (cross-attention weights)
      position_bias = layer_outputs[2]
      # append next layer key value states
      if use_cache:
        present_key_value_states = present_key_value_states + \
                                   (present_key_value_state,)

      if output_attentions:
        all_attentions = all_attentions + (layer_outputs[3],)
        if self.is_decoder:
          all_cross_attentions = all_cross_attentions + \
                                 (layer_outputs[5],)

    hidden_states = self.final_layer_norm(hidden_states)
    hidden_states = self.dropout(hidden_states)

    # Add last layer
    if output_hidden_states:
      all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
      return tuple(v for v in [
          hidden_states,
          present_key_value_states,
          all_hidden_states,
          all_attentions,
          all_cross_attentions,
      ] if v is not None)
    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=present_key_value_states,
        hidden_states=all_hidden_states,
        attentions=all_attentions,
        cross_attentions=all_cross_attentions,
    )


class VisionT5(GenerationMixin_VisionT5, T5ForConditionalGeneration):
  _keys_to_ignore_on_load_missing = [
      r'encoder\.embed_tokens\.weight',
      r'decoder\.embed_tokens\.weight',
      r'lm_head\.weight',
  ]
  _keys_to_ignore_on_load_unexpected = [
      r'decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight',
  ]

  def __init__(self, config: T5Config):
    super().__init__(config)

    self.config = config
    self.model_dim = config.d_model

    self.shared = nn.Embedding(config.vocab_size, config.d_model)

    encoder_config = copy.deepcopy(config)
    encoder_config.is_decoder = False
    encoder_config.use_cache = False
    encoder_config.is_encoder_decoder = False
    # ---- Modified ----#
    # self.encoder = T5Stack(encoder_config, self.shared)
    self.encoder = Encoder(encoder_config, self.shared)
    # ------------------#

    decoder_config = copy.deepcopy(config)
    decoder_config.is_decoder = True
    decoder_config.is_encoder_decoder = False
    decoder_config.num_layers = config.num_decoder_layers
    self.decoder = T5Stack(decoder_config, self.shared)

    self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    self.classifier = nn.Linear(config.d_model, config.pos_dim, bias=False)

    self.global_anp_classifier = nn.Linear(config.d_model, config.img_anp_dim)

    # Initialize weights and apply final processing
    self.init_weights()

    # Model parallel
    self.model_parallel = False
    self.device_map = None

    self.vinvl_region_num = config.vinvl_region_num

  def get_input_embeddings(self):
    return super().get_input_embeddings()

  def set_input_embeddings(self, new_embeddings):
    return super().set_input_embeddings(new_embeddings)

  def set_output_embeddings(self, new_embeddings):
    return super().set_output_embeddings(new_embeddings)

  def get_encoder(self):
    return super().get_encoder()

  def get_decoder(self):
    return super().get_decoder()

  def forward(
      self,
      input_ids=None,
      attention_mask=None,
      vis_feats=None,
      vis_attention_mask=None,
      img_label=None,
      img_anp_label=None,
      decoder_input_ids=None,
      decoder_attention_mask=None,
      head_mask=None,
      decoder_head_mask=None,
      encoder_outputs=None,
      past_key_values=None,
      inputs_embeds=None,
      decoder_inputs_embeds=None,
      labels=None,
      use_cache=None,
      output_attentions=None,
      output_hidden_states=None,
      return_dict=None,
      **kwargs,
  ):
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # Encode if needed (training, first prediction pass)
    if encoder_outputs is None:
      # Convert encoder inputs in embeddings if needed
      encoder_outputs = self.encoder(
          input_ids=input_ids,
          attention_mask=attention_mask,
          vis_feats=vis_feats,
          vis_attention_mask=vis_attention_mask,
          inputs_embeds=inputs_embeds,
          head_mask=head_mask,
          output_attentions=output_attentions,
          output_hidden_states=output_hidden_states,
          return_dict=return_dict,
      )
    elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
      encoder_outputs = BaseModelOutput(
          last_hidden_state=encoder_outputs[0],
          hidden_states=encoder_outputs[1]
          if len(encoder_outputs) > 1 else None,
          attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
      )

    hidden_states = encoder_outputs[0]

    if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
      # get decoder inputs from shifting lm labels to the right
      decoder_input_ids = self._shift_right(labels)

    # If decoding with past key value states, only the last tokens
    # should be given as an input
    if past_key_values is not None:
      # assert labels is not None, "Decoder should not use cached key states when training." # TODO
      if decoder_input_ids is not None:
        decoder_input_ids = decoder_input_ids[:, -1:]
      if decoder_inputs_embeds is not None:
        decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

    if attention_mask is None:
      attention_mask = input_ids.ne(self.config.pad_token_id).to(
          dtype=hidden_states.dtype, device=hidden_states.device)

    if vis_attention_mask is None:
      batch_size, text_seq_length = attention_mask.size()
      vis_seq_length = encoder_outputs[0].size(1) - text_seq_length
      vis_attention_mask = attention_mask.new_ones(batch_size, vis_seq_length)
    elif self.config.enable_global_img_feat:
      vis_attention_mask = torch.cat([
          vis_attention_mask.new_ones(self.encoder.vis_embeds.size(0), 1),
          vis_attention_mask
      ],
                                     dim=1)
    encoder_attention_mask = torch.cat([attention_mask, vis_attention_mask],
                                       dim=1)

    decoder_outputs = self.decoder(
        input_ids=decoder_input_ids,
        attention_mask=decoder_attention_mask,
        inputs_embeds=decoder_inputs_embeds,
        past_key_values=past_key_values,
        encoder_hidden_states=hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        head_mask=decoder_head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    sequence_output = decoder_outputs[0]

    if self.config.tie_word_embeddings:
      # Rescale output before projecting on vocab
      # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
      sequence_output = sequence_output * (self.model_dim**-0.5)

    lm_logits = self.lm_head(sequence_output)
    i = int(self.config.enable_global_img_feat)
    vis_simi = torch.matmul(sequence_output,
                            self.encoder.vis_embeds[:, i:, :].transpose(1, 2))
    loss = None
    img_region_simi_kl_loss = None
    img_anp_kl_loss = None

    if labels is not None:
      vis_simi, cumsum, _ = self.get_vision_similarity_in_image(
          labels, vis_simi)

      mask_img_label = []
      for st, ed in zip(cumsum[:-1], cumsum[1:]):
        nums = (ed - st)
        mask_img_label.append([True] * nums + [False] *
                              (img_label.size(1) - nums))

      mask_img_label = torch.tensor(mask_img_label)
      img_label = img_label[mask_img_label]

      loss_fct = CrossEntropyLoss(ignore_index=-100)
      loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
      kl_loss_fct = KLDivLoss(reduction='batchmean')

      if img_label.size(0) == 0:
        img_region_simi_kl_loss = torch.tensor(0.0).to(vis_simi.device)
      else:
        img_region_simi_kl_loss = kl_loss_fct(input=F.log_softmax(vis_simi.view(
            -1, 3, self.vinvl_region_num).mean(dim=1),
                                                                  dim=1),
                                              target=img_label)
      loss += img_region_simi_kl_loss

    if self.config.enable_global_img_feat and isinstance(
        img_anp_label, torch.Tensor):
      global_vis_embeds = self.encoder.vis_embeds[:, 0, :]
      global_anp_preds = self.global_anp_classifier(global_vis_embeds)
      if img_anp_label.size(0) == 0:
        img_anp_kl_loss = torch.tensor(0.0).to(global_anp_preds.device)
      else:
        img_anp_kl_loss = kl_loss_fct(input=F.log_softmax(global_anp_preds,
                                                          dim=1),
                                      target=img_anp_label)
      loss += img_anp_kl_loss

    if not return_dict:
      output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
      return ((loss,) + output) if loss is not None else output

    return Seq2SeqLMOutput_VisionT5(
        loss=loss,
        logits=lm_logits,
        vis_similarities=vis_simi,
        past_key_values=decoder_outputs.past_key_values,
        decoder_hidden_states=decoder_outputs.hidden_states,
        decoder_attentions=decoder_outputs.attentions,
        cross_attentions=decoder_outputs.cross_attentions,
        encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        encoder_hidden_states=encoder_outputs.hidden_states,
        encoder_attentions=encoder_outputs.attentions,
    )

  def prepare_inputs_for_generation(self,
                                    input_ids,
                                    past=None,
                                    attention_mask=None,
                                    head_mask=None,
                                    decoder_head_mask=None,
                                    cross_attn_head_mask=None,
                                    use_cache=None,
                                    encoder_outputs=None,
                                    **kwargs):

    # cut decoder_input_ids if past is used
    if past is not None:
      input_ids = input_ids[:, -1:]

    output = {
        'decoder_input_ids': input_ids,
        'past_key_values': past,
        'encoder_outputs': encoder_outputs,
        'attention_mask': attention_mask,
        'head_mask': head_mask,
        'decoder_head_mask': decoder_head_mask,
        'use_cache': use_cache,
    }

    if 'vis_attention_mask' in kwargs:
      output['vis_attention_mask'] = kwargs['vis_attention_mask']

    return output

  def _expand_inputs_for_generation(
      input_ids,
      expand_size=1,
      is_encoder_decoder=False,
      attention_mask=None,
      encoder_outputs=None,
      **model_kwargs,
  ):

    expanded_return_idx = (torch.arange(input_ids.shape[0]).view(-1, 1).repeat(
        1, expand_size).view(-1).to(input_ids.device))
    input_ids = input_ids.index_select(0, expanded_return_idx)

    if 'token_type_ids' in model_kwargs:
      token_type_ids = model_kwargs['token_type_ids']
      model_kwargs['token_type_ids'] = token_type_ids.index_select(
          0, expanded_return_idx)

    if attention_mask is not None:
      model_kwargs['attention_mask'] = attention_mask.index_select(
          0, expanded_return_idx)

    # ---- Modified ----#
    if model_kwargs.get('vis_attention_mask', None) is not None:
      model_kwargs['vis_attention_mask'] = model_kwargs[
          'vis_attention_mask'].index_select(0, expanded_return_idx)
    # ------------------#

    if is_encoder_decoder:
      if encoder_outputs is None:
        raise ValueError(
            'If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.'
        )
      encoder_outputs[
          'last_hidden_state'] = encoder_outputs.last_hidden_state.index_select(
              0,
              expanded_return_idx.to(encoder_outputs.last_hidden_state.device))
      model_kwargs['encoder_outputs'] = encoder_outputs
    return input_ids, model_kwargs


@dataclass
class Seq2SeqLMOutput_VisionT5(ModelOutput):
  loss: Optional[torch.FloatTensor] = None
  logits: torch.FloatTensor = None
  vis_similarities: torch.FloatTensor = None
  past_key_values: Optional[List[torch.FloatTensor]] = None
  decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
  decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
  cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
  encoder_last_hidden_state: Optional[torch.FloatTensor] = None
  encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
  encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
