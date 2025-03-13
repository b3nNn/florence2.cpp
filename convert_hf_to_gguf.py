import torch
from transformers import AutoModelForCausalLM
import gguf
import numpy as np
import os
import logging
import argparse
import json

logging.basicConfig(level=logging.INFO)  # Adjusted to INFO for cleaner output
logger = logging.getLogger(__name__)

CUSTOM_ARCH_FLORENCE2 = "florence2"

def validate_mapping(gguf_file, mapping_file):
    """Validate that all tensors in the mapping exist in the GGUF file with correct names."""
    gguf_reader = gguf.GGUFReader(gguf_file)
    gguf_tensors = {tensor.name for tensor in gguf_reader.tensors}

    with open(mapping_file, 'r') as f:
        tensor_mapping = json.load(f)

    mapped_tensors = set(tensor_mapping.values())

    # Check for missing tensors in GGUF
    missing_in_gguf = mapped_tensors - gguf_tensors
    if missing_in_gguf:
        logger.error(f"Validation failed: Tensors in mapping not found in GGUF: {missing_in_gguf}")
        return False

    # Check for unmapped tensors in GGUF (optional, but useful for completeness)
    unmapped_in_gguf = gguf_tensors - mapped_tensors
    if unmapped_in_gguf:
        logger.error(f"Validation failed: Tensors in GGUF not in mapping: {unmapped_in_gguf}")
        return False

    logger.info("Tensor mapping validation passed")
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir_model", help="directory containing the model")
    parser.add_argument("--outfile", help="output GGUF file")
    args = parser.parse_args()

    dir_model = args.dir_model
    if not os.path.isdir(dir_model):
        logger.error(f"Error: {dir_model} is not a directory")
        exit(1)

    outfile = args.outfile or "florence2_large_ft.gguf"
    mapping_file = "tensors_mapping.json"

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        dir_model,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16
    )
    state_dict = model.state_dict()

    # Initialize GGUF writer with float16
    gguf_writer = gguf.GGUFWriter(outfile, CUSTOM_ARCH_FLORENCE2, use_temp_file=True)

    # Metadata
    gguf_writer.add_name("Florence-2-large-ft")
    gguf_writer.add_description("Florence-2 vision-language model in GGUF")
    gguf_writer.add_author("Microsoft Research")
    gguf_writer.add_uint32("vocab_size", model.config.vocab_size)
    gguf_writer.add_uint32("proj_dim", model.config.projection_dim)
    gguf_writer.add_uint32("bos_id", model.config.bos_token_id)
    gguf_writer.add_uint32("eos_id", model.config.eos_token_id)
    gguf_writer.add_uint32("pad_id", model.config.pad_token_id)

    try:
        gguf_writer.add_uint32("general.alignment", gguf.GGUF_DEFAULT_ALIGNMENT)
    except AttributeError:
        gguf_writer.add_uint32("general.alignment", 32)

    text_config = model.config.text_config
    gguf_writer.add_uint32("t_vocab_size", text_config.vocab_size)
    gguf_writer.add_uint32("t_max_pos", text_config.max_position_embeddings)
    gguf_writer.add_uint32("t_hidden", text_config.d_model)
    gguf_writer.add_uint32("t_dec_layers", text_config.decoder_layers)
    gguf_writer.add_uint32("t_enc_layers", text_config.encoder_layers)
    gguf_writer.add_uint32("t_heads", text_config.decoder_attention_heads)
    gguf_writer.add_uint32("t_ffn_dim", text_config.decoder_ffn_dim)

    vision_config = model.config.vision_config
    gguf_writer.add_uint32("v_img_size", 768)
    gguf_writer.add_uint32("v_stages", len(vision_config.depths))
    gguf_writer.add_array("v_patch_size", vision_config.patch_size)
    gguf_writer.add_array("v_patch_stride", vision_config.patch_stride)
    gguf_writer.add_array("v_patch_pad", vision_config.patch_padding)
    gguf_writer.add_array("v_dim_embed", vision_config.dim_embed)
    gguf_writer.add_array("v_heads", vision_config.num_heads)
    gguf_writer.add_array("v_depths", vision_config.depths)
    gguf_writer.add_uint32("v_win_size", vision_config.window_size)
    gguf_writer.add_uint32("v_proj_dim", vision_config.projection_dim)

    # Weight mappings
    weight_map_language = {
        "language_model.model.shared.weight": "t_shared_w",
        "language_model.model.decoder.embed_tokens.weight": "t_embd_w",
        "language_model.model.decoder.embed_positions.weight": "t_pos_w",
        "language_model.model.decoder.layernorm_embedding.weight": "t_lnorm_w",
        "language_model.model.decoder.layernorm_embedding.bias": "t_lnorm_b",
        "language_model.lm_head.weight": "t_out_w",
        "language_model.final_logits_bias": "t_logits_b",
        "language_model.model.encoder.embed_tokens.weight": "t_enc_embd_w",
        "language_model.model.encoder.embed_positions.weight": "t_enc_pos_w",
        "language_model.model.encoder.layernorm_embedding.weight": "t_enc_lnorm_w",
        "language_model.model.encoder.layernorm_embedding.bias": "t_enc_lnorm_b"
    }
    for i in range(text_config.decoder_layers):
        weight_map_language[f"language_model.model.decoder.layers.{i}.self_attn.q_proj.weight"] = f"t{i}a_qw"
        weight_map_language[f"language_model.model.decoder.layers.{i}.self_attn.k_proj.weight"] = f"t{i}a_kw"
        weight_map_language[f"language_model.model.decoder.layers.{i}.self_attn.v_proj.weight"] = f"t{i}a_vw"
        weight_map_language[f"language_model.model.decoder.layers.{i}.self_attn.out_proj.weight"] = f"t{i}a_ow"
        weight_map_language[f"language_model.model.decoder.layers.{i}.encoder_attn.q_proj.weight"] = f"t{i}c_qw"
        weight_map_language[f"language_model.model.decoder.layers.{i}.encoder_attn.k_proj.weight"] = f"t{i}c_kw"
        weight_map_language[f"language_model.model.decoder.layers.{i}.encoder_attn.v_proj.weight"] = f"t{i}c_vw"
        weight_map_language[f"language_model.model.decoder.layers.{i}.encoder_attn.out_proj.weight"] = f"t{i}c_ow"
        weight_map_language[f"language_model.model.decoder.layers.{i}.fc1.weight"] = f"t{i}f_upw"
        weight_map_language[f"language_model.model.decoder.layers.{i}.fc2.weight"] = f"t{i}f_dnw"
        weight_map_language[f"language_model.model.decoder.layers.{i}.self_attn_layer_norm.weight"] = f"t{i}a_nw"
        weight_map_language[f"language_model.model.decoder.layers.{i}.encoder_attn_layer_norm.weight"] = f"t{i}c_nw"
        weight_map_language[f"language_model.model.decoder.layers.{i}.final_layer_norm.weight"] = f"t{i}f_nw"
        weight_map_language[f"language_model.model.decoder.layers.{i}.self_attn.q_proj.bias"] = f"t{i}a_qb"
        weight_map_language[f"language_model.model.decoder.layers.{i}.self_attn.k_proj.bias"] = f"t{i}a_kb"
        weight_map_language[f"language_model.model.decoder.layers.{i}.self_attn.v_proj.bias"] = f"t{i}a_vb"
        weight_map_language[f"language_model.model.decoder.layers.{i}.self_attn.out_proj.bias"] = f"t{i}a_ob"
        weight_map_language[f"language_model.model.decoder.layers.{i}.encoder_attn.q_proj.bias"] = f"t{i}c_qb"
        weight_map_language[f"language_model.model.decoder.layers.{i}.encoder_attn.k_proj.bias"] = f"t{i}c_kb"
        weight_map_language[f"language_model.model.decoder.layers.{i}.encoder_attn.v_proj.bias"] = f"t{i}c_vb"
        weight_map_language[f"language_model.model.decoder.layers.{i}.encoder_attn.out_proj.bias"] = f"t{i}c_ob"
        weight_map_language[f"language_model.model.decoder.layers.{i}.fc1.bias"] = f"t{i}f_upb"
        weight_map_language[f"language_model.model.decoder.layers.{i}.fc2.bias"] = f"t{i}f_dnb"
        weight_map_language[f"language_model.model.decoder.layers.{i}.self_attn_layer_norm.bias"] = f"t{i}a_nb"
        weight_map_language[f"language_model.model.decoder.layers.{i}.encoder_attn_layer_norm.bias"] = f"t{i}c_nb"
        weight_map_language[f"language_model.model.decoder.layers.{i}.final_layer_norm.bias"] = f"t{i}f_nb"

    for i in range(text_config.encoder_layers):
        weight_map_language[f"language_model.model.encoder.layers.{i}.self_attn.q_proj.weight"] = f"te{i}a_qw"
        weight_map_language[f"language_model.model.encoder.layers.{i}.self_attn.k_proj.weight"] = f"te{i}a_kw"
        weight_map_language[f"language_model.model.encoder.layers.{i}.self_attn.v_proj.weight"] = f"te{i}a_vw"
        weight_map_language[f"language_model.model.encoder.layers.{i}.self_attn.out_proj.weight"] = f"te{i}a_ow"
        weight_map_language[f"language_model.model.encoder.layers.{i}.fc1.weight"] = f"te{i}f_upw"
        weight_map_language[f"language_model.model.encoder.layers.{i}.fc2.weight"] = f"te{i}f_dnw"
        weight_map_language[f"language_model.model.encoder.layers.{i}.self_attn_layer_norm.weight"] = f"te{i}a_nw"
        weight_map_language[f"language_model.model.encoder.layers.{i}.final_layer_norm.weight"] = f"te{i}f_nw"
        weight_map_language[f"language_model.model.encoder.layers.{i}.self_attn.q_proj.bias"] = f"te{i}a_qb"
        weight_map_language[f"language_model.model.encoder.layers.{i}.self_attn.k_proj.bias"] = f"te{i}a_kb"
        weight_map_language[f"language_model.model.encoder.layers.{i}.self_attn.v_proj.bias"] = f"te{i}a_vb"
        weight_map_language[f"language_model.model.encoder.layers.{i}.self_attn.out_proj.bias"] = f"te{i}a_ob"
        weight_map_language[f"language_model.model.encoder.layers.{i}.fc1.bias"] = f"te{i}f_upb"
        weight_map_language[f"language_model.model.encoder.layers.{i}.fc2.bias"] = f"te{i}f_dnb"
        weight_map_language[f"language_model.model.encoder.layers.{i}.self_attn_layer_norm.bias"] = f"te{i}a_nb"
        weight_map_language[f"language_model.model.encoder.layers.{i}.final_layer_norm.bias"] = f"te{i}f_nb"

    weight_map_vision = {
        "image_projection": "v_pjw",
        "image_proj_norm.weight": "v_pn_w",
        "image_proj_norm.bias": "v_pn_b",
        "image_pos_embed.row_embeddings.weight": "v_pos_row_w",
        "image_pos_embed.column_embeddings.weight": "v_pos_col_w",
        "visual_temporal_embed.pos_idx_to_embed": "v_temp_w",
        "vision_tower.convs.0.proj.weight": "v_c0w",
        "vision_tower.convs.0.proj.bias": "v_c0b",
        "vision_tower.convs.0.norm.weight": "v_n0w",
        "vision_tower.convs.0.norm.bias": "v_n0b",
        "vision_tower.convs.1.proj.weight": "v_c1w",
        "vision_tower.convs.1.proj.bias": "v_c1b",
        "vision_tower.convs.1.norm.weight": "v_n1w",
        "vision_tower.convs.1.norm.bias": "v_n1b",
        "vision_tower.convs.2.proj.weight": "v_c2w",
        "vision_tower.convs.2.proj.bias": "v_c2b",
        "vision_tower.convs.2.norm.weight": "v_n2w",
        "vision_tower.convs.2.norm.bias": "v_n2b",
        "vision_tower.convs.3.proj.weight": "v_c3w",
        "vision_tower.convs.3.proj.bias": "v_c3b",
        "vision_tower.convs.3.norm.weight": "v_n3w",
        "vision_tower.convs.3.norm.bias": "v_n3b"
    }
    block_idx = 0
    for stage, depth in enumerate(vision_config.depths):
        for layer in range(depth):
            gguf_prefix = f"v{block_idx}"
            prefix = f"vision_tower.blocks.{stage}.{layer}"
            weight_map_vision[f"{prefix}.spatial_block.conv1.fn.dw.weight"] = f"{gguf_prefix}s_c1w"
            weight_map_vision[f"{prefix}.spatial_block.conv1.fn.dw.bias"] = f"{gguf_prefix}s_c1b"
            weight_map_vision[f"{prefix}.spatial_block.window_attn.norm.weight"] = f"{gguf_prefix}s_nw"
            weight_map_vision[f"{prefix}.spatial_block.window_attn.norm.bias"] = f"{gguf_prefix}s_nb"
            weight_map_vision[f"{prefix}.spatial_block.window_attn.fn.qkv.weight"] = f"{gguf_prefix}s_qkvw"
            weight_map_vision[f"{prefix}.spatial_block.window_attn.fn.qkv.bias"] = f"{gguf_prefix}s_qkvb"
            weight_map_vision[f"{prefix}.spatial_block.window_attn.fn.proj.weight"] = f"{gguf_prefix}s_pw"
            weight_map_vision[f"{prefix}.spatial_block.window_attn.fn.proj.bias"] = f"{gguf_prefix}s_pb"
            weight_map_vision[f"{prefix}.spatial_block.conv2.fn.dw.weight"] = f"{gguf_prefix}s_c2w"
            weight_map_vision[f"{prefix}.spatial_block.conv2.fn.dw.bias"] = f"{gguf_prefix}s_c2b"
            weight_map_vision[f"{prefix}.spatial_block.ffn.norm.weight"] = f"{gguf_prefix}s_fnw"
            weight_map_vision[f"{prefix}.spatial_block.ffn.norm.bias"] = f"{gguf_prefix}s_fnb"
            weight_map_vision[f"{prefix}.spatial_block.ffn.fn.net.fc1.weight"] = f"{gguf_prefix}s_f1w"
            weight_map_vision[f"{prefix}.spatial_block.ffn.fn.net.fc1.bias"] = f"{gguf_prefix}s_f1b"
            weight_map_vision[f"{prefix}.spatial_block.ffn.fn.net.fc2.weight"] = f"{gguf_prefix}s_f2w"
            weight_map_vision[f"{prefix}.spatial_block.ffn.fn.net.fc2.bias"] = f"{gguf_prefix}s_f2b"
            weight_map_vision[f"{prefix}.channel_block.conv1.fn.dw.weight"] = f"{gguf_prefix}c_c1w"
            weight_map_vision[f"{prefix}.channel_block.conv1.fn.dw.bias"] = f"{gguf_prefix}c_c1b"
            weight_map_vision[f"{prefix}.channel_block.channel_attn.norm.weight"] = f"{gguf_prefix}c_nw"
            weight_map_vision[f"{prefix}.channel_block.channel_attn.norm.bias"] = f"{gguf_prefix}c_nb"
            weight_map_vision[f"{prefix}.channel_block.channel_attn.fn.qkv.weight"] = f"{gguf_prefix}c_qkvw"
            weight_map_vision[f"{prefix}.channel_block.channel_attn.fn.qkv.bias"] = f"{gguf_prefix}c_qkvb"
            weight_map_vision[f"{prefix}.channel_block.channel_attn.fn.proj.weight"] = f"{gguf_prefix}c_pw"
            weight_map_vision[f"{prefix}.channel_block.channel_attn.fn.proj.bias"] = f"{gguf_prefix}c_pb"
            weight_map_vision[f"{prefix}.channel_block.conv2.fn.dw.weight"] = f"{gguf_prefix}c_c2w"
            weight_map_vision[f"{prefix}.channel_block.conv2.fn.dw.bias"] = f"{gguf_prefix}c_c2b"
            weight_map_vision[f"{prefix}.channel_block.ffn.norm.weight"] = f"{gguf_prefix}c_fnw"
            weight_map_vision[f"{prefix}.channel_block.ffn.norm.bias"] = f"{gguf_prefix}c_fnb"
            weight_map_vision[f"{prefix}.channel_block.ffn.fn.net.fc1.weight"] = f"{gguf_prefix}c_f1w"
            weight_map_vision[f"{prefix}.channel_block.ffn.fn.net.fc1.bias"] = f"{gguf_prefix}c_f1b"
            weight_map_vision[f"{prefix}.channel_block.ffn.fn.net.fc2.weight"] = f"{gguf_prefix}c_f2w"
            weight_map_vision[f"{prefix}.channel_block.ffn.fn.net.fc2.bias"] = f"{gguf_prefix}c_f2b"
            block_idx += 1

    # Combine mappings into a single dictionary
    tensor_mapping = {}
    tensor_mapping.update(weight_map_language)
    tensor_mapping.update(weight_map_vision)

    # Export tensor mapping to JSON
    with open(mapping_file, 'w') as f:
        json.dump(tensor_mapping, f, indent=4)
    logger.info(f"Tensor mapping exported to {mapping_file}")

    # Process and write tensors
    total_bytes = 0
    tensor_count = 0
    for name, tensor in state_dict.items():
        gguf_name = tensor_mapping.get(name)
        if not gguf_name:
            logger.error(f"Critical error: Tensor {name} (shape: {tensor.shape}) not mapped in tensor_mapping")
            exit(1)

        if len(gguf_name) >= 64:
            logger.error(f"Tensor name {gguf_name} too long ({len(gguf_name)})")
            exit(1)

        data = tensor.cpu().detach().numpy()
        if data.dtype != np.float16:
            logger.info(f"Converting {gguf_name} from {data.dtype} to float16")
            data = data.astype(np.float16)
        data = np.ascontiguousarray(data)
        byte_size = data.nbytes
        total_bytes += byte_size
        tensor_count += 1

        if gguf_name in ["v_pjw", "t_out_w"]:
            logger.info(f"{gguf_name} first 10 values: {data.flatten()[:10]}")
            logger.info(f"{gguf_name} last 10 values: {data.flatten()[-10:]}")

        gguf_writer.add_tensor(gguf_name, data)

    logger.info(f"Total tensors written: {tensor_count}, Total bytes: {total_bytes}")
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()

    file_size = os.path.getsize(outfile)
    logger.info(f"Converted model to GGUF: {outfile}, File size: {file_size} bytes")

    # Validation step
    if not validate_mapping(outfile, mapping_file):
        logger.error(f"Validation of {mapping_file} against {outfile} failed")
        exit(1)

if __name__ == "__main__":
    main()