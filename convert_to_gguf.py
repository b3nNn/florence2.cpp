import os
import torch
import json
import gguf
from gguf import GGUFWriter, GGUFValueType

class Florence2Model:
    def __init__(self, dir_model, ftype, fname_out):
        self.dir_model = dir_model
        # Convert ftype string to integer
        ftype_map = {"f16": 1, "q8_0": 7}  # From gguf_writer.py: FTYPE_MOSTLY_F16 = 1, FTYPE_MOSTLY_Q8_0 = 7
        self.ftype = ftype_map.get(ftype, 1)  # Default to F16
        self.gguf_writer = GGUFWriter(fname_out, "florence2")

        # Load config
        config_path = os.path.join(dir_model, "config.json")
        with open(config_path, "r") as f:
            self.config = json.load(f)

        # Load weights
        model_path = os.path.join(dir_model, "pytorch_model.bin")
        self.state_dict = torch.load(model_path, map_location="cpu")

    def set_gguf_parameters(self):
        hparams = self.config
        vision_config = hparams.get("vision_config", {})
        language_config = hparams.get("language_config", {})

        self.gguf_writer.add_context_length(hparams.get("max_length", 1024))
        self.gguf_writer.add_embedding_length(vision_config.get("hidden_size", 1024))
        self.gguf_writer.add_block_count(hparams.get("decoder_layers", 12))
        self.gguf_writer.add_head_count(hparams.get("decoder_attention_heads", 16))
        self.gguf_writer.add_head_count_kv(hparams.get("num_kv_heads", 1))
        self.gguf_writer.add_layer_norm_eps(hparams.get("layer_norm_eps", 1e-5))
        self.gguf_writer.add_file_type(self.ftype)  # Now an integer (1 for F16)

        # Vision-specific metadata
        self.gguf_writer.add_key_value("vision_num_hidden_layers",
                                       vision_config.get("num_hidden_layers", 24),
                                       GGUFValueType.UINT32)
        self.gguf_writer.add_key_value("vision_num_attention_heads",
                                       vision_config.get("num_attention_heads", 16),
                                       GGUFValueType.UINT32)

    def write_tensors(self):
        for name, data_torch in self.state_dict.items():
            new_names = self.modify_tensors(data_torch, name)
            for new_name, data in new_names:
                print(f"Writing tensor: {new_name}, shape: {data.shape}")
                self.gguf_writer.add_tensor(new_name, data.numpy())

    def modify_tensors(self, data_torch, name, bid=None):
        # Vision Tower
        if name.startswith("vision_tower.convs"):
            idx = name.split(".")[2]
            if "proj.weight" in name:
                new_name = f"v.cn{idx}.w"
            elif "proj.bias" in name:
                new_name = f"v.cn{idx}.b"
            elif "norm.weight" in name:
                new_name = f"v.cn{idx}.n.w"
            elif "norm.bias" in name:
                new_name = f"v.cn{idx}.n.b"
            else:
                new_name = f"v.cn{idx}.{name.split('.')[-1][:5]}"
            return [(new_name, data_torch)]
        elif name.startswith("vision_tower"):
            parts = name.split(".")
            if "convs" in name:
                idx = parts[2]
                if "proj.weight" in name:
                    new_name = f"v.cn{idx}.w"
                elif "proj.bias" in name:
                    new_name = f"v.cn{idx}.b"
                elif "norm.weight" in name:
                    new_name = f"v.cn{idx}.n.w"
                elif "norm.bias" in name:
                    new_name = f"v.cn{idx}.n.b"
                else:
                    new_name = f"v.cn{idx}.{parts[-1][:5]}"
            elif "stage" in name:
                stage, block = parts[1].replace("stage", ""), parts[2].replace("block", "")
                blk_id = f"{stage}{block.zfill(2)}"
                if "spatial_block" in name:
                    base_name = f"v.b{blk_id}.s"
                    if "fn.dw.weight" in name:
                        new_name = f"{base_name}.d.w"
                    elif "fn.net.0.weight" in name:
                        new_name = f"{base_name}.n0.w"
                    elif "fn.net.2.weight" in name:
                        new_name = f"{base_name}.n2.w"
                    elif "fn.norm1.weight" in name:
                        new_name = f"{base_name}.n1.w"
                    elif "fn.norm2.weight" in name:
                        new_name = f"{base_name}.n2.w"
                    elif "fn.dw.bias" in name:
                        new_name = f"{base_name}.d.b"
                    elif "fn.net.0.bias" in name:
                        new_name = f"{base_name}.n0.b"
                    elif "fn.net.2.bias" in name:
                        new_name = f"{base_name}.n2.b"
                    elif "fn.norm1.bias" in name:
                        new_name = f"{base_name}.n1.b"
                    elif "fn.norm2.bias" in name:
                        new_name = f"{base_name}.n2.b"
                    elif "weight" in name:
                        new_name = f"{base_name}.w"
                    elif "bias" in name:
                        new_name = f"{base_name}.b"
                    else:
                        new_name = f"{base_name}.{parts[-1][:5]}"
                elif "channel_block" in name:
                    base_name = f"v.b{blk_id}.c"
                    if "fn.dw.weight" in name:
                        new_name = f"{base_name}.d.w"
                    elif "fn.net.0.weight" in name:
                        new_name = f"{base_name}.n0.w"
                    elif "fn.net.2.weight" in name:
                        new_name = f"{base_name}.n2.w"
                    elif "fn.norm1.weight" in name:
                        new_name = f"{base_name}.n1.w"
                    elif "fn.norm2.weight" in name:
                        new_name = f"{base_name}.n2.w"
                    elif "fn.dw.bias" in name:
                        new_name = f"{base_name}.d.b"
                    elif "fn.net.0.bias" in name:
                        new_name = f"{base_name}.n0.b"
                    elif "fn.net.2.bias" in name:
                        new_name = f"{base_name}.n2.b"
                    elif "fn.norm1.bias" in name:
                        new_name = f"{base_name}.n1.b"
                    elif "fn.norm2.bias" in name:
                        new_name = f"{base_name}.n2.b"
                    elif "weight" in name:
                        new_name = f"{base_name}.w"
                    elif "bias" in name:
                        new_name = f"{base_name}.b"
                    else:
                        new_name = f"{base_name}.{parts[-1][:5]}"
                else:
                    new_name = name[:63]
            else:
                new_name = name[:63]
            return [(new_name, data_torch)]

        # Language Model
        elif name.startswith("language_model.model.shared"):
            return [("t_emb.w", data_torch)]
        elif name.startswith("language_model.model.encoder.embed_positions"):
            return [("e_pos.w", data_torch)]
        elif name == "language_model.model.encoder.embed_tokens.weight":
            return [("e_tok.w", data_torch)]
        elif name == "language_model.model.encoder.layernorm_embedding.weight":
            return [("e_lne.w", data_torch)]
        elif name == "language_model.model.encoder.layernorm_embedding.bias":
            return [("e_lne.b", data_torch)]
        elif name.startswith("language_model.model.decoder.embed_positions"):
            return [("d_pos.w", data_torch)]
        elif name == "language_model.model.decoder.embed_tokens.weight":
            return [("d_tok.w", data_torch)]
        elif name.startswith("language_model.model.encoder.layers"):
            layer_id = name.split(".")[4]
            if "self_attn.q.weight" in name:
                new_name = f"e.l{layer_id}.s.q.w"
            elif "self_attn.k.weight" in name:
                new_name = f"e.l{layer_id}.s.k.w"
            elif "self_attn.v.weight" in name:
                new_name = f"e.l{layer_id}.s.v.w"
            elif "self_attn.out.weight" in name:
                new_name = f"e.l{layer_id}.s.o.w"
            elif "self_attn.out_proj.weight" in name:
                new_name = f"e.l{layer_id}.s.p.w"
            elif "fc1.weight" in name:
                new_name = f"e.l{layer_id}.f1.w"
            elif "fc2.weight" in name:
                new_name = f"e.l{layer_id}.f2.w"
            elif "self_attn_layer_norm.weight" in name:
                new_name = f"e.l{layer_id}.s.n.w"
            elif "self_attn_layer_norm.bias" in name:
                new_name = f"e.l{layer_id}.s.n.b"
            elif "final_layer_norm.weight" in name:
                new_name = f"e.l{layer_id}.f.n.w"
            elif "final_layer_norm.bias" in name:
                new_name = f"e.l{layer_id}.f.n.b"
            elif "self_attn.q.bias" in name:
                new_name = f"e.l{layer_id}.s.q.b"
            elif "self_attn.k.bias" in name:
                new_name = f"e.l{layer_id}.s.k.b"
            elif "self_attn.v.bias" in name:
                new_name = f"e.l{layer_id}.s.v.b"
            elif "self_attn.out.bias" in name:
                new_name = f"e.l{layer_id}.s.o.b"
            elif "self_attn.out_proj.bias" in name:
                new_name = f"e.l{layer_id}.s.p.b"
            elif "fc1.bias" in name:
                new_name = f"e.l{layer_id}.f1.b"
            elif "fc2.bias" in name:
                new_name = f"e.l{layer_id}.f2.b"
            elif "weight" in name:
                suffix = name.split(".")[-2][:2]
                new_name = f"e.l{layer_id}.w.{suffix}"
            elif "bias" in name:
                suffix = name.split(".")[-2][:2]
                new_name = f"e.l{layer_id}.b.{suffix}"
            else:
                new_name = f"e.l{layer_id}.{name.split('.')[-1][:5]}"
            return [(new_name, data_torch)]
        elif name.startswith("language_model.model.decoder.layers"):
            layer_id = name.split(".")[4]
            if "self_attn.q.weight" in name:
                new_name = f"d.l{layer_id}.s.q.w"
            elif "self_attn.k.weight" in name:
                new_name = f"d.l{layer_id}.s.k.w"
            elif "self_attn.v.weight" in name:
                new_name = f"d.l{layer_id}.s.v.w"
            elif "self_attn.q_proj.weight" in name:
                new_name = f"d.l{layer_id}.s.qp.w"
            elif "self_attn.k_proj.weight" in name:
                new_name = f"d.l{layer_id}.s.kp.w"
            elif "self_attn.v_proj.weight" in name:
                new_name = f"d.l{layer_id}.s.vp.w"
            elif "self_attn.out.weight" in name:
                new_name = f"d.l{layer_id}.s.o.w"
            elif "self_attn.out_proj.weight" in name:
                new_name = f"d.l{layer_id}.s.p.w"
            elif "encoder_attn.q.weight" in name:
                new_name = f"d.l{layer_id}.e.q.w"
            elif "encoder_attn.k.weight" in name:
                new_name = f"d.l{layer_id}.e.k.w"
            elif "encoder_attn.v.weight" in name:
                new_name = f"d.l{layer_id}.e.v.w"
            elif "encoder_attn.q_proj.weight" in name:
                new_name = f"d.l{layer_id}.e.qp.w"
            elif "encoder_attn.k_proj.weight" in name:
                new_name = f"d.l{layer_id}.e.kp.w"
            elif "encoder_attn.v_proj.weight" in name:
                new_name = f"d.l{layer_id}.e.vp.w"
            elif "encoder_attn.out.weight" in name:
                new_name = f"d.l{layer_id}.e.o.w"
            elif "encoder_attn.out_proj.weight" in name:
                new_name = f"d.l{layer_id}.e.p.w"
            elif "self_attn_layer_norm.weight" in name:
                new_name = f"d.l{layer_id}.s.n.w"
            elif "self_attn_layer_norm.bias" in name:
                new_name = f"d.l{layer_id}.s.n.b"
            elif "final_layer_norm.weight" in name:
                new_name = f"d.l{layer_id}.f.n.w"
            elif "final_layer_norm.bias" in name:
                new_name = f"d.l{layer_id}.f.n.b"
            elif "self_attn.q.bias" in name:
                new_name = f"d.l{layer_id}.s.q.b"
            elif "self_attn.k.bias" in name:
                new_name = f"d.l{layer_id}.s.k.b"
            elif "self_attn.v.bias" in name:
                new_name = f"d.l{layer_id}.s.v.b"
            elif "self_attn.q_proj.bias" in name:
                new_name = f"d.l{layer_id}.s.qp.b"
            elif "self_attn.k_proj.bias" in name:
                new_name = f"d.l{layer_id}.s.kp.b"
            elif "self_attn.v_proj.bias" in name:
                new_name = f"d.l{layer_id}.s.vp.b"
            elif "self_attn.out.bias" in name:
                new_name = f"d.l{layer_id}.s.o.b"
            elif "self_attn.out_proj.bias" in name:
                new_name = f"d.l{layer_id}.s.p.b"
            elif "encoder_attn.q.bias" in name:
                new_name = f"d.l{layer_id}.e.q.b"
            elif "encoder_attn.k.bias" in name:
                new_name = f"d.l{layer_id}.e.k.b"
            elif "encoder_attn.v.bias" in name:
                new_name = f"d.l{layer_id}.e.v.b"
            elif "encoder_attn.q_proj.bias" in name:
                new_name = f"d.l{layer_id}.e.qp.b"
            elif "encoder_attn.k_proj.bias" in name:
                new_name = f"d.l{layer_id}.e.kp.b"
            elif "encoder_attn.v_proj.bias" in name:
                new_name = f"d.l{layer_id}.e.vp.b"
            elif "encoder_attn.out.bias" in name:
                new_name = f"d.l{layer_id}.e.o.b"
            elif "encoder_attn.out_proj.bias" in name:
                new_name = f"d.l{layer_id}.e.p.b"
            elif "fc1.bias" in name:
                new_name = f"d.l{layer_id}.f1.b"
            elif "fc2.bias" in name:
                new_name = f"d.l{layer_id}.f2.b"
            elif "fc1.weight" in name:
                new_name = f"d.l{layer_id}.f1.w"
            elif "fc2.weight" in name:
                new_name = f"d.l{layer_id}.f2.w"
            elif "weight" in name:
                suffix = name.split(".")[-2][:2]
                new_name = f"d.l{layer_id}.w.{suffix}"
            elif "bias" in name:
                suffix = name.split(".")[-2][:2]
                new_name = f"d.l{layer_id}.b.{suffix}"
            else:
                new_name = f"d.l{layer_id}.{name.split('.')[-1][:5]}"
            return [(new_name, data_torch)]
        elif name == "language_model.lm_head.weight":
            return [("out.w", data_torch)]
        elif name == "language_model.final_logits_bias":
            return [("out.b", data_torch)]

        # Glue Layers
        elif name == "image_projection":
            return [("i_prj.w", data_torch)]
        elif name.startswith("image_proj_norm"):
            if "weight" in name:
                return [("i_prj.n.w", data_torch)]
            elif "bias" in name:
                return [("i_prj.n.b", data_torch)]
        elif name == "image_pos_embed":
            return [("i_pos.w", data_torch)]
        elif name == "image_pos_embed.column_embeddings.weight":
            return [("i_pos.c.w", data_torch)]
        elif name.startswith("visual_temporal_embed"):
            return [("v_tmp.w", data_torch)]

        # Fallback
        new_name = name[:63]  # Always truncate to 63 chars max
        return [(new_name, data_torch)]

    def write(self):
        self.set_gguf_parameters()
        self.write_tensors()
        self.gguf_writer.write_header_to_file()
        self.gguf_writer.write_kv_data_to_file()
        self.gguf_writer.write_tensors_to_file()
        self.gguf_writer.close()

def convert_hf_to_gguf(dir_model, outfile, outtype):
    model_instance = Florence2Model(dir_model=dir_model, ftype=outtype, fname_out=outfile)
    model_instance.write()
    print(f"Model successfully exported to '{outfile}'")

# Update gguf.py constants
if "FLORENCE2" not in gguf.MODEL_ARCH.__members__:
    gguf.MODEL_ARCH_NAMES[gguf.MODEL_ARCH.FLORENCE2] = "florence2"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert HF model to GGUF")
    parser.add_argument("dir_model", help="Directory containing HF model")
    parser.add_argument("--outfile", help="Output GGUF file path")
    parser.add_argument("--outtype", default="f16", help="Output type (f16, q8_0, etc.)")
    args = parser.parse_args()

    convert_hf_to_gguf(args.dir_model, args.outfile, args.outtype)
