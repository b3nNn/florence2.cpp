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
            new_name = name.replace(f"vision_tower.convs.{idx}", f"vision.conv{idx}")
            return [(new_name, data_torch)]

        elif name.startswith("vision_tower.keys"):
            parts = name.split(".")
            stage, block = parts[2], parts[3]
            blk_id = f"{stage}{block}"
            if "spatial_block" in name:
                new_name = name.replace(f"vision_tower.keys.{stage}.{block}.spatial_block", f"vision.blk{blk_id}.spatial")
            elif "channel_block" in name:
                new_name = name.replace(f"vision_tower.keys.{stage}.{block}.channel_block", f"vision.blk{blk_id}.channel")
            new_name = new_name.replace("fn.dw", "dw").replace("fn.net.", "").replace("fn.", "")
            return [(new_name, data_torch)]

        # Language Model
        elif name.startswith("language_model.model.shared"):
            return [("tok_embeddings.weight", data_torch)]
        elif name.startswith("language_model.model.encoder.embed_positions"):
            return [("enc_pos_embeddings.weight", data_torch)]
        elif name.startswith("language_model.model.decoder.embed_positions"):
            return [("dec_pos_embeddings.weight", data_torch)]
        elif name.startswith("language_model.model.encoder.layers"):
            layer_id = name.split(".")[4]
            new_name = name.replace(f"language_model.model.encoder.layers.{layer_id}", f"enc.blk{layer_id}")
            return [(new_name, data_torch)]
        elif name.startswith("language_model.model.decoder.layers"):
            layer_id = name.split(".")[4]
            new_name = name.replace(f"language_model.model.decoder.layers.{layer_id}", f"dec.blk{layer_id}")
            return [(new_name, data_torch)]
        elif name == "language_model.lm_head.weight":
            return [("output.weight", data_torch)]
        elif name == "language_model.final_logits_bias":
            return [("output.bias", data_torch)]

        # Glue Layers
        elif name == "image_projection":
            return [("img_proj.weight", data_torch)]
        elif name.startswith("image_proj_norm"):
            return [(name.replace("image_proj_norm", "img_proj_norm"), data_torch)]
        elif name.startswith("image_pos_embed"):
            return [(name.replace("image_pos_embed", "img_pos_embed"), data_torch)]
        elif name.startswith("visual_temporal_embed"):
            return [(name.replace("visual_temporal_embed", "vis_temp_embed"), data_torch)]

        return [(name, data_torch)]

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
