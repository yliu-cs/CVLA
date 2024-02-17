import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.augmentation as K
from transformers import AutoModel


class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, nhead, dropout):
        super().__init__()

        self.embedding_dim, self.nhead = embedding_dim, nhead
        self.head_dim = int(embedding_dim // nhead)
        if self.head_dim * self.nhead != self.embedding_dim:
            raise ValueError(f"Embedding dimension {self.embedding_dim} is not multiple of the number of attention heads {self.nhead}")
        
        self.query = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.key = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.value = nn.Linear(self.embedding_dim, self.embedding_dim)

        self.drop = nn.Dropout(dropout)
    
    def transpose_for_scores(self, x):
        x = x.view(x.size()[:-1] + (self.nhead, self.head_dim))
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_state, attn_mask=None):
        query_layer = self.transpose_for_scores(self.query(hidden_state))
        key_layer = self.transpose_for_scores(self.key(hidden_state))
        value_layer = self.transpose_for_scores(self.value(hidden_state))

        attn_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attn_scores = attn_scores / math.sqrt(self.head_dim)
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.drop(attn_probs)

        context_layer = torch.matmul(attn_probs, value_layer).permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(context_layer.size()[:-2] + (self.embedding_dim, ))

        return (context_layer, attn_probs)
    

class ResidualLinear(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, hidden_state, input_tensor):
        hidden_state = self.drop(self.linear(hidden_state))
        hidden_state = self.layer_norm(hidden_state + input_tensor)
        return hidden_state


class Attention(nn.Module):
    def __init__(self, embedding_dim, nhead, dropout):
        super().__init__()
        self.self_attn = SelfAttention(embedding_dim, nhead, dropout)
        self.res_linear = ResidualLinear(embedding_dim, embedding_dim, dropout)
    
    def forward(self, input_tensor, attn_mask=None):
        attn_output, attn_probs = self.self_attn(input_tensor, attn_mask)
        attn_output = self.res_linear(attn_output, input_tensor)
        return (attn_output, attn_probs)
    

class ActLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.act_fn = nn.GELU()
    
    def forward(self, hidden_state):
        return self.act_fn(self.linear(hidden_state))


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, nhead, dropout):
        super().__init__()
        self.attn = Attention(embedding_dim, nhead, dropout)
        self.act_linear = ActLinear(embedding_dim, hidden_dim)
        self.res_linear = ResidualLinear(hidden_dim, embedding_dim, dropout)
    
    def forward(self, hidden_state, attn_mask=None):
        attn_output, layer_attn = self.attn(hidden_state, attn_mask)
        layer_output = self.res_linear(self.act_linear(attn_output), attn_output)
        return (layer_output, layer_attn)


class Encoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, nhead, num_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(embedding_dim, hidden_dim, nhead, dropout) for _ in range(num_layers)
        ])
    
    def forward(self, hidden_state, attn_mask=None):
        if attn_mask is not None:
            ext_attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
            ext_attn_mask = (1.0 - ext_attn_mask) * torch.finfo(ext_attn_mask.dtype).min
        else:
            ext_attn_mask = None
        
        all_hidden_state, all_attn = [hidden_state], []
        for layer in self.layers:
            hidden_state, layer_attn = layer(hidden_state, ext_attn_mask)
            all_hidden_state.append(hidden_state.detach())
            all_attn.append(layer_attn.detach())
        
        return (hidden_state, all_hidden_state, all_attn)


class Pooler(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, embedding_dim)
        self.tanh = nn.Tanh()
    
    def forward(self, hidden_state):
        return self.tanh(self.linear(hidden_state[:, 0, :]))
    

class VisualPatchEmbedding(nn.Module):
    def __init__(self, max_temporal_len, patch_size, hidden_dim) -> None:
        super().__init__()
        self.max_temporal_len = max_temporal_len
        self.patch_size = patch_size
        assert patch_size[1] == patch_size[2]
        self.proj = nn.Conv3d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.pos_embed_spatial = nn.Parameter(torch.zeros((1, int((224 // patch_size[1]) ** 2), hidden_dim)))
        self.pos_embed_temporal = nn.Parameter(torch.zeros((1, max_temporal_len, hidden_dim)))
    
    def forward(self, visual):
        embedding = self.proj(visual).flatten(2).transpose(1, 2)
        embedding = self.layer_norm(self.linear(embedding))
        pos_embed = self.pos_embed_spatial.repeat(
            1, self.max_temporal_len, 1
        ) + torch.repeat_interleave(
            self.pos_embed_temporal
            , int((224 // self.patch_size[1]) ** 2)
            , dim=1
        )
        pos_embed = pos_embed.expand(embedding.size(0), -1, -1)
        embedding = embedding + pos_embed
        return embedding


class AcousticPatchEmbedding(nn.Module):
    def __init__(self, max_acoustic_len, patch_size, hidden_dim):
        super().__init__()
        self.proj = nn.Conv2d(1, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros((1, max_acoustic_len, hidden_dim)))
    
    def forward(self, acoustic):
        embedding = self.proj(acoustic).flatten(2).transpose(1, 2)
        embedding = self.layer_norm(self.linear(embedding))
        pos_embed = self.pos_embed.expand(embedding.size(0), -1, -1)
        embedding = embedding + pos_embed
        return embedding


class Contrastive(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones([]) * 0.05)
    
    def forward(self, x, y, y_mask, x_mask_fused_y):
        batch_size = x.size(0)

        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        y_mask = F.normalize(y_mask, dim=-1)
        x_mask_fused_y = F.normalize(x_mask_fused_y, dim=-1)

        sim_x_y = torch.matmul(x, y.T) / self.temperature
        sim_x_y_mask = torch.matmul(x, y_mask.T) / self.temperature
        sim_x_x_mask_fused_y = torch.matmul(x, x_mask_fused_y.T) / self.temperature

        x_y_logsm = F.log_softmax(
            torch.cat([
                sim_x_y
                , sim_x_y_mask - torch.diag_embed(torch.diag(sim_x_y_mask) + 1e5)
                , sim_x_x_mask_fused_y - torch.diag_embed(torch.diag(sim_x_x_mask_fused_y) + 1e5)
            ], dim=1)
            , dim=1
        )[:, :batch_size]
        x_y_mask_logsm = F.log_softmax(
            torch.cat([
                sim_x_y - torch.diag_embed(torch.diag(sim_x_y) + 1e5)
                , sim_x_y_mask
                , sim_x_x_mask_fused_y - torch.diag_embed(torch.diag(sim_x_x_mask_fused_y) + 1e5)
            ], dim=1)
            , dim=1
        )[:, batch_size:batch_size * 2]
        x_x_mask_fused_y_mask_logsm = F.log_softmax(
            torch.cat([
                sim_x_y - torch.diag_embed(torch.diag(sim_x_y) + 1e5)
                , sim_x_y_mask - torch.diag_embed(torch.diag(sim_x_y_mask) + 1e5)
                , sim_x_x_mask_fused_y
            ], dim=1)
            , dim=1
        )[:, batch_size * 2:]
        x_diag = torch.diag(x_y_logsm) + torch.diag(x_y_mask_logsm) + torch.diag(x_x_mask_fused_y_mask_logsm)
        loss_x = -torch.mean(x_diag)

        y_x_logsm = F.log_softmax(
            torch.cat([
                sim_x_y
                , sim_x_y_mask
                , sim_x_x_mask_fused_y
            ], dim=1).T
            , dim=1
        ).view(-1, batch_size, batch_size)
        y_diag = y_x_logsm.diagonal(dim1=1, dim2=2)
        loss_y = -torch.mean(y_diag.mean(dim=1))

        loss = loss_x + loss_y
        return loss


class VideoAugmentation(nn.Module):
    def __init__(self, pretrain_visual_aug_p, pretrain_num_visual_aug):
        super().__init__()
        self.visual_augmentation = K.VideoSequential(
            K.RandomErasing(scale=(0.1, 0.2), p=pretrain_visual_aug_p)
            , K.ColorJiggle(0.2, 0.3, 0.2, 0.3, p=pretrain_visual_aug_p)
            , K.RandomAffine(360, p=pretrain_visual_aug_p)
            , K.RandomGrayscale(p=pretrain_visual_aug_p)
            , K.RandomGaussianBlur((3, 3), (0.1, 2.0), p=pretrain_visual_aug_p)
            , K.RandomGaussianNoise(p=pretrain_visual_aug_p)
            , data_format="BCTHW"
            , same_on_frame=True
            , random_apply=pretrain_num_visual_aug
        )
        self.acoustic_augmentation = K.RandomErasing(
            scale=(0.1, 0.2), value=0.0, p=1.0
        )
    
    def forward(self, visual, acoustic):
        with torch.no_grad():
            visual_aug = self.visual_augmentation(visual)
            acoustic_aug = self.acoustic_augmentation(acoustic)
        return visual_aug, acoustic_aug


class CVLA(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.visual_patch_emb = VisualPatchEmbedding(params["max_temporal_len"], params["visual_patch_size"], params["embedding_dim"])
        self.acoustic_patch_emb = AcousticPatchEmbedding(params["max_acoustic_len"], params["acoustic_patch_size"], params["embedding_dim"])
        self.video_cls_token = nn.Parameter(torch.zeros(1, 1, params["embedding_dim"]))
        self.video_encoder = Encoder(params["embedding_dim"], params["video_hidden_dim"], params["video_nhead"], params["video_num_layers"], params["dropout"])
        self.video_mlp = nn.Sequential(
            nn.Linear(params["embedding_dim"], params["embedding_dim"])
            , nn.LayerNorm(params["embedding_dim"])
        )
        self.video_pooler = Pooler(params["embedding_dim"])

        self.text_encoder = AutoModel.from_pretrained(params["text_encoder"])
        self.text_mlp = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, params["embedding_dim"])
            , nn.LayerNorm(params["embedding_dim"])
        )
        self.text_pooler = Pooler(params["embedding_dim"])

        self.encoder = Encoder(params["embedding_dim"], params["fusion_hidden_dim"], params["fusion_nhead"], params["fusion_num_layers"], params["dropout"])
        self.pooler = Pooler(params["embedding_dim"])
        self.cls_head = nn.Linear(params["embedding_dim"], params["num_labels"])
        
        self.video_aug = VideoAugmentation(params["pretrain_visual_aug_p"], params["pretrain_num_visual_aug"])
        self.contrastive = Contrastive()
    
    def forward(self, inputs, targets=None, pretrain=False):
        visual, acoustic, (title_input_ids, title_attn_mask), (comment_input_ids, comment_attn_mask) = inputs
        batch_size = visual.size(0)

        visual_embedding = self.visual_patch_emb(visual)
        acoustic_embedding = self.acoustic_patch_emb(acoustic)
        video_embedding = torch.cat([
                self.video_cls_token.repeat(batch_size, 1, 1)
                , visual_embedding
                , acoustic_embedding
            ], dim=1
        )
        video_attn_mask = torch.ones(video_embedding.size()[:2]).cuda()
        if not pretrain:
            if "visual" not in self.params["modal"]:
                video_attn_mask.index_fill_(1, torch.arange(1, visual_embedding.size(1) + 1).cuda(), 0.0)
            if "acoustic" not in self.params["modal"]:
                video_attn_mask.index_fill_(1, torch.arange(1 + visual_embedding.size(1), video_attn_mask.size(1)).cuda(), 0.0)
        if not pretrain:
            if "title" not in self.params["modal"]:
                title_attn_mask = torch.zeros_like(title_attn_mask).cuda()
                if "comment" in self.params["modal"]:
                    title_attn_mask.index_fill_(1, torch.zeros(1, dtype=torch.long).cuda(), 1.0)
            if "comment" not in self.params["modal"]:
                comment_attn_mask = torch.zeros_like(comment_attn_mask, dtype=torch.long).cuda()
        text_input_ids = torch.cat([title_input_ids, comment_input_ids], dim=-1)
        text_attn_mask = torch.cat([title_attn_mask, comment_attn_mask], dim=-1)

        video_last_hidden_state = self.video_mlp(self.video_encoder(video_embedding, video_attn_mask)[0])
        text_last_hidden_state = self.text_mlp(self.text_encoder(text_input_ids, attention_mask=text_attn_mask).last_hidden_state)

        joint_last_hidden_state = torch.cat([video_last_hidden_state, text_last_hidden_state], dim=1)
        joint_attn_mask = torch.cat([video_attn_mask, text_attn_mask], dim=1)

        if pretrain:
            video_cls_embedding = self.video_pooler(video_last_hidden_state)
            text_cls_embedding = self.text_pooler(text_last_hidden_state)

            # Noise Contrastive Estimation
            text_drop_last_hidden_state = self.text_mlp(self.text_encoder(text_input_ids, attention_mask=text_attn_mask).last_hidden_state)
            text_drop_cls_embedding = self.text_pooler(text_drop_last_hidden_state)
            visual_aug, acoustic_aug = self.video_aug(visual, acoustic)
            visual_aug_embedding = self.visual_patch_emb(visual_aug)
            acoustic_aug_embedding = self.acoustic_patch_emb(acoustic_aug)
            video_aug_embedding = torch.cat([
                    self.video_cls_token.repeat(batch_size, 1, 1)
                    , visual_aug_embedding
                    , acoustic_aug_embedding
                ], dim=1
            )
            video_aug_last_hidden_state = self.video_mlp(self.video_encoder(video_aug_embedding)[0])
            video_aug_cls_embedding = self.video_pooler(video_aug_last_hidden_state)
            video_aug_joint_cls_embedding = self.pooler(self.encoder(video_aug_last_hidden_state)[0])
            text_drop_joint_cls_embedding = self.pooler(self.encoder(text_drop_last_hidden_state)[0])
            loss = self.contrastive(video_cls_embedding, text_cls_embedding, text_drop_cls_embedding, video_aug_joint_cls_embedding) \
                + self.contrastive(text_cls_embedding, video_cls_embedding, video_aug_cls_embedding, text_drop_joint_cls_embedding)
            
            return loss
        else:
            # fine-tune
            last_hidden_state, _, attention = self.encoder(joint_last_hidden_state, joint_attn_mask)
            joint_cls_embedding = self.pooler(last_hidden_state)
            logits = self.cls_head(joint_cls_embedding)

            with torch.no_grad():
                attn = torch.stack(attention).mean(1).mean(0).mean(0)
            
            if targets is not None:
                loss = F.cross_entropy(logits, targets)
                return (loss, logits), (joint_cls_embedding, attn)
            else:
                return logits, (joint_cls_embedding, attn)