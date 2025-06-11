import numpy as np
from torch.nn import functional as F

from .utils import *
from model.transformer_modules.TransformerEncoders import *  
from transformers import BertTokenizer, BertModel
import torchvision.models as models
import transformers
import requests
import clip




# my model begining

class InputUnitLinguistic_Bert(nn.Module):
    def __init__(self, vocab_size, layers=2, wordvec_dim=300, module_dim=512):
        super(InputUnitLinguistic_Bert, self).__init__()

        self.activ = nn.GELU()
        self.encoder_embed = nn.Embedding(vocab_size, wordvec_dim)
        self.proj_l = nn.Sequential(
            nn.Linear(wordvec_dim, module_dim),
            self.activ,
            nn.Dropout(p=0.1),
        )
        self.proj_bert = nn.Sequential(
            nn.Linear(768, module_dim),
            self.activ,
            nn.Dropout(p=0.1),
        )
        self.TransformerEncoder_text = TransformerEncoder(embed_dim=module_dim, pos_flag='sincos', pos_dropout=0.1,
                                                          num_heads=8, attn_dropout=0.1, res_dropout=0.1,
                                                          activ_dropout=0.1, activation='gelu', num_layers=layers)
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.avgpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.embed_diff = nn.Linear(self.resnet.fc.out_features, module_dim)  # diff_MLP --> (1000, 384)
        self.frame_diff_norm = nn.LayerNorm(module_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        self.BertEncoder = BertModel.from_pretrained("pretrained_berts/bert-base-cased")
        self.module_dim = module_dim

    def forward(self, text, text_input_bert, text_mask_bert, text_ids_bert, text_len):
        with torch.no_grad():
            text_embedding_bert = self.BertEncoder(input_ids=text_input_bert, attention_mask=text_mask_bert,
                                             token_type_ids=text_ids_bert)
        text_embedding = self.proj_bert(text_embedding_bert[0].permute(1, 0, 2))
        text_embedding = self.TransformerEncoder_text(text_embedding, None,
                                                           text_len.squeeze())
        return text_embedding


class MCL_ID(nn.Module):
    def __init__(self, motion_dim, appearance_dim, module_dim, word_dim, vocab, question_type):
        super(MCL_ID, self).__init__()

        self.question_type = question_type


        encoder_vocab_size = len(vocab['question_token_to_idx'])

        self.num_classes = len(vocab['answer_token_to_idx'])
        self.linguistic_input_unit = InputUnitLinguistic_Bert(vocab_size=encoder_vocab_size, layers=4,
                                                              wordvec_dim=word_dim, module_dim=module_dim)

        self.embed_patch = torch.nn.Linear(768, module_dim)
        self.module_dim = module_dim
        self.mm_decode = nn.Linear(module_dim * 2, module_dim, bias=True)
        hidden_dim = 768
        self.appearance_feat_proj = nn.Linear(1024, module_dim)
        self.proj_layer = nn.Linear(module_dim, hidden_dim)
        self.embed_ln = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, module_dim)
        self.gate_proj = nn.Linear(3 * module_dim, module_dim)
        config = transformers.GPT2Config(
            vocab_size=1,
            n_embd=hidden_dim
        )
        #self.transformer = GPT2Model(config)
        self.learnable_token = nn.Parameter(torch.zeros(1, 32, module_dim))
        
        
        self.ATransformer = TransformerEncoder_localmask(embed_dim=module_dim, pos_flag='learned', pos_dropout=0.0, num_heads=8,
                                               attn_dropout=0.0, res_dropout=0.1, activ_dropout=0.1,
                                               activation='gelu', num_layers=2)
        self.DATransformer = TransformerEncoder(embed_dim=module_dim, pos_flag='learned', pos_dropout=0.0, num_heads=8,
                                                attn_dropout=0.0, res_dropout=0.1, activ_dropout=0.1,
                                                activation='gelu', num_layers=2)
        self.LDTransformer = TransformerEncoder(embed_dim=module_dim, pos_flag='learned', pos_dropout=0.0, num_heads=8,
                                                attn_dropout=0.0, res_dropout=0.1, activ_dropout=0.1,
                                                activation='gelu', num_layers=2)
        self.fuse_Transformer = TransformerEncoder(embed_dim=module_dim*2, pos_flag='learned', pos_dropout=0.0, num_heads=8,
                                                attn_dropout=0.0, res_dropout=0.1, activ_dropout=0.1,
                                                activation='gelu', num_layers=2)
        self.tao = 0.1 #0.1
        self.space_cl_loss = NTXentLoss(module_dim, self.tao)
        self.space_cl_loss_diff = NTXentLoss(module_dim, self.tao)
        self.space_cl_loss_visual = NTXentLoss(module_dim, self.tao)

        self.mm_cl_loss = NTXentLoss_neg(module_dim, self.tao)
        self.mm_cl_fine_loss = NTXentLoss1_neg(module_dim, self.tao)


        self.classifier = nn.Sequential(nn.Dropout(0.15),
                                        nn.Linear(module_dim * 48, module_dim * 2),
                                        nn.ELU(),
                                        nn.BatchNorm1d(module_dim * 2),
                                        nn.Dropout(0.15),
                                        nn.Linear(module_dim * 2, 4))
                                        
        self.mm_decode=nn.Linear(module_dim*2, module_dim,bias=True)


    def forward(self, video_ids, question_ids,answer_idx,clip_model,ans_candidates, ans_candidates_len, batch_clips_data, video_motion_feat, \
                question_input_bert, question_mask_bert, question_ids_bert, answer_input_bert, answer_mask_bert,
                answer_ids_bert, ans_candidates_input_bert, ans_candidates_mask_bert, ans_candidates_ids_bert,
                question_bert_len, answer_bert_len, ans_candidates_bert_len, \
                question, answer, question_len, answer_len, appearance_dict, motion_dict, ques_type):

        # bs,8,4,3,224,224
        batch_size, num_clips, num_frames, c, h, w = batch_clips_data.shape  # torch.Size([128, 8, 4, 3, 224, 224])
        batch_clips_data = batch_clips_data.view(batch_size, num_clips * num_frames, c, h, w).contiguous()  # bs,32,3,224,224
        batch_size, f, c, h, w = batch_clips_data.shape
        frame_diff = batch_clips_data[:, 1:] - batch_clips_data[:, :-1]  # (b, 31, c, h, w)

        zero_diff = torch.zeros_like(batch_clips_data[:, :1])  # (b, 1, c, h, w)
        frame_diff = torch.cat([zero_diff, frame_diff], dim=1)  # (b, 32, c, h, w)
        frame_diff = frame_diff.view(-1, c, h, w)

        batch_clips_data = batch_clips_data.view(batch_size * f, c, h, w)
        device = batch_clips_data.device

        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).reshape(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).reshape(1, 3, 1, 1)
        batch_clips_data = batch_clips_data.to(dtype=torch.float32)
        batch_clips_data = (batch_clips_data / 255.0 - mean) / std
        batch_clips_data = batch_clips_data.cuda()

        frame_diff = frame_diff.to(dtype=torch.float32)
        frame_diff = (frame_diff / 255.0 - mean) / std
        frame_diff = frame_diff.cuda()

        with torch.no_grad():
            obs_embeddings = clip_model.encode_image(batch_clips_data)
            obs_embeddings_diff = clip_model.encode_image(frame_diff)
        video_appearance_feat = obs_embeddings.view(batch_size, f, -1).permute(1, 0,2)
        video_appearance_feat_diff = obs_embeddings_diff.view(batch_size, f, -1)
        video_appearance_feat_diff = video_appearance_feat_diff.permute(1, 0, 2)
        visual = self.ATransformer(video_appearance_feat, video_appearance_feat)
        diff_visual = self.DATransformer(video_appearance_feat_diff, visual)
        visual = visual.permute(1, 0, 2)
        diff_visual = diff_visual.permute(1, 0, 2)
        loss = []
        sp_cl_loss = self.space_cl_loss(visual, diff_visual)
        loss.append(sp_cl_loss)
        batch_size = question.size(0)
        # get image, word, and sentence embeddings#clip-text
        question_embedding_bert = self.linguistic_input_unit(question, question_input_bert,
                                                             question_mask_bert,question_ids_bert, 
                                                             question_bert_len)  # shape:length x batch_size x channel_size

        candiates_embedding_bert = []
        for i in range(4):
            tem_embedding_bert = self.linguistic_input_unit(ans_candidates[:, i, :],
                                                            ans_candidates_input_bert[:, i, :], \
                                                            ans_candidates_mask_bert[:, i, :],ans_candidates_ids_bert[:,i,:], 
                                                            ans_candidates_bert_len[:, i, :])

            candiates_embedding_bert.append(tem_embedding_bert)

        candiates_embedding_bert = torch.stack(candiates_embedding_bert,2)
        question_features = question_embedding_bert.permute(1, 0, 2)
        space_cl_loss_diff = self.space_cl_loss_diff(question_features, diff_visual)
        space_cl_loss_visual = self.space_cl_loss_visual(question_features, visual)
        loss.append(space_cl_loss_diff)
        loss.append(space_cl_loss_visual)
        fusion_gate = torch.sigmoid(self.gate_proj(torch.cat([question_features, visual, diff_visual], dim=-1)))
        fused_visual = fusion_gate * diff_visual + (1 - fusion_gate) * visual
        fusion_mm_fea=torch.cat((fused_visual, question_features),-1).permute(1, 0, 2)
        fusion_mm_fea=self.fuse_Transformer(fusion_mm_fea,fusion_mm_fea)
        fusion_mm_fea=self.mm_decode(fusion_mm_fea)
        if self.training:
            answer_index = torch.nn.functional.one_hot(answer_idx, num_classes=4)
            ans_mask = (answer_index == 1)
            true_embedding_bert = torch.masked_select(candiates_embedding_bert, ans_mask.unsqueeze(0).unsqueeze(-1))
            true_embedding_bert = true_embedding_bert.reshape(question_embedding_bert.shape[0], batch_size, \
                                                              question_embedding_bert.shape[2]).contiguous()

            neg_mask = (answer_index != 1)
            negative_embedding_bert = torch.masked_select(candiates_embedding_bert, neg_mask.unsqueeze(0).unsqueeze(-1))
            negative_embedding_bert = negative_embedding_bert.reshape(question_embedding_bert.shape[0], batch_size, 3, \
                                                                      question_embedding_bert.shape[2]).contiguous()
            padding1 = torch.zeros((video_appearance_feat.shape[0] - true_embedding_bert.shape[0], true_embedding_bert.shape[1], 3,
                                    true_embedding_bert.shape[2])).cuda()
            negative_embedding_bert = torch.cat((negative_embedding_bert, padding1), 0)

            padding = torch.zeros((video_appearance_feat.shape[0] - true_embedding_bert.shape[0], true_embedding_bert.shape[1],
                                   true_embedding_bert.shape[2])).cuda()
            true_embedding_bert = torch.cat((true_embedding_bert, padding), 0)
            #bs,32,512  bs,32,512  bs,32,3,512
            mm_cl_loss = self.mm_cl_loss(fusion_mm_fea.permute(1, 0, 2), true_embedding_bert.permute(1, 0, 2),
                                         negative_embedding_bert.permute(1, 0, 2, 3))
            mm_cl_fine_loss = self.mm_cl_fine_loss(fusion_mm_fea.permute(1, 0, 2), true_embedding_bert.permute(1, 0, 2),
                                                   negative_embedding_bert.permute(1, 0, 2, 3))
            loss.append(mm_cl_loss)
            loss.append(mm_cl_fine_loss)

            score = []
            for i in range(4):
                can_embedding_bert = candiates_embedding_bert[:, :, i, :]
                padding = torch.zeros((video_appearance_feat.shape[0] - can_embedding_bert.shape[0], can_embedding_bert.shape[1], \
                                       can_embedding_bert.shape[2])).cuda()

                can_embedding_bert = torch.cat((can_embedding_bert, padding), 0)
                can_embedding_bert_1 = can_embedding_bert.permute(1, 0, 2).reshape(batch_size, -1).contiguous()

                fusion_mm_fea_1 = fusion_mm_fea.permute(1, 0, 2).reshape(batch_size, -1).contiguous()

                tem = torch.sum(can_embedding_bert_1 * fusion_mm_fea_1, -1)
                score.append(tem)
            score = torch.stack(score, -1)
            out = score  # F.softmax(score,-1)

        else:
            score = []
            for i in range(4):
                can_embedding_bert = candiates_embedding_bert[:, :, i, :]
                padding = torch.zeros((video_appearance_feat.shape[0] - can_embedding_bert.shape[0], can_embedding_bert.shape[1], \
                                       can_embedding_bert.shape[2])).cuda()
                can_embedding_bert = torch.cat((can_embedding_bert, padding), 0)
                can_embedding_bert_1 = can_embedding_bert.permute(1, 0, 2).reshape(batch_size, -1).contiguous()
                fusion_mm_fea_1 =fusion_mm_fea.permute(1, 0, 2).reshape(batch_size, -1).contiguous()

                tem = torch.sum(can_embedding_bert_1 * fusion_mm_fea_1, -1)
                score.append(tem)
            score = torch.stack(score, -1)
            out = score  # F.softmax(score,-1)

        return loss, out



class NTXentLoss(torch.nn.Module):

    def __init__(self, module_dim, temperature):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.fc = nn.Sequential(
            nn.Linear(module_dim, module_dim // 4, bias=True),
            nn.GELU(),
            nn.Linear(module_dim // 4, module_dim // 8, bias=True)
        )

    def forward(self, zis, zjs):
        shape = zis.shape
        zis = self.fc(zis)
        zjs = self.fc(zjs)
        zis = zis.reshape(shape[0], -1)
        zjs = zjs.reshape(shape[0], -1)

        batch_size = shape[0]

        zis1 = F.normalize(zis, p=2, dim=-1)
        zjs1 = F.normalize(zjs, p=2, dim=-1)

        similarity_matrix = torch.matmul(zis1, zjs1.permute(1, 0))

        shape = similarity_matrix.shape
        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix)
        positives = l_pos  # .view(shape[0],self.batch_size, 1)
        positives = positives / self.temperature

        diag = np.eye(batch_size)
        mask = torch.from_numpy((diag))
        mask = (1 - mask).type(torch.bool).cuda()
        negatives = similarity_matrix[mask].view(batch_size, batch_size - 1)
        negatives = negatives / self.temperature

        loss = -torch.log((torch.exp(positives)) / (torch.exp(positives) + torch.sum(torch.exp(negatives), -1, True)))
        return loss.sum() / (batch_size)


class NTXentLoss_neg(torch.nn.Module):

    def __init__(self, module_dim, temperature):
        super(NTXentLoss_neg, self).__init__()
        self.temperature = temperature
        self.fc = nn.Sequential(
            nn.Linear(module_dim, module_dim // 4, bias=True),
            nn.GELU(),
            nn.Linear(module_dim // 4, module_dim // 8, bias=True)
        )

    def forward(self, zis, zjs, neg):
        shape = zis.shape

        zis = self.fc(zis)
        zjs = self.fc(zjs)
        zis = zis.reshape(shape[0], -1)
        zjs = zjs.reshape(shape[0], -1)

        n1 = self.fc(neg[:, :, 0, :]).reshape(shape[0], -1)
        n2 = self.fc(neg[:, :, 1, :]).reshape(shape[0], -1)
        n3 = self.fc(neg[:, :, 2, :]).reshape(shape[0], -1)

        batch_size = shape[0]

        zis1 = F.normalize(zis, p=2, dim=-1)
        zjs1 = F.normalize(zjs, p=2, dim=-1)

        n1 = F.normalize(n1, p=2, dim=-1)
        n2 = F.normalize(n2, p=2, dim=-1)
        n3 = F.normalize(n3, p=2, dim=-1)

        similarity_matrix = torch.matmul(zis1, zjs1.permute(1, 0))

        neg1 = torch.sum(zis1 * n1, -1) / self.temperature
        neg2 = torch.sum(zis1 * n2, -1) / self.temperature
        neg3 = torch.sum(zis1 * n3, -1) / self.temperature

        shape = similarity_matrix.shape
        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix)
        positives = l_pos  # .view(shape[0],self.batch_size, 1)
        positives = positives / self.temperature

        diag = np.eye(batch_size)
        mask = torch.from_numpy((diag))
        mask = (1 - mask).type(torch.bool).cuda()
        negatives = similarity_matrix[mask].view(batch_size, batch_size - 1)
        negatives = negatives / self.temperature

        loss = -torch.log((torch.exp(positives)) / (
                    torch.exp(positives) + torch.sum(torch.exp(negatives), -1, True) + torch.exp(neg1) + torch.exp(
                neg2) + torch.exp(neg3)))
        return loss.sum() / (batch_size)



class NTXentLoss1_neg(torch.nn.Module):

    def __init__(self, module_dim, temperature):
        super(NTXentLoss1_neg, self).__init__()
        self.temperature = temperature
        self.fc = nn.Sequential(
            nn.Linear(module_dim, module_dim // 4, bias=True),
            nn.GELU(),
            nn.Linear(module_dim // 4, module_dim // 8, bias=True)
        )

    def forward(self, zis, zjs, neg):
        shape = zis.shape
        batch_size = shape[0]
        token = shape[1]
        zis = self.fc(zis)
        zjs = self.fc(zjs)
        zis = zis.permute(1, 0, 2)
        zjs = zjs.permute(1, 0, 2)
        zis1 = F.normalize(zis, p=2, dim=-1)
        zjs1 = F.normalize(zjs, p=2, dim=-1)
        similarity_matrix = torch.matmul(zis1, zjs1.permute(0, 2,
                                                            1))  # torch.sqrt(F.relu(2-2*torch.matmul(zis1,zjs1.permute(0,2,1))))

        n1 = self.fc(neg[:, :, 0, :]).permute(1, 0, 2)
        n2 = self.fc(neg[:, :, 1, :]).permute(1, 0, 2)
        n3 = self.fc(neg[:, :, 2, :]).permute(1, 0, 2)
        n1 = F.normalize(n1, p=2, dim=-1)
        n2 = F.normalize(n2, p=2, dim=-1)
        n3 = F.normalize(n3, p=2, dim=-1)

        neg1 = torch.sum(zis1 * n1, -1, True) / self.temperature
        neg2 = torch.sum(zis1 * n2, -1, True) / self.temperature
        neg3 = torch.sum(zis1 * n3, -1, True) / self.temperature

        shape = similarity_matrix.shape
        # filter out the scores from the positive samples
        l_pos = torch.diagonal(similarity_matrix, dim1=1, dim2=2)
        positives = l_pos.view(shape[0], batch_size,
                               1)  # torch.cat([l_pos, r_pos]).view(shape[0],2 * self.batch_size, 1)
        positives /= self.temperature

        diag = np.eye(batch_size)
        mask = torch.from_numpy((diag))
        mask = (1 - mask).type(torch.bool).cuda()
        negatives = similarity_matrix[:, mask].view(shape[0], batch_size, batch_size - 1)
        negatives /= self.temperature

        loss = -torch.log((torch.exp(positives)) / (
                    torch.exp(positives) + torch.sum(torch.exp(negatives), -1, True) + torch.exp(neg1) + torch.exp(
                neg2) + torch.exp(neg3)))
        return loss.sum() / (batch_size * token)