from munkres import Munkres
from torch import nn
from sklearn import metrics
from inference import euclidean_dist
from loss import Sample_Consistency, Community_Versatility, Community_Commonality
from einops import rearrange
import torch
import torch.nn.functional as F
import numpy as np
import time
import evaluation
import math

class Net(nn.Module):
    def __init__(self, in_dims, class_num, c_num=2, category=30):

        super(Net, self).__init__()
        if category==30:
            self.GroupWiseLayer='111111'
            self.BatchNormType='11011'
        else:
            self.GroupWiseLayer='100111'
            self.BatchNormType='11011'
        
        self.Dropout=0.2
        self.ElActivationType="Normalize"
        self.ActivationType='None'
        self.representation_dim=0
        self.McDecoder=False
        self.encoder_adaption = nn.ModuleList([
            get_ffn([in_dims[i], 1024], with_bn=self.BatchNormType[0] == '1', drop_out=self.Dropout)
            for i in range(c_num if self.GroupWiseLayer[0] == '1' else 1)])
        self.encoder_ = nn.ModuleList([
            get_ffn([1024,1024, 512], with_bn=self.BatchNormType[1] == '1', drop_out=self.Dropout)
            for _ in range(c_num if self.GroupWiseLayer[1] == '1' else 1)])
        if self.representation_dim == 0:
            self.representation_dim = class_num
        self.class_num = class_num
        self.c_num = c_num

        if self.ElActivationType == 'None':
            el_activation_ = []
        elif self.ElActivationType == 'Normalize':
            el_activation_ = []
        elif self.ElActivationType == 'BnNormalize':
            el_activation_ = [nn.BatchNorm1d(self.representation_dim)]
        elif self.ElActivationType == 'BnReNormalize':
            el_activation_ = [nn.BatchNorm1d(self.representation_dim), nn.ReLU()]
        elif self.ElActivationType == 'BnRe':
            el_activation_ = [nn.BatchNorm1d(self.representation_dim), nn.ReLU()]
        else:
            raise NotImplementedError('')
        self.el_activation_ = el_activation_
        self.encoder_linear = nn.ModuleList([
            get_ffn([512, 256], with_bn=self.BatchNormType[2] == '1', drop_out=self.Dropout,
                    last_layers=[nn.Linear(256, self.representation_dim)] + self.el_activation_)
            for _ in range(c_num if self.GroupWiseLayer[2] == '1' else 1)])
        dec_in = self.representation_dim
        if self.McDecoder:
            dec_in *= c_num
        self.dec_in = dec_in
        self.decoder_linear = nn.ModuleList([
            get_ffn([self.dec_in, 256, 512], with_bn=self.BatchNormType[3] == '1', drop_out=self.Dropout)
            for _ in range(c_num if self.GroupWiseLayer[3] == '1' else 1)])

        if self.ActivationType == 'None':
            final_activation_ = []
        elif self.ActivationType == 'Sigmoid':
            final_activation_ = [nn.Sigmoid()]
        elif self.ActivationType == 'Tanh':
            final_activation_ = [nn.Tanh()]
        else:
            raise NotImplementedError('')
        self.final_activation_ = final_activation_
        self.decoder_ = nn.ModuleList([
            get_ffn([512, 1024, 1024], with_bn=self.BatchNormType[4] == '1', drop_out=self.Dropout)
            for _ in range(c_num if self.GroupWiseLayer[4] == '1' else 1)])

        self.decoder_adaption = nn.ModuleList([
            get_ffn([], last_layers=[nn.Linear(1024, in_dims[i])] + self.final_activation_)
            for i in range(c_num if self.GroupWiseLayer[5] == '1' else 1)])

        self.linear_classify_fusion=nn.Sequential(*[nn.Linear(self.class_num*2, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, category)])
        self.linear_classify1=nn.Sequential(*[nn.Linear(self.class_num, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, category)])
        self.linear_classify2=nn.Sequential(*[nn.Linear(self.class_num, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, category)])
        self.ce_loss=nn.CrossEntropyLoss()

    def encoder(self,x1,x2):
        if self.GroupWiseLayer[0]=='1':
            latent1=self.encoder_adaption[0](x1)
            latent2=self.encoder_adaption[1](x2)
        else:
            latent1=self.encoder_adaption[0](x1)
            latent2=self.encoder_adaption[0](x2)
        if self.GroupWiseLayer[1]=='1':
            latent1=self.encoder_[0](latent1)
            latent2=self.encoder_[1](latent2)
        else:
            latent1=self.encoder_[0](latent1)
            latent2=self.encoder_[0](latent2)
        if self.GroupWiseLayer[2]=='1':
            latent1=self.encoder_linear[0](latent1)
            latent2=self.encoder_linear[1](latent2)
        else:
            latent1=self.encoder_linear[0](latent1)
            latent2=self.encoder_linear[0](latent2)
        
        return F.normalize(latent1,dim=1),F.normalize(latent2,dim=1)
    def decoder(self, latent1, latent2):
        latent1=self.decoder_linear[0](latent1)
        latent2=self.decoder_linear[1](latent2)
        latent1=self.decoder_[0](latent1)
        latent2=self.decoder_[1](latent2)
        latent1=self.decoder_adaption[0](latent1)
        latent2=self.decoder_adaption[1](latent2)
        return latent1,latent2
def get_ffn(dims, last_layers=None, with_bn=False, drop_out=0):
    layers = []
    for ind in range(len(dims) - 1):
        in_dim = dims[ind]
        out_dim = dims[ind + 1]
        layers.append(nn.Linear(in_dim, out_dim))
        if with_bn:
            layers.append(nn.BatchNorm1d(out_dim))
        layers.append(nn.ReLU())
        if drop_out:
            layers.append(nn.Dropout(drop_out))
    if last_layers is not None:
        layers.extend(last_layers)
    return nn.Sequential(*layers)

class CAMERA(nn.Module):
    """CAMERA: Cross-modal Attention-based Representation Alignment."""

    def __init__(self, config):
        """Constructor.

        Args:
          config: parameters defined in configure.py.
        """
        super().__init__()
        self.config = config

        self.autoencoder=Net(in_dims=config['Autoencoder']['in_dims'], class_num=config['training']['dim'], c_num=2, category=config['training']['num'])
        self.dim = config['training']['dim']
        self.num = config['training']['num']
        self.mutual_attention = MutualAttention(dim=self.dim, num=self.num).cuda()
        self.consistency = Sample_Consistency(temperature=0.5)
        self.versatility = Community_Versatility(batch_size=self.num, temperature=1)
        self.projector = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.BatchNorm1d(self.dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim, self.dim),
        )
        self.l2_norm = F.normalize

    def to_device(self, device):
        """ to cuda if gpu is used """
        self.autoencoder.to(device)
        self.mutual_attention.to(device)
        self.projector.to(device)

    def train(self,args, config, train_loader, eval_train_loader, index_loader,eval_test_loader, logger,accumulated_metrics, optimizer, device):
        """Training the model.

            Args:
              config: parameters which defined in configure.py.
              logger: print the information.
              x1_train: data of view 1
              x2_train: data of view 2
              Y_list: labels
              mask: generate missing data
              optimizer: adam is used in our experiments
              device: to cuda if gpu is used
            Returns:
              clustering performance: acc, nmi ,ari


        """

        index_swap=None
        time0=time.time()
        mutual_attention_use=True
        for epoch in range(args.epoch):
            time1=time.time()
            loss_epoch_total, loss_epoch_rec, loss_epoch_com, loss_epoch_con, loss_epoch_ver, loss_epoch_ce = 0, 0, 0, 0, 0, 0
            for iter, (batch_x1, batch_x2, gt_label, _, mask, align, _) in enumerate(train_loader):
                rec_1=None
                rec_2=None
                batch_x1=batch_x1.cuda()
                batch_x2=batch_x2.cuda()
                gt_label=gt_label.cuda()
                pariwise_idx=(mask[:, 1]==1) & (mask[:, 0]==1) & (align[:]==1)
                only1_idx = torch.logical_xor(mask[:, 0] == 1, pariwise_idx)
                only2_idx = torch.logical_xor(mask[:, 1] == 1, pariwise_idx)

                # Get index on first evaluation epoch
                if epoch == 49 and index_swap is None:
                    if args.aligned_prop != 0 and args.complete_prop != 0:
                        index_swap = self.get_index(config, index_loader)
                    else:
                        index_swap = self.get_index_completely(config, index_loader)

                # Encode pair-wise data
                s_both_1, s_both_2 = self.autoencoder.encoder(batch_x1[pariwise_idx, :], batch_x2[pariwise_idx, :])
                z_both_1, z_both_2, u_both_1, u_both_2, attn_both_1, attn_both_2 = self.mutual_attention(s_both_1, s_both_2)
                
                # Decode pair-wise data
                if batch_x1[pariwise_idx,:].shape[0] != 0:
                    rec_both_1,rec_both_2=self.autoencoder.decoder(F.normalize(z_both_1,dim=1),F.normalize(z_both_2,dim=1))
                    rec1=rec_both_1
                    rec2=rec_both_2
                    attn1=attn_both_1
                    attn2=attn_both_2
                else:
                    rec1, rec2 = None, None
                    attn1, attn2 = None, None
                
                # Encode unpaired data
                if batch_x1[only1_idx, :].shape[0] != 0 or batch_x2[only2_idx, :].shape[0] != 0:
                    s_single_1, s_single_2 = self.autoencoder.encoder(
                        batch_x1[only1_idx, :], batch_x2[only2_idx, :])
                    z_single_1, z_single_2, u_single_1, u_single_2, attn_single_1, attn_single_2 = self.mutual_attention(s_single_1, s_single_2)
                    rec_single_1, rec_single_2 = self.autoencoder.decoder(
                        F.normalize(z_single_1, dim=1), F.normalize(z_single_2, dim=1))
                    
                    if attn1 is None:
                        attn1, attn2 = attn_single_1, attn_single_2
                    else:
                        attn1 = torch.cat([attn1, attn_single_1], dim=0)
                        attn2 = torch.cat([attn2, attn_single_2], dim=0)
                    
                    if rec1 is None:
                        rec1, rec2 = rec_single_1, rec_single_2
                    else:
                        rec1 = torch.cat([rec1, rec_single_1], dim=0)
                        rec2 = torch.cat([rec2, rec_single_2], dim=0)       
                s1 = torch.cat([batch_x1[pariwise_idx, :], batch_x1[only1_idx, :]], dim=0)
                s2 = torch.cat([batch_x2[pariwise_idx, :], batch_x2[only2_idx, :]], dim=0)
                
                # Reconstruction loss
                loss_epoch_reconstruction = (F.mse_loss(rec1, s1.detach()) + F.mse_loss(rec2, s2.detach())) / 2
                # Community Commonality loss
                loss_commonality = Community_Commonality(attn1, attn2)
                
                loss_total = loss_epoch_reconstruction * args.rec_lambda + loss_commonality
                
                # Community Consistency loss
                loss_consistency = 0
                if batch_x1[pariwise_idx, :].shape[0] != 0:
                    loss_consistency = self.consistency(
                        self.l2_norm(self.projector(z_both_1), dim=1),
                        self.l2_norm(self.projector(z_both_2), dim=1))
                    loss_total += loss_consistency

                # Community Versatility loss
                loss_versatility = 0
                if epoch >= config['training']['pretrain_epoch'] - 1 and index_swap is not None:
                    has_unpaired = batch_x1[only1_idx].shape[0] != 0 or batch_x2[only2_idx].shape[0] != 0
                    if has_unpaired:
                        community_1 = (u_both_1 * args.complete_prop + u_single_1 * (1 - args.complete_prop) / 2) / ((1 + args.complete_prop) / 2)
                        community_2 = (u_both_2 * args.complete_prop + u_single_2 * (1 - args.complete_prop) / 2) / ((1 + args.complete_prop) / 2)
                    else:
                        community_1, community_2 = u_both_1, u_both_2
                    
                    community_1 = community_1[index_swap, :]
                    loss_versatility = self.versatility(
                        self.l2_norm(self.projector(community_1), dim=1),
                        self.l2_norm(self.projector(community_2), dim=1))
                    loss_total += loss_versatility

                # supervised loss
                represent_concat=torch.cat([z_both_1, z_both_2], dim=1)
                pred_rep=self.autoencoder.linear_classify_fusion(represent_concat)
                loss_sup_fuison=self.autoencoder.ce_loss(pred_rep,gt_label[pariwise_idx])
                loss_sup=(self.autoencoder.ce_loss(self.autoencoder.linear_classify1(z_both_1),gt_label[pariwise_idx])+self.autoencoder.ce_loss(self.autoencoder.linear_classify1(z_both_2),gt_label[pariwise_idx]))/2

                loss_total=(loss_sup+loss_sup_fuison) * args.sup_lambda + loss_total

                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

                loss_epoch_total += loss_total.item()
                loss_epoch_rec += loss_epoch_reconstruction.item()
                loss_epoch_com += loss_commonality.item()
                loss_epoch_con += loss_consistency.item() if isinstance(loss_consistency, torch.Tensor) else 0
                loss_epoch_ver += loss_versatility.item() if isinstance(loss_versatility, torch.Tensor) else 0
                loss_epoch_ce += (loss_sup + loss_sup_fuison).item()
        
            if (epoch + 1) % config['print_num'] == 0:
                output = "Epoch : {:.0f}/{:.0f} ===> Learing Rate = {:.4f}" \
                         "===> Rec loss = {:.4e} ===> Com loss = {:.4e} ===> Con loss = {:.4e} ===> Ver loss = {:.4e} ===> CE loss = {:.4e} ===> Loss = {:.4e}" \
                    .format((epoch + 1), args.epoch, optimizer.state_dict()['param_groups'][0]['lr'],
                            loss_epoch_rec, loss_epoch_com, loss_epoch_con, loss_epoch_ver, loss_epoch_ce, loss_epoch_total)

                logger.info("\033[2;29m" + output + "\033[0m")

            epoch_train_time=time.time()-time1
            # evalution
            if (epoch + 1) % config['print_num'] == 0:

                time2=time.time()
                fusion_out_test, fusion_RD_out_test,fusion_DR_out_test,fusion_R_out_test,fusion_D_out_test, labels_test,pred_out_test,pred_RD_out_test,pred_DR_out_test,pred_R_out_test,pred_D_out_test=self.both_infer(args, config, logger, eval_test_loader,index_loader, device, epoch,index_swap,mutual_attention_use)
                epoch_eval_time=time.time()-time2

                scores_RD = evaluation.score(labels_test, pred_RD_out_test)
                scores_DR = evaluation.score(labels_test, pred_DR_out_test)
                scores = evaluation.score(labels_test, pred_out_test)
                scores_onlyrgb = evaluation.score(labels_test, pred_R_out_test)
                scores_onlydepth = evaluation.score(labels_test, pred_D_out_test)

                accumulated_metrics['RGB'].append(scores_RD)
                accumulated_metrics['Depth'].append(scores_DR)
                accumulated_metrics['RGB-D'].append(scores)
                accumulated_metrics['onlyRGB'].append(scores_onlyrgb)
                accumulated_metrics['onlyDepth'].append(scores_onlydepth)

                logger.info('\033[2;29m RGB   Accuracy on the test set is {:.4f}'.format(scores_RD))
                logger.info('\033[2;29m Depth Accuracy on the test set is {:.4f}'.format(scores_DR))
                logger.info('\033[2;29m RGB+D Accuracy on the test set is {:.4f}'.format(scores))
                logger.info('\033[2;29m onlyRGB Accuracy on the test set is {:.4f}'.format(scores_onlyrgb))
                logger.info('\033[2;29m onlyDepth Accuracy on the test set is {:.4f}'.format(scores_onlydepth))
                print(f"It took {epoch_train_time:.2f}s for a epoch training.") 
                print(f"It took {epoch_eval_time:.2f}s for evaluation.") 
        test_train_time=time.time()-time0
        return accumulated_metrics['RGB'][-1], accumulated_metrics['Depth'][-1], accumulated_metrics['RGB-D'][
            -1], accumulated_metrics['onlyRGB'][-1], accumulated_metrics['onlyDepth'][-1], test_train_time


    def both_infer(self, args, config, logger, eval_train_loader, index_dataloader, device, epoch, index_swap, mutual_attention_use):
        with torch.no_grad():
            self.autoencoder.eval()
            self.mutual_attention.eval()
            self.projector.eval()
            latent_fusion_out = []
            latent_fusion_RD_out = []
            latent_fusion_DR_out = []
            latent_fusion_R_out = []
            latent_fusion_D_out = []

            lable_fusion_out = []
            lable_fusion_RD_out=[]
            lable_fusion_DR_out=[]
            lable_fusion_R_out=[]
            lable_fusion_D_out=[]

            class_labels1 = []
            class_labels2 = []

            #get_index
            if index_swap is None:
                for (x1_train, x2_train, label1, label2, mask, align, _) in index_dataloader:
                    x1_train=x1_train.cuda()
                    x2_train=x2_train.cuda()
                    #deal missing data
                    common_idx_eval = (mask[:, 0] == 1) & (mask[:, 1] == 1)
                    if x1_train[common_idx_eval].shape[0] == 0:
                        common_idx_eval[:]=True
                    imgs_latent_eval_common, txts_latent_eval_common = self.autoencoder.encoder(x1_train[common_idx_eval],x2_train[common_idx_eval])
                    swap_infer = self.mutual_attention.get_swap(imgs_latent_eval_common, txts_latent_eval_common, config['training']['num'])
                    print("Infer_index_swap:",swap_infer)
                    index_swap=swap_infer

            for (x1_train, x2_train, label1, label2, mask, align, _) in eval_train_loader:
                class_labels1.extend((label1.cpu()).numpy())
                class_labels2.extend((label2.cpu()).numpy())
                x1_train=x1_train.cuda()
                x2_train=x2_train.cuda()
                mask=mask.cuda()
                img_idx_eval = mask[:, 0] == 1
                txt_idx_eval = mask[:, 1] == 1

                # no_missing_data
                imgs_latent_x, txts_latent_x = self.autoencoder.encoder(x1_train[img_idx_eval],x2_train[txt_idx_eval])
                
                imgs_latent_z, txts_latent_z,_,_,attn_img_exist,attn_txt_exist=self.mutual_attention(imgs_latent_x, txts_latent_x)

                # MA-based Imputation
                img2txt_recon_z, txt2img_recon_z,attn_txt_missing,attn_img_missing= self.mutual_attention.dual_pre(imgs_latent_x, txts_latent_x, index_swap)

                # R->D
                latent_code_img_eval_RD = imgs_latent_z
                latent_code_txt_eval_RD = img2txt_recon_z
                latent_fusion_RD = torch.cat([latent_code_img_eval_RD, latent_code_txt_eval_RD],dim=1).cpu().numpy()

                # D->R
                latent_code_txt_eval_DR = txts_latent_z
                latent_code_img_eval_DR = txt2img_recon_z
                latent_fusion_DR= torch.cat([latent_code_img_eval_DR, latent_code_txt_eval_DR],dim=1).cpu().numpy()

                # R+D
                latent_fusion = torch.cat([imgs_latent_z, txts_latent_z], dim=1).cpu().numpy()

                latent_fusion_out.extend(latent_fusion)
                latent_fusion_RD_out.extend(latent_fusion_RD)
                latent_fusion_DR_out.extend(latent_fusion_DR)
                latent_fusion_R_out.extend(imgs_latent_z.cpu().numpy())
                latent_fusion_D_out.extend(txts_latent_z.cpu().numpy())

                lable_fusion_out=torch.argmax(F.softmax(self.autoencoder.linear_classify_fusion(torch.cat([imgs_latent_z, txts_latent_z], dim=1)),dim=1),dim=1).cpu().numpy()
                lable_fusion_RD_out=torch.argmax(F.softmax(self.autoencoder.linear_classify_fusion(torch.cat([latent_code_img_eval_RD, latent_code_txt_eval_RD], dim=1)),dim=1),dim=1).cpu().numpy()
                lable_fusion_DR_out=torch.argmax(F.softmax(self.autoencoder.linear_classify_fusion(torch.cat([latent_code_img_eval_DR, latent_code_txt_eval_DR], dim=1)),dim=1),dim=1).cpu().numpy()
                lable_fusion_R_out=torch.argmax(F.softmax(self.autoencoder.linear_classify1(imgs_latent_z),dim=1),dim=1).cpu().numpy()
                lable_fusion_D_out=torch.argmax(F.softmax(self.autoencoder.linear_classify1(txts_latent_z),dim=1),dim=1).cpu().numpy()

            fusion_out=np.array(latent_fusion_out)
            fusion_RD_out=np.array(latent_fusion_RD_out)
            fusion_DR_out=np.array(latent_fusion_DR_out)
            fusion_R_out=np.array(latent_fusion_R_out)
            fusion_D_out=np.array(latent_fusion_D_out)

            lable_out=np.array(lable_fusion_out)
            lable_RD_out=np.array(lable_fusion_RD_out)
            lable_DR_out=np.array(lable_fusion_DR_out)
            lable_R_out=np.array(lable_fusion_R_out)
            lable_D_out=np.array(lable_fusion_D_out)
            labels1 = np.array(class_labels1)

            self.autoencoder.train()
            self.mutual_attention.train()
            self.projector.train()
        return fusion_out, fusion_RD_out, fusion_DR_out, fusion_R_out, fusion_D_out, labels1, lable_out, lable_RD_out, lable_DR_out, lable_R_out, lable_D_out
    def get_index(self, config, test_dataloader):
        """Get index swap from test data"""
        with torch.no_grad():
            self.autoencoder.eval()
            self.mutual_attention.eval()
            for batch_x1, batch_x2, label1, label2, mask, align, index in test_dataloader:
                batch_x1, batch_x2, mask, align = batch_x1.cuda(), batch_x2.cuda(), mask.cuda(), align.cuda()
                both_idx_eval = (mask[:, 0] == 1) & (mask[:, 1] == 1) & (align[:] == 1)
                if batch_x1[both_idx_eval].shape[0] == 0:
                    both_idx_eval[:] = True
                s_both_1, s_both_2 = self.autoencoder.encoder(
                    batch_x1[both_idx_eval], batch_x2[both_idx_eval])
                index_swap = self.mutual_attention.get_swap(
                    s_both_1, s_both_2, config['training']['num'])
                print("Infer_index_swap:", index_swap)
            self.autoencoder.train()
            self.mutual_attention.train()
        return index_swap

    def get_index_completely(self, config, test_dataloader):
        """Get index swap using Hungarian algorithm"""
        num = config['training']['num']
        with torch.no_grad():
            self.autoencoder.eval()
            self.mutual_attention.eval()
            for batch_x1, batch_x2, label1, label2, mask, align, index in test_dataloader:
                batch_x1, batch_x2, mask = batch_x1.cuda(), batch_x2.cuda(), mask.cuda()
                s1, s2 = self.autoencoder.encoder(
                    batch_x1[mask[:, 0], :], batch_x2[mask[:, 1], :])
                C = euclidean_dist(s1, s2)
                attn1, attn2 = self.mutual_attention.get_attn(s1, s2)
                num_1 = attn1.size(0)
                attn1_knn = torch.zeros(num_1, config['training']['num']).cuda()

                for i in range(num_1):
                    idx = torch.argsort(C[i, :])
                    attn1_knn[i] = attn2[idx[0]]

                label_1 = torch.argmax(attn1, dim=1).detach().cpu().numpy()
                label_2 = torch.argmax(attn1_knn, dim=1).detach().cpu().numpy()
                confusion_matrix = metrics.confusion_matrix(label_1, label_2, labels=list(range(num)))
                cost_matrix = calculate_cost_matrix(confusion_matrix, num)
                indices = Munkres().compute(cost_matrix)
                kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices).astype(int)

            self.autoencoder.train()
            self.mutual_attention.train()
        return kmeans_to_true_cluster_labels

class MutualAttention(nn.Module):
    def __init__(self, batch_size=256, num=21, dim=128):
        super().__init__()
        self.num = num
        self.dim = dim
        self.l2_norm = F.normalize
        self.num_heads = 1
        self.attention = Attention(dim=dim, num=num, head=self.num_heads)
        self.c_token_1 = nn.Linear(dim, num, bias=False)
        self.c1 = self.c_token_1.weight
        self.c_token_2 = nn.Linear(dim, num, bias=False)
        self.c2 = self.c_token_2.weight

    def pre(self, z1, c1, z2, c2):
        c1 = F.normalize(c1, dim=1)
        c2 = F.normalize(c2, dim=1)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        return z1, c1, z2, c2
    def get_swap(self, z1, z2, num):
        s1, c1, s2, c2 = self.pre(z1, self.c1, z2, self.c2)
        attn1,attn2=self.attention.get_attention(s1, c1, s2, c2)
        label_img = torch.argmax(attn1.mean(0), dim=1).detach().cpu().numpy()
        label_txt = torch.argmax(attn2.mean(0), dim=1).detach().cpu().numpy()
        confusion_matrix = metrics.confusion_matrix(label_img, label_txt, labels=[i for i in range(num)])
        cost_matrix = calculate_cost_matrix(confusion_matrix, num)
        indices = Munkres().compute(cost_matrix)
        kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices).astype(int)
        return kmeans_to_true_cluster_labels

    def get_attn(self, z1, z2):
        s1, c1, s2, c2 = self.pre(z1, self.c1, z2, self.c2)
        attn1,attn2=self.attention.get_attention(s1, c1, s2, c2)
        soft1 = attn1.mean(0)
        soft2 = attn2.mean(0)
        return soft1,soft2

    def dual_pre(self, z1, z2, index_swap):
        s1, c1, s2, c2 = self.pre(z1, self.c1, z2, self.c2)
        rec1,rec2,attn1,attn2=self.attention.recovery(s1, c1, s2, c2,index_swap)
        rec1 = F.normalize(rec1, dim=1) + F.normalize(s2, dim=1)
        rec2 = F.normalize(rec2, dim=1) + F.normalize(s1, dim=1)
        return rec2,rec1,attn1,attn2

    def forward(self, z1, z2):
        s1, c1, s2, c2 = self.pre(z1, self.c1, z2, self.c2)
        z1,z2,c_img,c_txt,attn1,attn2=self.attention(s1, c1, s2, c2)
        z1 = F.normalize(z1, dim=1) + F.normalize(s1, dim=1)
        z2 = F.normalize(z2, dim=1) + F.normalize(s2, dim=1)
        c_img = F.normalize(c_img, dim=1) + F.normalize(c1, dim=1)
        c_txt = F.normalize(c_txt, dim=1) + F.normalize(c2, dim=1)
        
        return z1, z2,c_img,c_txt, attn1.mean(0), attn2.mean(0)


class Attention(nn.Module):
    def __init__(self, dim=128, num=10, head=1):
        super().__init__()
        self.scale = dim ** -0.5
        self.num_heads = head
        self.dim = dim
        self.num = num
        # view1参数
        self.q_proj_v1 = nn.Linear(dim, dim, bias=True)
        self.k_proj_v1 = nn.Linear(dim, dim, bias=True)
        self.v_proj1_v1 = nn.Linear(dim, dim, bias=True)
        self.v_proj2_v1 = nn.Linear(dim, dim, bias=True)
        self.norm1_v1 = nn.LayerNorm(dim)
        self.norm2_v1 = nn.LayerNorm(dim)
        # view2参数
        self.q_proj_v2 = nn.Linear(dim, dim, bias=True)
        self.k_proj_v2 = nn.Linear(dim, dim, bias=True)
        self.v_proj1_v2 = nn.Linear(dim, dim, bias=True)
        self.v_proj2_v2 = nn.Linear(dim, dim, bias=True)
        self.norm1_v2 = nn.LayerNorm(dim)
        self.norm2_v2 = nn.LayerNorm(dim)

    def _compute_attention(self, s, c, q_proj, k_proj, v_proj1, v_proj2, norm1, norm2):
        B = s.size(0)
        q = rearrange(
            q_proj(norm1(s)),
            "n (h c)-> h n c",
            h=self.num_heads,
            n=B,
            c=self.dim // self.num_heads,
        )
        k = rearrange(
            k_proj(norm2(c)),
            "n (h c)-> h n c",
            n=self.num,
            h=self.num_heads,
            c=self.dim // self.num_heads,
        )
        v_s = rearrange(
            v_proj1(norm1(s)),
            "n (h c)-> h n c",
            h=self.num_heads,
            n=B,
            c=self.dim // self.num_heads,
        )
        v_c = rearrange(
            v_proj2(norm2(c)),
            "n (h c)-> h n c",
            n=self.num,
            h=self.num_heads,
            c=self.dim // self.num_heads,
        )
        sim = (q @ k.transpose(-2, -1)) * self.scale
        attn = sim.softmax(dim=-1)
        return attn, v_s, v_c

    def get_attention(self, s1, c1, s2, c2):
        attn1, v_s1, v_c1 = self._compute_attention(
            s1, c1,
            self.q_proj_v1, self.k_proj_v1, self.v_proj1_v1, self.v_proj2_v1,
            self.norm1_v1, self.norm2_v1
        )
        attn2, v_s2, v_c2 = self._compute_attention(
            s2, c2,
            self.q_proj_v2, self.k_proj_v2, self.v_proj1_v2, self.v_proj2_v2,
            self.norm1_v2, self.norm2_v2
        )
        return attn1, attn2

    def recovery(self, s1, c1, s2, c2, index_swap):
        B1 = s1.size(0)
        B2 = s2.size(0)
        attn1, v_s1, v_c1 = self._compute_attention(
            s1, c1,
            self.q_proj_v1, self.k_proj_v1, self.v_proj1_v1, self.v_proj2_v1,
            self.norm1_v1, self.norm2_v1
        )
        attn2, v_s2, v_c2 = self._compute_attention(
            s2, c2,
            self.q_proj_v2, self.k_proj_v2, self.v_proj1_v2, self.v_proj2_v2,
            self.norm1_v2, self.norm2_v2
        )
        attn1 = attn1[:, :, index_swap]
        attn1_swap = attn1.mean(0)
        attn2_swap = attn2.mean(0)
        rev_swap = np.argsort(index_swap)
        attn2 = attn2[:, :, rev_swap]
        impute_z2 = rearrange(
            attn1 @ v_c2,
            "h n c -> n (h c)",
            h=self.num_heads,
            n=B1,
            c=self.dim // self.num_heads,
        )
        impute_z1 = rearrange(
            attn2 @ v_c1,
            "h n c -> n (h c)",
            h=self.num_heads,
            n=B2,
            c=self.dim // self.num_heads,
        )
        return impute_z1, impute_z2, attn1_swap, attn2_swap

    def forward(self, s1, c1, s2, c2):
        B1 = s1.size(0)
        B2 = s2.size(0)
        attn1, v_s1, v_c1 = self._compute_attention(
            s1, c1,
            self.q_proj_v1, self.k_proj_v1, self.v_proj1_v1, self.v_proj2_v1,
            self.norm1_v1, self.norm2_v1
        )
        attn2, v_s2, v_c2 = self._compute_attention(
            s2, c2,
            self.q_proj_v2, self.k_proj_v2, self.v_proj1_v2, self.v_proj2_v2,
            self.norm1_v2, self.norm2_v2
        )
        z1 = rearrange(
            attn1 @ v_c1,
            "h n c -> n (h c)",
            h=self.num_heads,
            n=B1,
            c=self.dim // self.num_heads,
        )
        z2 = rearrange(
            attn2 @ v_c2,
            "h n c -> n (h c)",
            h=self.num_heads,
            n=B2,
            c=self.dim // self.num_heads,
        )
        c1 = rearrange(
            attn1.transpose(-2, -1) @ v_s1,
            "h n c -> n (h c)",
            h=self.num_heads,
            n=self.num,
            c=self.dim // self.num_heads,
        )
        c2 = rearrange(
            attn2.transpose(-2, -1) @ v_s2,
            "h n c -> n (h c)",
            h=self.num_heads,
            n=self.num,
            c=self.dim // self.num_heads,
        )
        return z1, z2, c1, c2, attn1, attn2

def calculate_cost_matrix(confusion, n_clusters):
    """Calculate cost matrix from confusion matrix"""
    cost_matrix = np.zeros((n_clusters, n_clusters))
    for j in range(n_clusters):
        s = np.sum(confusion[:, j])
        for i in range(n_clusters):
            cost_matrix[j, i] = s - confusion[i, j]
    return cost_matrix


def get_cluster_labels_from_indices(indices):
    """Extract cluster labels from Hungarian algorithm indices"""
    n_clusters = len(indices)
    cluster_labels = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_labels[i] = indices[i][1]
    return cluster_labels
