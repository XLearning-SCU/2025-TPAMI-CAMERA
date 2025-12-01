from munkres import Munkres
from torch import nn
from sklearn import metrics, svm
from inference import euclidean_dist
from loss import Sample_Consistency, Community_Versatility, Community_Commonality
from net import Net, NetCov
from einops import rearrange
import torch
import torch.nn.functional as F
import numpy as np
import time
import evaluation
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

class CAMERA(nn.Module):
    """CAMERA: Cross-modal Attention-based Representation Alignment."""

    def __init__(self, config):
        """Constructor.

        Args:
          config: parameters defined in configure.py.
        """
        super().__init__()
        self.config = config
        cov = config['training']['cov']
        if cov:
            print("Covnet")
            self.autoencoder = NetCov(in_dims=config['Autoencoder']['in_dims'], 
                                      class_num=config['training']['dim'], c_num=2)
        else:
            self.autoencoder = Net(in_dims=config['Autoencoder']['in_dims'], 
                                   class_num=config['training']['dim'], c_num=2)
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
            time1 = time.time()
            loss_epoch_rec, loss_epoch_com, loss_epoch_con, loss_epoch_ver, loss_epoch_total = 0, 0, 0, 0, 0
            
            for iter, (batch_x1, batch_x2, _, _, mask, align, _) in enumerate(train_loader):
                batch_x1, batch_x2 = batch_x1.cuda(), batch_x2.cuda()
                pariwise_idx = (mask[:, 1] == 1) & (mask[:, 0] == 1) & (align[:] == 1)
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
                loss_reconstruction = (F.mse_loss(rec1, s1.detach()) + F.mse_loss(rec2, s2.detach())) / 2
                # Community Commonality loss
                loss_commonality = Community_Commonality(attn1, attn2)
                
                loss_total = loss_reconstruction * args.rec_lambda + loss_commonality
                
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

                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

                loss_epoch_total += loss_total.item()
                loss_epoch_rec += loss_reconstruction.item()
                loss_epoch_com += loss_commonality.item()
                loss_epoch_con += loss_consistency.item() if isinstance(loss_consistency, torch.Tensor) else 0
                loss_epoch_ver += loss_versatility.item() if isinstance(loss_versatility, torch.Tensor) else 0

            if (epoch + 1) % config['print_num'] == 0:
                lr = optimizer.state_dict()['param_groups'][0]['lr']
                output = f"Epoch : {epoch + 1:.0f}/{args.epoch:.0f} ===> Learning Rate = {lr:.4f} " \
                         f"===> Rec = {loss_epoch_rec:.4e} ===> Com = {loss_epoch_com:.4e} " \
                         f"===> Con = {loss_epoch_con:.4e} ===> Ver = {loss_epoch_ver:.4e} ===> Total = {loss_epoch_total:.4e}"
                logger.info("\033[2;29m" + output + "\033[0m")
                
            epoch_train_time=time.time()-time1
            # evalution
            if (epoch + 1) % config['print_num'] == 0:
                time2=time.time()
                latent_fusion_train,labels_train = self.both_infer(args, config, logger, eval_train_loader,index_loader, device, epoch,index_swap,mutual_attention_use)
                epoch_eval_time=time.time()-time2
                latent_fusion_test,labels_test = self.both_infer(args, config, logger, eval_test_loader,index_loader, device, epoch,index_swap,mutual_attention_use)

                # SVM Classification
                clf = svm.SVC()  
                clf.fit(latent_fusion_train, labels_train)  
                label_pre = clf.predict(latent_fusion_test)

                scores = accuracy_score(labels_test, label_pre)
                precision = precision_score(labels_test, label_pre, average='macro', labels=np.unique(labels_train))
                precision = np.round(precision, 2)

                f_score = f1_score(labels_test, label_pre, average='macro', labels=np.unique(labels_train))
                f_score = np.round(f_score, 2)

                accumulated_metrics['acc'].append(scores)
                accumulated_metrics['precision'].append(precision)
                accumulated_metrics['f_measure'].append(f_score)
                logger.info('\033[2;29m Accuracy on the test set is {:.4f}'.format(scores))
                logger.info('\033[2;29m Precision on the test set is {:.4f}'.format(precision))
                logger.info('\033[2;29m F_score on the test set is {:.4f}'.format(f_score))
                print(f"It took {epoch_train_time:.2f}s for a epoch training.") 
                print(f"It took {epoch_eval_time:.2f}s for evaluation.") 
        test_train_time=time.time()-time0
        return accumulated_metrics['acc'][-1], accumulated_metrics['precision'][-1], accumulated_metrics['f_measure'][
            -1], test_train_time


    def both_infer(self, args, config, logger, test_dataloader, index_dataloader, device, epoch, index_swap, mutual_attention_use):
        with torch.no_grad():
            self.autoencoder.eval()
            self.mutual_attention.eval()
            self.projector.eval()
            fusion_out = []
            class_labels1 = []

            # Get index if needed
            if index_swap is None:
                for (x1_train, x2_train, label1, label2, mask, align, _) in index_dataloader:
                    x1_train = x1_train.cuda()
                    x2_train = x2_train.cuda()
                    # deal missing data
                    common_idx_eval = (mask[:, 0] == 1) & (mask[:, 1] == 1) & (align[:] == 1)
                    if x1_train[common_idx_eval].shape[0] == 0:
                        common_idx_eval[:] = True
                    imgs_latent_eval_common, txts_latent_eval_common = self.autoencoder.encoder(x1_train[common_idx_eval], x2_train[common_idx_eval])
                    swap_infer = self.mutual_attention.get_swap(imgs_latent_eval_common, txts_latent_eval_common, config['training']['num'])
                    print("Infer_index_swap:", swap_infer)
                    index_swap = swap_infer

            # Encode test-time data
            for batch_x1, batch_x2, label1, label2, mask, align, _ in test_dataloader:
                eval_batchsize = batch_x1.shape[0]
                class_labels1.extend(label1.cpu().numpy())
                batch_x1, batch_x2, mask = batch_x1.cuda(), batch_x2.cuda(), mask.cuda()

                # Compute indices
                img_idx_eval = mask[:, 0] == 1
                txt_idx_eval = mask[:, 1] == 1
                img_missing_idx_eval = ~img_idx_eval
                txt_missing_idx_eval = ~txt_idx_eval
                both_existing_idx = img_idx_eval & txt_idx_eval

                # Initialize outputs
                z1_eval = torch.zeros(eval_batchsize, config['training']['dim']).to(device)
                z2_eval = torch.zeros(eval_batchsize, config['training']['dim']).to(device)
                attn_img_eval = torch.zeros(eval_batchsize, config['training']['num']).to(device)
                attn_txt_eval = torch.zeros(eval_batchsize, config['training']['num']).to(device)

                # Process existing data
                if img_idx_eval.any() and txt_idx_eval.any():
                    s1, s2 = self.autoencoder.encoder(
                        batch_x1[img_idx_eval], batch_x2[txt_idx_eval])
                    z1_both, z2_both = s1, s2
                    z1, z2, _, _, attn1_exist, attn2_exist = \
                        self.mutual_attention(s1, s2)
                    z1_eval[img_idx_eval] = z1
                    z2_eval[txt_idx_eval] = z2

                    # MA-based Imputation
                    if img_missing_idx_eval.any() or txt_missing_idx_eval.any():
                        s1_missing, s2_missing = self.autoencoder.encoder(
                            batch_x1[txt_missing_idx_eval], batch_x2[img_missing_idx_eval])
                        z1_recon, z2_recon, attn2_missing, attn1_missing = \
                            self.mutual_attention.dual_pre(s1_missing, s2_missing, index_swap)
                        z1_eval[img_missing_idx_eval] = z2_recon
                        z2_eval[txt_missing_idx_eval] = z1_recon
                        attn_img_eval[img_missing_idx_eval] = attn1_missing
                        attn_txt_eval[txt_missing_idx_eval] = attn2_missing

                # Compute fusion representation
                z1_eval = F.normalize(z1_eval, dim=1)
                z2_eval = F.normalize(z2_eval, dim=1)

                if args.aligned_prop == 1.0:
                    fusion_out.extend(torch.cat([z1_eval, z2_eval], dim=1).cpu().numpy())
                    continue

                # MA-based Alignment
                attn_img_eval[img_idx_eval] = attn1_exist[:, index_swap]
                attn_txt_eval[txt_idx_eval] = attn2_exist

                _, max_img_index = torch.max(attn_img_eval, dim=1)
                _, max_txt_index = torch.max(attn_txt_eval, dim=1)
                atom_realign = (max_img_index != max_txt_index) & both_existing_idx
                realign_num = torch.count_nonzero(atom_realign)

                if realign_num > 0:
                    z1_both_exist = z1_both[atom_realign[img_idx_eval], :]
                    z2_both_exist = z2_both[atom_realign[txt_idx_eval], :]
                    persudo_label_img_realign = max_img_index[atom_realign]
                    persudo_label_txt_realign = max_txt_index[atom_realign]

                    z2_realign = z2_eval[atom_realign]
                    C = euclidean_dist(z1_both_exist, z2_both_exist)
                    z2_realign_list = []
                    min_num = min(5, realign_num)

                    for i in range(realign_num):
                        idx2 = torch.argsort(C[i, :])
                        realign_sample = None
                        for j in range(min_num):
                            if persudo_label_img_realign[i] == persudo_label_txt_realign[idx2[j]]:
                                realign_sample = idx2[j]
                                break
                        sample_idx = realign_sample if realign_sample is not None else idx2[0]
                        z2_realign_list.append(z2_realign[sample_idx].view(1, -1))

                    z2_realigned = torch.cat(z2_realign_list, dim=0)
                    z2_eval[atom_realign] = z2_realigned

                z1_eval = F.normalize(z1_eval, dim=1)
                z2_eval = F.normalize(z2_eval, dim=1)
                fusion_out.extend(torch.cat([z1_eval, z2_eval], dim=1).cpu().numpy())

            # Finalize outputs
            latent_fusion = np.array(fusion_out)
            labels1 = np.array(class_labels1)
            self.autoencoder.train()
            self.mutual_attention.train()
            self.projector.train()
        return latent_fusion, labels1

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
