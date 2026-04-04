import os 
import inspect
import importlib

import wget
import numpy as np
from scipy.stats import pearsonr
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter
import torch.nn.functional as F
from einops import rearrange

from models.module import GlobalEncoder, NeighborEncoder, FusionEncoder, FusionEncoder_concat 
from .builder import get_encoder

def compute_similarity_loss(sim_matrix1, sim_matrix2):
    # sim_matrix1, sim_matrix2: (N, N)
    mask = ~torch.eye(sim_matrix1.size(0), dtype=torch.bool, device=sim_matrix1.device)
    sim_matrix1_offdiag = sim_matrix1[mask]
    sim_matrix2_offdiag = sim_matrix2[mask]
    loss = F.mse_loss(sim_matrix1_offdiag, sim_matrix2_offdiag)
    return loss


def load_model_weights(path: str):       
        """Load pretrained ResNet18 model without final fc layer.

        Args:
            path (str): path_for_pretrained_weight

        Returns:
            torchvision.models.resnet.ResNet: ResNet model with pretrained weight
        """
        
        resnet = torchvision.models.__dict__['resnet18'](weights=None)
        
        ckpt_dir = './weights'
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = f'{ckpt_dir}/tenpercent_resnet18.ckpt'
        
        # prepare the checkpoint
        if not os.path.exists(ckpt_path):
            ckpt_url='https://github.com/ozanciga/self-supervised-histopathology/releases/download/tenpercent/tenpercent_resnet18.ckpt'
            wget.download(ckpt_url, out=ckpt_dir)
            
        state = torch.load(path)
        state_dict = state['state_dict']
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)
        
        model_dict = resnet.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        if state_dict == {}:
            print('No weight could be loaded..')
        model_dict.update(state_dict)
        resnet.load_state_dict(model_dict)
        resnet.fc = nn.Identity()

        return resnet
    
ENCODER_EMB_DIM = {
    'res18':512,
    'conch':512,
    'uni': 1024,
    'res50': 1024,
    'res101': 1024,
    'res152': 1024,
} 



class Img2Gene(pl.LightningModule):
    """Model class for Img2Gene"""
    
    def __init__(self, 
                 num_genes=250,
                 emb_dim=512,
                 depth1=2,
                 depth2=2,
                 depth3=2,
                 num_heads1=8,
                 num_heads2=8,
                 num_heads3=8,
                 mlp_ratio1=2.0,
                 mlp_ratio2=2.0,
                 mlp_ratio3=2.0,
                 dropout1=0.1,
                 dropout2=0.1,
                 dropout3=0.1,
                 kernel_size=3,
                 res_neighbor=(5, 5),
                 learning_rate=0.0001,
                 encoder="res50",
                 fusion='add', 
                 effect_type='NIE',
                 num_path=35,
                 model_name='testing',
                 max_batch_size=512, 
                 hyper_p = 1,
                 hyper_ip = 0.1,
                 hyper_ig = 1,
                 ):

        super().__init__()
        self.validation_step_outputs = []
        self.validation_corr = []
        """Img2Gene model 

        Args:
            num_genes (int): Number of genes to predict.
            emb_dim (int): Embedding dimension for images. Defaults to 512.
            depth1 (int): Depth of FusionEncoder. Defaults to 2.
            depth2 (int): Depth of GlobalEncoder. Defaults to 2.
            depth3 (int): Depth of NeighborEncoder. Defaults to 2.
            num_heads1 (int): Number of heads for FusionEncoder. Defaults to 8.
            num_heads2 (int): Number of heads for GlobalEncoder. Defaults to 8.
            num_heads3 (int): Number of heads for NeighborEncoder. Defaults to 8.
            mlp_ratio1 (float): mlp_ratio (MLP dimension/emb_dim) for FusionEncoder. Defaults to 2.0.
            mlp_ratio2 (float): mlp_ratio (MLP dimension/emb_dim) for GlobalEncoder. Defaults to 2.0.
            mlp_ratio3 (float): mlp_ratio (MLP dimension/emb_dim) for NeighborEncoder. Defaults to 2.0.
            dropout1 (float): Dropout rate for FusionEncoder. Defaults to 0.1.
            dropout2 (float): Dropout rate for GlobalEncoder. Defaults to 0.1.
            dropout3 (float): Dropout rate for NeighborEncoder. Defaults to 0.1.
            kernel_size (int): Kernel size of convolution layer in PEGH. Defaults to 3.
        """
        
        super().__init__()

        self.validation_step_outputs = []
        self.validation_corr = []

        
        emb_dim = ENCODER_EMB_DIM[encoder]
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        self.emb_dim = emb_dim
        self.max_batch_size = max_batch_size
        
        # Initialize best metrics
        self.best_loss = np.inf
        self.best_cor = -1
        self.average_ratio = 0.0005
        self.num_genes = num_genes
        self.alpha = 0.3
        self.num_n = res_neighbor[0]
        self.effect_type = effect_type
        self.num_pathways = num_path
        self.model_name = model_name
        self.lambda_p = hyper_p
        self.lambda_ig = hyper_ig
        self.lambda_ip = hyper_ip
        
        if encoder == "res18": 
            # Target Encoder
            resnet18 = load_model_weights("weights/tenpercent_resnet18.ckpt")
            module=list(resnet18.children())[:-2]
        elif encoder == "conch":
            model = get_encoder(encoder)  
            module=list(model.children())#[:-2]
        elif encoder == "uni":
            model = get_encoder(encoder)  
            module=list(model.children())#[:-2]
        elif encoder == "res50":
            model = get_encoder(encoder)  
            module=list(model.children())[:-1]
        elif encoder == "res101":
            model = get_encoder(encoder)  
            module=list(model.children())[:-2]
        elif encoder == "res152":
            model = get_encoder(encoder)  
            module=list(model.children())[:-2]

        self.target_encoder = nn.Sequential(*module)
        self.fc_target = nn.Linear(emb_dim, num_genes)

        # Neighbor Encoder
        self.neighbor_encoder = NeighborEncoder(emb_dim, depth3, num_heads3, int(emb_dim*mlp_ratio3), dropout = dropout3, resolution=res_neighbor)
        self.fc_neighbor = nn.Linear(emb_dim, num_genes)

        # Global Encoder        
        self.global_encoder = GlobalEncoder(emb_dim, depth2, num_heads2, int(emb_dim*mlp_ratio2), dropout2, kernel_size)
        self.fc_global = nn.Linear(emb_dim, num_genes)
        
        # Fusion Layer
        if fusion == "add":
            self.fusion_encoder = FusionEncoder(emb_dim, depth1, num_heads1, int(emb_dim*mlp_ratio1), dropout1)
            self.fc = nn.Linear(emb_dim, num_genes)
            self.pathway_fc = nn.Linear(emb_dim, self.num_pathways)
        elif fusion == 'concat':
            self.fusion_encoder = FusionEncoder_concat(emb_dim, depth1, num_heads1, int(emb_dim*mlp_ratio1),dropout1)
            self.fc = nn.Linear(emb_dim*2, num_genes)
            self.pathway_fc = nn.Linear(emb_dim*2, self.num_pathways)


        self.register_buffer("avg_target", torch.zeros(14*14,emb_dim))
        self.register_buffer("avg_neighbor", torch.zeros(5*5,emb_dim))
        self.register_buffer("avg_global", torch.zeros(1,emb_dim))


    def forward(self, x, x_total, position, neighbor, mask, pathway, pid=None, sid=None):
        """Forward pass of img2gene"""
        
        # Target tokens
        target_token = self.target_encoder(x) 
        B, dim, w, h = target_token.shape
        target_token = rearrange(target_token, 'b d h w -> b (h w) d', d=dim, w=w, h=h)
    
        # Neighbor tokens
        neighbor_token = self.neighbor_encoder(neighbor, mask) 
        
        # Global tokens
        if pid is None:
            global_token = self.global_encoder(x_total, position.squeeze()).squeeze()  
            if sid is not None:
                global_token = global_token[sid]
        else:
            pid = pid.view(-1)
            sid = sid.view(-1)
            global_token = torch.zeros((len(x_total), x_total[0].shape[1])).to(x.device)
            
            pid_unique = pid.unique()
            for pu in pid_unique:
                ind = int(torch.argmax((pid == pu).int()))
                x_g = x_total[ind].unsqueeze(0)
                pos = position[ind]
                
                emb = self.global_encoder(x_g, pos).squeeze() 
                global_token[pid == pu] = emb[sid[pid == pu]].float()
    
        # Fusion tokens
        fusion_token = self.fusion_encoder(target_token, neighbor_token, global_token, mask=mask) 

        if self.effect_type == 'None':
            logit = fusion_token  # B x 512
        else:
            if self.effect_type == 'TDE': # total direct effect
                with torch.no_grad():
                    avg_fusion = self.fusion_encoder(self.avg_target.unsqueeze(0).repeat(B, 1, 1), neighbor_token, global_token, mask=mask)
            elif self.effect_type == 'TIE': # total indirect
                with torch.no_grad():
                    avg_fusion = self.fusion_encoder(target_token, self.avg_neighbor.unsqueeze(0).repeat(B, 1, 1), self.avg_global.repeat(B, 1), mask=mask)
            elif self.effect_type == 'TE': # total effect
                with torch.no_grad():
                    avg_fusion = self.fusion_encoder(self.avg_target.unsqueeze(0).repeat(B, 1, 1), self.avg_neighbor.unsqueeze(0).repeat(B, 1, 1),self.avg_global.repeat(B, 1), mask=mask)
            elif self.effect_type == 'NDE': # natural/pure direct effect
                fusion_token = self.fusion_encoder(target_token, self.avg_neighbor.unsqueeze(0).repeat(B, 1, 1),self.avg_global.repeat(B, 1), mask=mask) 
                with torch.no_grad():
                    avg_fusion = self.fusion_encoder(self.avg_target.unsqueeze(0).repeat(B, 1, 1), self.avg_neighbor.unsqueeze(0).repeat(B, 1, 1),self.avg_global.repeat(B, 1), mask=mask)
            elif self.effect_type == 'NIE': # natural/pure indirect effect
                fusion_token = self.fusion_encoder(self.avg_target.unsqueeze(0).repeat(B, 1, 1), neighbor_token, global_token, mask=mask) 
                with torch.no_grad():
                    avg_fusion = self.fusion_encoder(self.avg_target.unsqueeze(0).repeat(B, 1, 1), self.avg_neighbor.unsqueeze(0).repeat(B, 1, 1),self.avg_global.repeat(B, 1), mask=mask)
            logit = fusion_token - avg_fusion

        output = self.fc(logit) # B x num_genes
        out_pathway = self.pathway_fc(logit)
        out_target = self.fc_target(target_token.mean(1)) # B x num_genes
        out_neighbor = self.fc_neighbor(neighbor_token.mean(1)) # B x num_genes
        out_global = self.fc_global(global_token) # B x num_genes

        return output, out_target, out_neighbor,out_global,out_pathway,logit

    
    def training_step(self, batch, batch_idx):
        """Train the model. Transfer knowledge from fusion to each module.

        """
        patch, exp, pid, sid, wsi, position, neighbor, mask, pathway  = batch
        
        outputs = self(patch, wsi, position, neighbor, mask, pathway, pid, sid)
        
        # Fusion loss
        loss = F.mse_loss(outputs[0].view_as(exp), exp)                   # Supervised loss for Fusion
        
        # Target loss
        loss += F.mse_loss(outputs[1].view_as(exp), exp) * (1-self.alpha) # Supervised loss for Target
        loss += F.mse_loss(outputs[0], outputs[1]) * self.alpha           # Distillation loss for Target
        
        # Neighbor loss
        loss += F.mse_loss(outputs[2].view_as(exp), exp) * (1-self.alpha) # Supervised loss for Neighbor
        loss += F.mse_loss(outputs[0], outputs[2]) * self.alpha           # Distillation loss for Neighbor
            
        # Global loss
        loss += F.mse_loss(outputs[3].view_as(exp), exp) * (1-self.alpha) # Supervised loss for Global
        loss += F.mse_loss(outputs[0], outputs[3]) * self.alpha           # Distillation loss for Global
            

        # for img feature
        image_norm = F.normalize(outputs[5], p=2, dim=1) # Normalize features
        image_similarity = torch.mm(image_norm, image_norm.t()) # Compute cosine similarity

        # for gene expression
        gene_centered = exp - exp.mean(dim=1, keepdim=True) # Center features
        gene_norm = F.normalize(gene_centered, p=2, dim=1) # Normalize features
        gene_similarity = torch.mm(gene_norm, gene_norm.t()) # Compute cosine similarity (equivalent to Pearson correlation for centered data)


        # for pathway data
        pathway_centered = pathway - pathway.mean(dim=1, keepdim=True) # Center features
        pathway_norm = F.normalize(pathway_centered, p=2, dim=1) # Normalize features
        pathway_similarity = torch.mm(pathway_norm, pathway_norm.t()) # Compute cosine similarity (equivalent to Pearson correlation for centered data)

        # pathway loss
        loss += F.mse_loss(outputs[4].view_as(pathway), pathway_norm) * self.lambda_p  # Supervised loss for pathway

        img_pathway_loss = compute_similarity_loss(image_similarity, pathway_similarity) 
        loss += img_pathway_loss * self.lambda_ip

        img_gene_loss = compute_similarity_loss(image_similarity, gene_similarity) 
        loss += img_gene_loss * self.lambda_ig

        self.log('train_loss', loss, sync_dist=True)
        return loss

    def forward_batch(self, patches, wsi, position, neighbors, masks, pathways, sids=None):
        """Forward pass for a batch of patches, handling large samples by splitting into smaller batches.
        
        Args:
            patches: (N, 3, H, W) - all patches for one sample
            wsi: global features
            position: position information
            neighbors: (N, num_neighbors, dim) - neighbor features
            masks: (N, num_neighbors) - masking table
            pathways: (N, num_pathways) - pathway data
            sids: optional spot indices
            
        Returns:
            predictions: (N, num_genes)
        """
        n_patches = patches.shape[0]
        
        if n_patches <= self.max_batch_size:
            # Process all at once
            outputs = self(patches, wsi, position, neighbors, masks, pathways, sid=sids)
            return outputs[0]
        
        # Split into smaller batches
        all_preds = []
        
        for start_idx in range(0, n_patches, self.max_batch_size):
            end_idx = min(start_idx + self.max_batch_size, n_patches)
            
            batch_patches = patches[start_idx:end_idx]
            batch_neighbors = neighbors[start_idx:end_idx]
            batch_masks = masks[start_idx:end_idx]
            batch_pathways = pathways[start_idx:end_idx]
            batch_sids = torch.arange(start_idx, end_idx).to(patches.device) if sids is None else sids[start_idx:end_idx]
            
            outputs = self(batch_patches, wsi, position, batch_neighbors, batch_masks, batch_pathways, sid=batch_sids)
            all_preds.append(outputs[0])
        
        return torch.cat(all_preds, dim=0)

    def validation_step(self, batch, batch_idx):
        """Validating the model in a sample. Calucate MSE and PCC for all spots in the sample.

        Returns:
            dict: 
                val_loss: MSE loss between pred and label
                corr: PCC between pred and label (across genes)
        """
       # patch, exp, pid, sid, wsi, position, neighbor, mask = batch
        patch, exp, _, wsi, position, name, neighbor, mask, pathway = batch

        # Squeeze batch dimension (batch_size=1 for validation)
        patch = patch.squeeze(0)      # (N, 3, H, W)
        exp = exp.squeeze(0)          # (N, num_genes)
        neighbor = neighbor.squeeze(0) # (N, num_neighbors, dim)
        mask = mask.squeeze(0)        # (N, num_neighbors)
        pathway = pathway.squeeze(0)  # (N, num_pathways)
 
        # Use forward_batch to handle large samples
        pred = self.forward_batch(patch, wsi, position, neighbor, mask, pathway)

        # Compute loss
        loss = F.mse_loss(pred, exp)

        # Compute correlation
        pred=pred.cpu().numpy().T
        exp=exp.cpu().numpy().T    

        r = []
        for g in range(self.num_genes):
            try:
                corr = pearsonr(pred_np[g], exp_np[g])[0]
                if np.isnan(corr):
                    corr = 0.0
            except:
                corr = 0.0
            r.append(corr)
        rr = torch.Tensor(r)
        
        self.get_meta(name)
        self.validation_step_outputs.append(loss)
        self.validation_corr.append(rr)
        return {"val_loss":loss, "corr":rr}
    

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_step_outputs).mean()
        avg_corr = torch.stack(self.validation_corr)

        os.makedirs(f"results/{self.model_name}/validation", exist_ok=True)
        if self.best_cor < avg_corr.mean():
            torch.save(avg_corr.cpu(), f"results/{self.model_name}/validation/R_{self.patient}")
            torch.save(avg_loss.cpu(), f"results/{self.model_name}/validation/loss_{self.patient}")
            
            self.best_cor = avg_corr.mean()
            self.best_loss = avg_loss
        
        self.log('valid_loss', avg_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('R', avg_corr.nanmean(), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.validation_step_outputs.clear()  # free memory
        self.validation_corr.clear() # free memory

    def test_step(self, batch, batch_idx):
        """Testing the model in a sample. """

        patch, exp, _, wsi, position, name, neighbor, mask, pathway = batch

        # Squeeze batch dimension
        patch = patch.squeeze(0)
        exp = exp.squeeze(0)
        neighbor = neighbor.squeeze(0)
        mask = mask.squeeze(0)
        pathway = pathway.squeeze(0)
        
        # Handle special cases
        if isinstance(name, (list, tuple)):
            sample_name = name[0]
        else:
            sample_name = name
            
        if '10x_breast' in sample_name:
            wsi = wsi[0].unsqueeze(0)
            position = position[0]
            
            # Use forward_batch for large samples
            pred = self.forward_batch(patch, wsi, position, neighbor, mask, pathway)
            
            ind_match = np.load(f'/home/wzhang/st/data/test/{sample_name}/ind_match.npy', allow_pickle=True)
            self.num_genes = len(ind_match)
            pred = pred[:, ind_match]
        else:        
            pred = self.forward_batch(patch, wsi, position, neighbor, mask, pathway)
            
        mse = F.mse_loss(pred.view_as(exp), exp)
        mae = F.l1_loss(pred.view_as(exp), exp)
        
        pred=pred.cpu().numpy().T
        exp=exp.cpu().numpy().T
        
        r = []
        for g in range(self.num_genes):
            try:
                corr = pearsonr(pred_np[g], exp_np[g])[0]
                if np.isnan(corr):
                    corr = 0.0
            except:
                corr = 0.0
            r.append(corr)
        rr = torch.Tensor(r)
        
        self.get_meta(name)
        
        os.makedirs(f"results/{self.model_name}/{self.data}/{self.patient}", exist_ok=True)
        np.save(f"results/{self.model_name}/{self.data}/{self.patient}/{name[0]}", pred.T)
        np.save(f"results/{self.model_name}/{self.data}/{self.patient}/{name[0]}_exp", exp.T)
        return {"MSE":mse, "MAE":mae, "corr":rr}
    
    def test_epoch_end(self, outputs):
        avg_mse = torch.stack([x["MSE"] for x in outputs]).nanmean()
        avg_mae = torch.stack([x["MAE"] for x in outputs]).nanmean()
        avg_corr = torch.stack([x["corr"] for x in outputs]).nanmean(0)

        os.makedirs(f"results/{self.model_name}/{self.data}/{self.patient}", exist_ok=True)
        torch.save(avg_mse.cpu(), f"results/{self.model_name}/{self.data}/{self.patient}/MSE")
        torch.save(avg_mae.cpu(), f"results/{self.model_name}/{self.data}/{self.patient}/MAE")
        torch.save(avg_corr.cpu(), f"results/{self.model_name}/{self.data}/{self.patient}/cor")
        
    def predict_step(self, batch, batch_idx):
        patches, sids, wsi, position, neighbors, masks, pathway = batch
        
        patches = patches.squeeze(0)
        sids = sids.squeeze(0)
        neighbors = neighbors.squeeze(0)
        masks = masks.squeeze(0)
        pathway = pathway.squeeze(0)

        # Use forward_batch for prediction
        preds = self.forward_batch(patches, wsi, position, neighbors, masks, pathway, sids)
        
        return preds
    
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        StepLR = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.9)
        optim_dict = {'optimizer': optim, 'lr_scheduler': StepLR}
        return optim_dict
    
    def get_meta(self, name):
        if '10x_breast' in name[0]:
            self.patient = name[0]
            self.data = "test"
        else:
            name = name[0]
            self.data = name.split("+")[1]
            self.patient = name.split("+")[0]
            
            if self.data == 'her2st':
                self.patient = self.patient[0]
            elif self.data == 'stnet':
                self.data = "stnet"
                patient = self.patient.split('_')[0]
                if patient in ['BC23277', 'BC23287', 'BC23508']:
                    self.patient = 'BC1'
                elif patient in ['BC24105', 'BC24220', 'BC24223']:
                    self.patient = 'BC2'
                elif patient in ['BC23803', 'BC23377', 'BC23895']:
                    self.patient = 'BC3'
                elif patient in ['BC23272', 'BC23288', 'BC23903']:
                    self.patient = 'BC4'
                elif patient in ['BC23270', 'BC23268', 'BC23567']:
                    self.patient = 'BC5'
                elif patient in ['BC23269', 'BC23810', 'BC23901']:
                    self.patient = 'BC6'
                elif patient in ['BC23209', 'BC23450', 'BC23506']:
                    self.patient = 'BC7'
                elif patient in ['BC23944', 'BC24044']:
                    self.patient = 'BC8'
            elif self.data == 'skin':
                self.patient = self.patient.split('_')[0]
    
    def load_model(self):
        name = self.hparams.MODEL.name
        if '_' in name:
            camel_name = ''.join([i.capitalize() for i in name.split('_')])
        else:
            camel_name = name
        try:
            Model = getattr(importlib.import_module(
                f'models.{name}'), camel_name)
        except:
            raise ValueError('Invalid Module File Name or Invalid Class Name!')
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.MODEL.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams.MODEL, arg)
        args1.update(other_args)
        return Model(**args1)
      
class CustomWriter(BasePredictionWriter):
    def __init__(self, pred_dir, write_interval, emb_dir=None, names=None):
        super().__init__(write_interval)
        self.pred_dir = pred_dir
        self.emb_dir = emb_dir
        self.names = names

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        for i, batch in enumerate(batch_indices[0]):
            torch.save(predictions[0][i][0], os.path.join(self.pred_dir, f"{self.names[i]}.pt"))
            torch.save(predictions[0][i][1], os.path.join(self.emb_dir, f"{self.names[i]}.pt"))

        # optionally, you can also save `batch_indices` to get the information about the data index
        # from your prediction data
        # torch.save(batch_indices, os.path.join(self.output_dir, f"batch_indices_{trainer.global_rank}.pt"))


