import os
import torch
import numpy as np
import argparse
import pandas as pd
from scipy.stats import pearsonr


def list_sorted_subfolders(folder_path):
    """List all subfolders in a directory and sort them by name."""
    subfolders = [f.name for f in os.scandir(folder_path) if f.is_dir()]
    return [os.path.join(folder_path, i) for i in sorted(subfolders)]

def patient_results(path):
    """Load MAE, MSE, and correlation results from a given path."""
    mae = torch.load(os.path.join(path, 'MAE'))
    mse = torch.load(os.path.join(path, 'MSE'))
    cor = torch.load(os.path.join(path, 'cor'))  # shape: (num_genes,)
    return mae, mse, cor

def load_all_label_data(label_dir):
    data_list = []
    for file in sorted(os.listdir(label_dir)):
        if file.endswith('.npy'):
            data = np.load(os.path.join(label_dir, file))  # shape: (n, num_genes)
            data_list.append(data)
    return np.concatenate(data_list, axis=0)  # shape: (N_total, num_genes)

def get_top_heg_indices(data, top_k=50):
    gene_means = np.mean(data, axis=0)
    return np.argsort(gene_means)[-top_k:][::-1]


def get_top_hvg_indices(data, top_k=50):
    data = data / (np.sum(data, axis=1, keepdims=True) + 1e-8)  # normalize each spot
    gene_vars = np.var(data, axis=0)
    return np.argsort(gene_vars)[-top_k:][::-1]

def metrics(dataset, result_dir, label_dir, gene_names_file, output_csv, top_k=50):
    """Calculate and save evaluation metrics."""

    # Get all subfolders in the output directory
    res_subfolders = list_sorted_subfolders(result_dir)

    if len(res_subfolders) == 0:
        print(f"No result subfolders found in {result_dir}")
        return

    mae_list, mse_list, pcc_all = [], [], []

    for path in res_subfolders:
        try:
            mae, mse, cor = patient_results(path)
            mae_list.append(mae)
            mse_list.append(mse)
            pcc_all.append(cor.numpy() if torch.is_tensor(cor) else cor)
        except Exception as e:
            print(f"Error loading results from {path}: {e}")
            continue

    if len(pcc_all) == 0:
        print("No valid results found")
        return

    pcc_all = np.stack(pcc_all)  # shape: (num_patients, num_genes)
    num_genes = pcc_all.shape[1]
    
    print(f"Loaded {len(pcc_all)} samples, {num_genes} genes")

    actual_top_k = min(top_k, num_genes)
    if actual_top_k < top_k:
        print(f"Adjusting top_k from {top_k} to {actual_top_k} (num_genes={num_genes})")

    # HPG: highly predictive genes
    rank_sum = np.zeros(num_genes)
    for cor in pcc_all:
        ranks = np.argsort(cor)
        for i, idx in enumerate(ranks):
            rank_sum[idx] += i
    hpg_indices = np.argsort(rank_sum)[-actual_top_k:]

    # HEG: highly expressed genes
    all_label_data = load_all_label_data(label_dir)
    heg_indices = get_top_heg_indices(all_label_data, top_k=actual_top_k)

    # HVG: highly variable genes
    hvg_indices = get_top_hvg_indices(all_label_data, top_k=actual_top_k)

    # Compute means and stds
    mae_mean, mae_std = np.mean(mae_list), np.std(mae_list)
    mse_mean, mse_std = np.mean(mse_list), np.std(mse_list)

    pcc_all_mean = np.mean(np.nanmean(pcc_all, axis=1))
    pcc_all_std = np.std(np.nanmean(pcc_all, axis=1))

    pcc_hpg_mean = np.mean(np.nanmean(pcc_all[:, hpg_indices], axis=1))
    pcc_hpg_std = np.std(np.nanmean(pcc_all[:, hpg_indices], axis=1))

    pcc_heg_mean = np.mean(np.nanmean(pcc_all[:, heg_indices], axis=1))
    pcc_heg_std = np.std(np.nanmean(pcc_all[:, heg_indices], axis=1))

    pcc_hvg_mean = np.mean(np.nanmean(pcc_all[:, hvg_indices], axis=1))
    pcc_hvg_std = np.std(np.nanmean(pcc_all[:, hvg_indices], axis=1))

    
    df = pd.DataFrame([{
        "dataset": dataset,
        "num_samples": len(pcc_all),
        "num_genes": num_genes,
        "top_k": actual_top_k,
        "mae_mean": round(mae_mean, 3),
        "mae_std": round(mae_std, 2),
        "mse_mean": round(mse_mean, 3),
        "mse_std": round(mse_std, 2),
        "pcc_all_mean": round(pcc_all_mean, 3),
        "pcc_all_std": round(pcc_all_std, 2),
        "pcc_hpg_mean": round(pcc_hpg_mean, 3),
        "pcc_hpg_std": round(pcc_hpg_std, 2),
        "pcc_hvg_mean": round(pcc_hvg_mean, 3),
        "pcc_hvg_std": round(pcc_hvg_std, 2),
        "pcc_heg_mean": round(pcc_heg_mean, 3),
        "pcc_heg_std": round(pcc_heg_std, 2),
    }])
    df.to_csv(output_csv, index=False)
    print(f"✅ CSV saved to: {output_csv}")

    print(f'\ndataset: {dataset}')
    print(f"num_samples: {len(pcc_all)}, num_genes: {num_genes}, top_k: {actual_top_k}")
    print("MAE mean: %.3f, std: %.2f" % (mae_mean, mae_std))
    print("MSE mean: %.3f, std: %.2f" % (mse_mean, mse_std))
    print("PCC ALL mean: %.3f, std: %.2f" % (pcc_all_mean, pcc_all_std))
    print("PCC HPG mean: %.3f, std: %.2f" % (pcc_hpg_mean, pcc_hpg_std))
    print("PCC HEG mean: %.3f, std: %.2f" % (pcc_heg_mean, pcc_heg_std))
    print("PCC HVG mean: %.3f, std: %.2f" % (pcc_hvg_mean, pcc_hvg_std))

def eval_model(dataset, dir_model, model_name, num_path, encoder='res50', fusion="add", effect_type='TE', gpu=0, overwrite=False,hyper_p=1,hyper_ip=1,hyper_ig=0.1):

    FOLD_DICT = {'her2st': 8, 'skin': 4, 'stnet': 8, 'gse': 4}

    if dataset not in FOLD_DICT:
        print(f"Warning: dataset '{dataset}' not in FOLD_DICT, using default 4 folds")
        num_folds = 4
    else:
        num_folds = FOLD_DICT[dataset]

    for fold in range(num_folds):
        cur_dir = os.path.join(dir_model, model_name, f'{dataset}_fold_{fold}')
        
        if not os.path.exists(cur_dir):
            print(f"Warning: {cur_dir} does not exist, skipping...")
            continue
            
        ckpt_files = [f for f in os.listdir(cur_dir) if f.endswith('.ckpt')]
        
        if len(ckpt_files) == 0:
            print(f"Warning: No .ckpt files found in {cur_dir}, skipping...")
            continue
            
        for f in ckpt_files:
            cmd = f"python main.py --config {dataset}/Img2Gene --mode test --fold {fold} --model_path " + \
                      f"{os.path.join(cur_dir, f)} --encoder {encoder} --fusion {fusion} --gpu {gpu} --effect_type {effect_type} --model_name {model_name} --num_path {num_path}  --hyper_ig {hyper_ig} --hyper_p {hyper_p} --hyper_ip {hyper_ip}"
            print(cmd)
            os.system(cmd)

if __name__ == '__main__':

    # Parsing the command-line arguments
    parser = argparse.ArgumentParser(description='evaluate model by given parameters')
    parser.add_argument('--dataset', type=str, required=True, choices=['her2st', 'skin', 'stnet', 'gse'], help='datasets')
    parser.add_argument('--dir_model', type=str, default='logs/', help='dir of checkpoints')
    parser.add_argument('--encoder', type=str, default='res50', help='encoder')
    parser.add_argument('--fusion', type=str, default='add', help='fusion')
    parser.add_argument('--effect_type', type=str, default='TE', help='effect type')
    parser.add_argument('--gpu', type=int, default=7, help='gpu id')
    parser.add_argument('--overwrite', action='store_true', help='overwrite existing results')
    parser.add_argument('--model_name', type=str, help='Model name')
    parser.add_argument('--num_path', type=int, default=35, help='herst: 50, skin: 35, stnet: 37, gse: 10')
    parser.add_argument('--num_folds', type=int, default=None, help='Number of folds for cross-validation (overrides default)')
    parser.add_argument('--top_k', type=int, default=50, help='Top k genes for HPG/HEG/HVG calculation')
    parser.add_argument('--hyper_p', type=float, default=1, help='weight of pathway loss')
    parser.add_argument('--hyper_ip', type=float, default=1, help='weight of image pathway loss')
    parser.add_argument('--hyper_ig', type=float, default=0.1, help='weight of image gene loss')
    args = parser.parse_args()
     
    eval_model(dataset=args.dataset,
               dir_model=args.dir_model,
               model_name=args.model_name, 
               num_path=args.num_path, 
               encoder=args.encoder,
               fusion=args.fusion, 
               effect_type=args.effect_type,
               gpu=args.gpu, 
               overwrite=args.overwrite,
               hyper_p=args.hyper_p,
               hyper_ip=args.hyper_ip,
               hyper_ig=args.hyper_ig
               )
     
    # ------------------------- metric calculation ------------------------- #
    if args.dataset == "her2st":
        label_dir = f"/home/wzhang/st/jbhi/label/her2st_label"
        gene_name_file = f"/home/wzhang/data/st/data_uni/her2st/genes_her2st.npy"
    elif args.dataset == "skin":
        label_dir = f"/home/wzhang/st/jbhi/label/skin_label"
        gene_name_file = f"/home/wzhang/data/st/data_uni/skin/genes_skin.npy"
    elif args.dataset == "stnet":
        label_dir = f"/home/wzhang/st/jbhi/label/stnet_label"
        gene_name_file = f"/home/wzhang/data/st/data_uni/stnet/genes_stnet.npy"
    elif args.dataset == "gse":
        label_dir = f"/home/wzhang/st/jbhi/label/gse240429_label_250"
        gene_name_file = f"/home/wzhang/data/st/GSE240429_output/genes_240429.npy"
        
        if not os.path.exists(label_dir):
            print(f"Warning: label_dir {label_dir} does not exist!")
            print("You may need to create label files first.")
            print("Alternatively, set the correct path for your GSE dataset.")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    result_dir = os.path.join('results', args.model_name, args.dataset)  
    
    if not os.path.exists(result_dir):
        print(f"Warning: result_dir {result_dir} does not exist!")
        print("Please run evaluation first.")
    else:
        # the csv file that save all results
        output_csv = f"{args.dataset}_{args.model_name}.csv" 
        
        # Calculate and save metrics
        metrics(dataset=args.dataset,
                result_dir=result_dir,
                label_dir=label_dir,
                gene_names_file=gene_name_file,
                output_csv=output_csv,
                top_k=args.top_k)
