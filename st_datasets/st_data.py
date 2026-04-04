import os
from glob import glob
import warnings
warnings.filterwarnings('ignore')

import pickle
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from PIL import ImageFile, Image
import torch
import torchvision
import torchvision.transforms as transforms
import scprep as scp

from utils import smooth_exp

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# Optional imports for new-format datasets (gse)
try:
    from scipy import sparse
    import scanpy as sc
    import h5py
except ImportError:
    sparse = None
    sc = None
    h5py = None

'''
Notice that this is a rewritten version to integrate the data format for [her2st, skin, stnet] from the triplex, denoted as old,
and the new data format for gse240429, denoted as new.
'''

# ---------------------------------------------------------------------------
# Datasets that use the new h5ad / h5 based format.
# Add new dataset names here when they follow the same directory layout as gse.
# ---------------------------------------------------------------------------
NEW_FORMAT_DATASETS = {'gse'}


def normalize_adata(adata, cpm=False):
    """Normalize an AnnData object (library-size + log1p)."""
    adata = adata.copy()
    if cpm:
        sc.pp.normalize_total(adata, target_sum=1e6)
    else:
        sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    return adata


# ═══════════════════════════════════════════════════════════════════════════
#  Base class
# ═══════════════════════════════════════════════════════════════════════════
class BaselineDataset(torch.utils.data.Dataset):
    """Base class that defines shared image transforms."""

    def __init__(self):
        super(BaselineDataset, self).__init__()

        self.train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomApply(
                [torchvision.transforms.RandomRotation((90, 90))]
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

        self.test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])


# ═══════════════════════════════════════════════════════════════════════════
#  Unified STDataset
# ═══════════════════════════════════════════════════════════════════════════
class STDataset(BaselineDataset):
    """Unified ST Dataset that supports both old-format (skin / her2st / stnet)
    and new-format (gse) datasets.

    The format is determined automatically by ``kwargs['type']``.
    """

    def __init__(
        self,
        mode: str,
        fold: int = 0,
        extract_mode: str = None,
        test_data=None,
        **kwargs,
    ):
        super().__init__()

        self.mode = mode
        self.fold = fold
        self.extract_mode = extract_mode
        self.data = kwargs.get('type', 'default')
        self.is_new_format = self.data in NEW_FORMAT_DATASETS

        if self.is_new_format:
            if sc is None or h5py is None:
                raise ImportError(
                    "scanpy and h5py are required for new-format datasets "
                    "(e.g. gse).  Install them with:  pip install scanpy h5py"
                )
            self._init_new_format(**kwargs)
        else:
            self._init_old_format(**kwargs)

    # ===================================================================
    #  initialisation  (skin / her2st / stnet)
    # ===================================================================
    def _init_old_format(self, **kwargs):
        self.gt_dir = kwargs['t_global_dir']
        self.num_neighbors = kwargs['num_neighbors']
        self.neighbor_dir = f"{kwargs['neighbor_dir']}_{self.num_neighbors}_224"
        self.use_pyvips = kwargs['use_pyvips']
        self.r = kwargs['radius'] // 2

        self.data_dir = f"{kwargs['data_dir']}/{kwargs['type']}"

        # --- sample names ---------------------------------------------------
        names = os.listdir(self.data_dir + '/ST-spotfiles')
        names.sort()
        names = [i.split('_selection.tsv')[0] for i in names]

        if self.mode in ["external_test", "inference"]:
            self.names = names

        elif self.mode == "extraction":
            self.names = names
            if self.extract_mode == "neighbor":
                self.names = [
                    n for n in self.names
                    if not os.path.exists(
                        os.path.join(self.neighbor_dir, f"{n}.pt")
                    )
                ]
            elif self.extract_mode == "target":
                self.names = [
                    n for n in self.names
                    if not os.path.exists(
                        os.path.join(self.gt_dir, f"{n}.pt")
                    )
                ]
        else:
            # cross-validation splits
            if self.data == 'stnet':
                kf = KFold(8, shuffle=True, random_state=2021)
                patients = np.array([
                    'BC23209', 'BC23270', 'BC23803', 'BC24105', 'BC24220',
                    'BC23268', 'BC23269', 'BC23272', 'BC23277', 'BC23287',
                    'BC23288', 'BC23377', 'BC23450', 'BC23506', 'BC23508',
                    'BC23567', 'BC23810', 'BC23895', 'BC23901', 'BC23903',
                    'BC23944', 'BC24044', 'BC24223',
                ])
                _, ind_val = [i for i in kf.split(patients)][self.fold]
                patients_val = patients[ind_val]
                te_names = []
                for pp in patients_val:
                    te_names += [i for i in names if pp in i]

            elif self.data == 'her2st':
                patients = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
                te_names = [i for i in names if patients[self.fold] in i]

            elif self.data == 'skin':
                patients = ['P2', 'P5', 'P9', 'P10']
                te_names = [i for i in names if patients[self.fold] in i]
            else:
                raise ValueError(
                    f"Unknown old-format dataset: {self.data}. "
                    f"Add CV split logic or add it to NEW_FORMAT_DATASETS."
                )

            tr_names = list(set(names) - set(te_names))
            self.names = tr_names if self.mode == 'train' else te_names

        # --- load images -----------------------------------------------------
        if self.use_pyvips:
            self.img_dict = {i: self._get_img_old(i) for i in self.names}
            with open(f"{self.data_dir}/slide_shape.pickle", "rb") as f:
                self.img_shape_dict = pickle.load(f)
        else:
            self.img_dict = {
                i: np.array(self._get_img_old(i)) for i in self.names
            }

        # --- pathway ---------------------------------------------------------
        self.pathway_dict = {i: self._get_pathway_old(i) for i in self.names}

        # --- meta (position + counts) ----------------------------------------
        self.meta_dict = {i: self._get_meta_old(i) for i in self.names}

        # --- expression ------------------------------------------------------
        if self.mode not in ["extraction", "inference"]:
            gene_list = list(np.load(
                self.data_dir + f'/genes_{self.data}.npy', allow_pickle=True
            ))
            self.exp_dict = {
                i: scp.transform.log(
                    scp.normalize.library_size_normalize(m[gene_list])
                )
                for i, m in self.meta_dict.items()
            }
            self.exp_dict = {
                i: smooth_exp(m).values for i, m in self.exp_dict.items()
            }

        # --- centres / locations ---------------------------------------------
        if self.mode == "external_test":
            self.center_dict = {
                i: np.floor(m[['pixel_y', 'pixel_x']].values).astype(int)
                for i, m in self.meta_dict.items()
            }
            self.loc_dict = {
                i: m[['x', 'y']].values for i, m in self.meta_dict.items()
            }
        else:
            self.center_dict = {
                i: np.floor(m[['pixel_x', 'pixel_y']].values).astype(int)
                for i, m in self.meta_dict.items()
            }
            self.loc_dict = {
                i: m[['y', 'x']].values for i, m in self.meta_dict.items()
            }

        self.lengths = [len(v) for v in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))

    # ===================================================================
    #  data-loading helpers
    # ===================================================================
    def _get_img_old(self, name):
        img_dir = self.data_dir + '/ST-imgs'
        if self.data == 'her2st':
            pre = img_dir + '/' + name[0] + '/' + name
            fig_name = os.listdir(pre)[0]
            path = pre + '/' + fig_name
        elif self.data == 'stnet' or '10x_breast' in self.data:
            path = glob(img_dir + '/*' + name + '.tif')[0]
        elif 'DRP' in self.data:
            path = glob(img_dir + '/*' + name + '.svs')[0]
        else:
            path = glob(img_dir + '/*' + name + '.jpg')[0]

        if self.use_pyvips:
            import pyvips as pv
            im = pv.Image.new_from_file(path, level=0)
        else:
            im = Image.open(path)
        return im

    def _get_cnt_old(self, name):
        path = self.data_dir + '/ST-cnts/' + name + '_sub.parquet'
        return pd.read_parquet(path)

    def _get_pos_old(self, name):
        path = self.data_dir + '/ST-spotfiles/' + name + '_selection.tsv'
        df = pd.read_csv(path, sep='\t')
        x = np.around(df['x'].values).astype(int)
        y = np.around(df['y'].values).astype(int)
        df['id'] = [str(x[i]) + 'x' + str(y[i]) for i in range(len(x))]
        return df

    def _get_pathway_old(self, name):
        path = (
            self.data_dir + '/ST-pathways/' + name + '_sub_pathway.parquet'
        )
        return pd.read_parquet(path)

    def _get_meta_old(self, name):
        pos = self._get_pos_old(name)
        if 'DRP' not in self.data:
            cnt = self._get_cnt_old(name)
            meta = cnt.join(pos.set_index('id'), how='inner')
        else:
            meta = pos
        if self.mode == "external_test":
            meta = meta.sort_values(['x', 'y'])
        else:
            meta = meta.sort_values(['y', 'x'])
        return meta

    # ===================================================================
    #  masking / pyvips helpers
    # ===================================================================
    def make_masking_table(self, x, y, img_shape):
        mask_tb = torch.ones(self.num_neighbors ** 2)

        def create_mask(ind, mask_tb, window):
            if y - self.r * window < 0:
                mask_tb[
                    self.num_neighbors * ind
                    : self.num_neighbors * ind + self.num_neighbors
                ] = 0
            if y + self.r * window > img_shape[0]:
                mask_tb[
                    (self.num_neighbors ** 2 - self.num_neighbors * (ind + 1))
                    : (self.num_neighbors ** 2 - self.num_neighbors * ind)
                ] = 0
            if x - self.r * window < 0:
                mask = [
                    i + ind
                    for i in range(self.num_neighbors ** 2)
                    if i % self.num_neighbors == 0
                ]
                mask_tb[mask] = 0
            if x + self.r * window > img_shape[1]:
                mask = [
                    i - ind
                    for i in range(self.num_neighbors ** 2)
                    if i % self.num_neighbors == (self.num_neighbors - 1)
                ]
                mask_tb[mask] = 0
            return mask_tb

        ind = 0
        window = self.num_neighbors
        while window >= 3:
            mask_tb = create_mask(ind, mask_tb, window)
            ind += 1
            window -= 2
        return mask_tb

    def extract_patches_pyvips(self, slide, x, y, img_shape):
        tile_size = self.r * 2
        expansion_size = tile_size * self.num_neighbors
        padding_color = 255

        x_lt = x - tile_size * 2
        y_lt = y - tile_size * 2
        x_rd = x + tile_size * 3
        y_rd = y + tile_size * 3

        x_left_pad = max(0, -x_lt)
        x_right_pad = max(0, x_rd - img_shape[1])
        y_up_pad = max(0, -y_lt)
        y_down_pad = max(0, y_rd - img_shape[0])

        x_lt = max(x_lt, 0)
        y_lt = max(y_lt, 0)
        width = min(x_rd, img_shape[1]) - x_lt
        height = min(y_rd, img_shape[0]) - y_lt

        im = slide.extract_area(x_lt, y_lt, width, height)
        im = np.array(im)[:, :, :3]

        if x_left_pad or x_right_pad or y_up_pad or y_down_pad:
            padded_image = np.full(
                (expansion_size, expansion_size, 3), padding_color, dtype='uint8'
            )
            padded_image[
                y_up_pad : y_up_pad + height, x_left_pad : x_left_pad + width
            ] = im
            image = padded_image
        else:
            image = im
        return image

    # ===================================================================
    #  initialisation  (gse)
    # ===================================================================
    def _init_new_format(self, **kwargs):
        self.data_dir = kwargs.get('data_dir', '')
        self.model_name = kwargs.get('model_name', 'uni_v1')

        self.neighbor_dir = f"{self.data_dir}/emb/neighbor/{self.model_name}"
        self.global_dir = f"{self.data_dir}/emb/global/{self.model_name}"
        self.adata_dir = f"{self.data_dir}/adata"
        self.patches_dir = f"{self.data_dir}/patches"
        self.splits_dir = f"{self.data_dir}/splits"
        self.pathway_dir = kwargs.get(
            'pathway_dir', f"{self.data_dir}/pathway"
        )

        self.r = kwargs.get('radius', 224) // 2
        self.num_neighbors = kwargs.get('num_neighbors', 5)

        self.gene_type = kwargs.get('gene_type', 'mean')
        self.num_genes = kwargs.get('num_genes', 1000)
        self.num_outputs = kwargs.get('num_outputs', 250)
        self.num_pathways = kwargs.get('num_path', 35)
        self.pathway_key = kwargs.get('pathway_key', 'aucell_scores')

        # gene list
        gene_path = f"{self.data_dir}/{self.gene_type}_{self.num_genes}genes.json"
        if os.path.isfile(gene_path):
            with open(gene_path, 'r') as f:
                self.genes = json.load(f)['genes']
            if self.gene_type == 'mean':
                self.genes = self.genes[: self.num_outputs]
        else:
            raise FileNotFoundError(f"Gene list file not found: {gene_path}")

        self.normalize = kwargs.get('normalize', True)
        self.cpm = kwargs.get('cpm', False)
        self.smooth = kwargs.get('smooth', False)

        # sample names
        self._load_sample_names_new()

        # data
        self._load_data_new()

    # ===================================================================
    #  GSE FORMAT  –  sample-name helpers
    # ===================================================================
    def _load_sample_names_new(self):
        if self.mode in ["external_test", "inference"]:
            test_path = f"{self.splits_dir}/test_{self.fold}.csv"
            if os.path.isfile(test_path):
                self.names = pd.read_csv(test_path)['sample_id'].tolist()
            else:
                self.names = [
                    f.split('.')[0]
                    for f in os.listdir(self.patches_dir)
                    if f.endswith('.h5')
                ]

        elif self.mode == "extraction":
            all_names = [
                f.split('.')[0]
                for f in os.listdir(self.patches_dir)
                if f.endswith('.h5')
            ]
            self.names = all_names
            if self.extract_mode == "neighbor":
                self.names = [
                    n for n in self.names
                    if not os.path.exists(f"{self.neighbor_dir}/{n}.h5")
                ]
            elif self.extract_mode == "target":
                self.names = [
                    n for n in self.names
                    if not os.path.exists(f"{self.global_dir}/{n}.h5")
                ]
        else:
            train_path = f"{self.splits_dir}/train_{self.fold}.csv"
            test_path = f"{self.splits_dir}/test_{self.fold}.csv"
            if os.path.isfile(train_path) and os.path.isfile(test_path):
                tr_names = pd.read_csv(train_path)['sample_id'].tolist()
                te_names = pd.read_csv(test_path)['sample_id'].tolist()
            else:
                raise FileNotFoundError(
                    f"Split files not found: {train_path} or {test_path}"
                )
            self.names = tr_names if self.mode == 'train' else te_names

        self.id2name = dict(enumerate(self.names))

    # ===================================================================
    #  GSE FORMAT  –  bulk data loading
    # ===================================================================
    def _load_data_new(self):
        self.adata_dict = {}
        self.exp_dict = {}
        self.center_dict = {}
        self.loc_dict = {}
        self.pathway_dict = {}
        self.img_dict = {}

        for name in self.names:
            adata = self._load_adata(name)
            self.adata_dict[name] = adata

            # expression
            if self.mode not in ["extraction", "inference"]:
                exp = (
                    adata.X.toarray() if sparse.issparse(adata.X) else adata.X
                )
                if self.smooth:
                    exp_df = pd.DataFrame(
                        exp, index=adata.obs_names, columns=adata.var_names
                    )
                    exp_df = smooth_exp(exp_df)
                    exp = exp_df.values
                self.exp_dict[name] = exp

            # positions
            if 'spatial' in adata.obsm:
                positions = adata.obsm['spatial']
            elif (
                'pixel_x' in adata.obs.columns
                and 'pixel_y' in adata.obs.columns
            ):
                positions = adata.obs[['pixel_x', 'pixel_y']].values
            elif (
                'array_row' in adata.obs.columns
                and 'array_col' in adata.obs.columns
            ):
                positions = adata.obs[['array_row', 'array_col']].values
            else:
                positions = np.zeros((len(adata), 2))

            self.center_dict[name] = positions.astype(int)

            if (
                'array_row' in adata.obs.columns
                and 'array_col' in adata.obs.columns
            ):
                self.loc_dict[name] = adata.obs[
                    ['array_row', 'array_col']
                ].values
            else:
                self.loc_dict[name] = positions

            # pathway
            self.pathway_dict[name] = self._load_pathway_new(
                name, len(adata)
            )

            # image patches
            self.img_dict[name] = self._load_patches_h5(name)

        self.lengths = [len(self.adata_dict[n]) for n in self.names]
        self.cumlen = np.cumsum(self.lengths)

    # ===================================================================
    #  GSE FORMAT  –  per-sample loaders
    # ===================================================================
    def _load_adata(self, name):
        path = f"{self.adata_dir}/{name}.h5ad"
        adata = sc.read_h5ad(path)
        adata = adata[:, self.genes]
        if self.normalize:
            adata = normalize_adata(adata, cpm=self.cpm)
        return adata

    def _load_patches_h5(self, name, idx=None):
        path = f"{self.patches_dir}/{name}.h5"
        with h5py.File(path, 'r') as f:
            img = f['img'][idx] if idx is not None else f['img'][:]
        return img

    def _load_neighbor_emb(self, name, idx=None):
        path = f"{self.neighbor_dir}/{name}.h5"
        with h5py.File(path, 'r') as f:
            emb_key = 'embeddings' if 'embeddings' in f else 'features'
            if idx is not None:
                emb = f[emb_key][idx]
                mask = f['mask_tb'][idx]
            else:
                emb = f[emb_key][:]
                mask = f['mask_tb'][:]
        return torch.FloatTensor(emb), torch.LongTensor(mask)

    def _load_global_emb(self, name):
        path = f"{self.global_dir}/{name}.h5"
        with h5py.File(path, 'r') as f:
            emb_key = 'embeddings' if 'embeddings' in f else 'features'
            emb = f[emb_key][:]
            coords = f['coords'][:] if 'coords' in f else None
        return torch.FloatTensor(emb), coords

    def _load_pathway_new(self, name, n_spots):
        possible_paths = [
            f"{self.pathway_dir}/{name}.h5ad",
            f"{self.pathway_dir}/{name}_pathway.h5ad",
        ]
        pathway_path = None
        for p in possible_paths:
            if os.path.isfile(p):
                pathway_path = p
                break

        if pathway_path is not None:
            try:
                pw_adata = sc.read_h5ad(pathway_path)
                pathway_data = None

                # try obsm
                for key in [
                    self.pathway_key, 'pathway', 'aucell', 'X_pathway'
                ]:
                    if key in pw_adata.obsm:
                        pathway_data = pw_adata.obsm[key]
                        break

                # try X
                if pathway_data is None and pw_adata.X is not None:
                    pathway_data = (
                        pw_adata.X.toarray()
                        if sparse.issparse(pw_adata.X)
                        else np.array(pw_adata.X)
                    )

                # try layers
                if pathway_data is None and 'pathway' in pw_adata.layers:
                    pathway_data = pw_adata.layers['pathway']
                    if sparse.issparse(pathway_data):
                        pathway_data = pathway_data.toarray()

                if pathway_data is not None:
                    if sparse.issparse(pathway_data):
                        pathway_data = pathway_data.toarray()
                    pathway_data = np.array(pathway_data)
                    pathway_data = self._adjust_pathway_dims(pathway_data)

                    if pathway_data.shape[0] != n_spots:
                        print(
                            f"Warning: Pathway spots ({pathway_data.shape[0]})"
                            f" != adata spots ({n_spots}) for {name}"
                        )
                        if pathway_data.shape[0] > n_spots:
                            pathway_data = pathway_data[:n_spots]
                        else:
                            padded = np.zeros(
                                (n_spots, pathway_data.shape[1])
                            )
                            padded[: pathway_data.shape[0]] = pathway_data
                            pathway_data = padded

                    print(
                        f"Loaded pathway for {name}: "
                        f"shape {pathway_data.shape}"
                    )
                    return torch.FloatTensor(pathway_data)

            except Exception as e:
                print(f"Warning: Error loading pathway for {name}: {e}")

        return torch.zeros((n_spots, self.num_pathways))

    def _adjust_pathway_dims(self, pathway_data):
        n_spots, n_original = pathway_data.shape
        if n_original == self.num_pathways:
            return pathway_data
        elif n_original > self.num_pathways:
            return pathway_data[:, : self.num_pathways]
        else:
            padded = np.zeros((n_spots, self.num_pathways))
            padded[:, :n_original] = pathway_data
            return padded

    # ===================================================================
    #  __len__
    # ===================================================================
    def __len__(self):
        if self.mode == 'train':
            return self.cumlen[-1]
        else:
            return len(self.names)

    # ===================================================================
    #  __getitem__  –  dispatcher
    # ===================================================================
    def __getitem__(self, index):
        if self.is_new_format:
            return self._getitem_new(index)
        else:
            return self._getitem_old(index)

    # ===================================================================
    #  __getitem__  –  OLD FORMAT
    # ===================================================================
    def _getitem_old(self, index):
        if self.mode == 'train':
            i = 0
            while index >= self.cumlen[i]:
                i += 1
            idx = index
            if i > 0:
                idx = index - self.cumlen[i - 1]

            name = self.id2name[i]

            im = self.img_dict[name]
            if self.use_pyvips:
                img_shape = self.img_shape_dict[name]
            else:
                img_shape = im.shape

            center = self.center_dict[name][idx]
            x, y = center

            mask_tb = self.make_masking_table(x, y, img_shape)

            if self.use_pyvips:
                patches = im.extract_area(
                    x, y, self.r * 2, self.r * 2
                ).numpy()[:, :, :3]
            else:
                patches = im[y - self.r : y + self.r, x - self.r : x + self.r, :]

            if self.mode == "external_test":
                patches = self.test_transforms(patches)
            else:
                patches = self.train_transforms(patches)

            # pathway
            pathway_data = self.pathway_dict[name]
            pathway = pathway_data.iloc[idx]
            pathway = torch.tensor(pathway.values, dtype=torch.float32)

            exps = self.exp_dict[name][idx]
            exps = torch.Tensor(exps)

            sid = torch.LongTensor([idx])

            neighbors = torch.load(
                self.data_dir + f"/{self.neighbor_dir}/{name}.pt"
            )[idx]

        else:
            # non-train modes
            i = index
            name = self.id2name[i]

            im = self.img_dict[name]
            if self.use_pyvips:
                img_shape = self.img_shape_dict[name]
            else:
                img_shape = im.shape

            centers = self.center_dict[name]
            n_patches = len(centers)

            if self.extract_mode == 'neighbor':
                patches = torch.zeros(
                    (
                        n_patches,
                        3,
                        2 * self.r * self.num_neighbors,
                        2 * self.r * self.num_neighbors,
                    )
                )
            else:
                patches = torch.zeros(
                    (n_patches, 3, 2 * self.r, 2 * self.r)
                )
            mask_tb = torch.ones((n_patches, self.num_neighbors ** 2))

            for j in range(n_patches):
                center = centers[j]
                x, y = center

                mask_tb[j] = self.make_masking_table(x, y, img_shape)

                if self.extract_mode == 'neighbor':
                    k_ranges = [
                        (self.r * 2 * k, self.r * 2 * (k + 1))
                        for k in range(self.num_neighbors)
                    ]
                    m_ranges = [
                        (self.r * 2 * m, self.r * 2 * (m + 1))
                        for m in range(self.num_neighbors)
                    ]
                    patch = torch.zeros(
                        (
                            3,
                            2 * self.r * self.num_neighbors,
                            2 * self.r * self.num_neighbors,
                        )
                    )

                    if self.use_pyvips:
                        patch_unnorm = self.extract_patches_pyvips(
                            im, x, y, img_shape
                        )
                        for k, (k_start, k_end) in enumerate(k_ranges):
                            for m, (m_start, m_end) in enumerate(m_ranges):
                                n = k * self.num_neighbors + m
                                if mask_tb[j, n] != 0:
                                    patch_data = patch_unnorm[
                                        k_start:k_end, m_start:m_end, :
                                    ]
                                    transformed_patch = self.test_transforms(
                                        patch_data
                                    )
                                    patch[
                                        :, k_start:k_end, m_start:m_end
                                    ] = transformed_patch
                    else:
                        y_start = y - self.r * self.num_neighbors
                        x_start = x - self.r * self.num_neighbors

                        for k, (k_start, k_end) in enumerate(k_ranges):
                            for m, (m_start, m_end) in enumerate(m_ranges):
                                n = k * self.num_neighbors + m
                                if mask_tb[j, n] != 0:
                                    tmp = im[
                                        y_start + k_start : y_start + k_end,
                                        x_start + m_start : x_start + m_end,
                                        :,
                                    ]
                                    patch[
                                        :, k_start:k_end, m_start:m_end
                                    ] = self.test_transforms(tmp)
                else:
                    if self.use_pyvips:
                        patch = im.extract_area(
                            x, y, self.r * 2, self.r * 2
                        ).numpy()[:, :, :3]
                    else:
                        patch = im[
                            y - self.r : y + self.r,
                            x - self.r : x + self.r,
                            :,
                        ]
                    patch = self.test_transforms(patch)

                patches[j] = patch

            if self.mode == "extraction":
                return patches

            if self.mode != "inference":
                exps = self.exp_dict[name]
                exps = torch.Tensor(exps)
                pathway_data = self.pathway_dict[name]
                pathway = pathway_data.values
                pathway = torch.Tensor(pathway)

            sid = torch.arange(n_patches)
            neighbors = torch.load(
                self.data_dir + f"/{self.neighbor_dir}/{name}.pt"
            )

        wsi = torch.load(
            self.data_dir + f"/{self.gt_dir}/{name}.pt"
        )

        pid = torch.LongTensor([i])
        pos = self.loc_dict[name]
        position = torch.LongTensor(pos)

        if self.mode not in ["external_test", "inference"]:
            name += f"+{self.data}"

        if self.mode == 'train':
            return (
                patches, exps, pid, sid, wsi,
                position, neighbors, mask_tb, pathway,
            )
        elif self.mode == 'inference':
            return (
                patches, sid, wsi, position,
                neighbors, mask_tb, pathway,
            )
        else:
            return (
                patches, exps, sid, wsi, position,
                name, neighbors, mask_tb, pathway,
            )

    # ===================================================================
    #  __getitem__  –  GSE FORMAT
    # ===================================================================
    def _getitem_new(self, index):
        if self.mode == 'train':
            # find which sample this index belongs to
            i = 0
            while index >= self.cumlen[i]:
                i += 1
            idx = index
            if i > 0:
                idx = index - self.cumlen[i - 1]

            name = self.id2name[i]

            # image patch
            img = self.img_dict[name][idx]
            patches = self.train_transforms(img)

            # expression
            exps = torch.FloatTensor(self.exp_dict[name][idx])

            # neighbour embeddings
            neighbors, mask_tb = self._load_neighbor_emb(name, idx)

            # global embeddings
            wsi, _ = self._load_global_emb(name)

            # position
            position = torch.LongTensor(self.loc_dict[name])

            # pathway
            pathway = self.pathway_dict[name][idx]

            pid = torch.LongTensor([i])
            sid = torch.LongTensor([idx])

            return (
                patches, exps, pid, sid, wsi,
                position, neighbors, mask_tb, pathway,
            )

        else:
            # test / validation / extraction / inference
            i = index
            name = self.id2name[i]

            imgs = self.img_dict[name]
            n_patches = len(imgs)

            patches = torch.stack(
                [self.test_transforms(img) for img in imgs], dim=0
            )

            # neighbour embeddings
            neighbors, mask_tb = self._load_neighbor_emb(name)

            # global embeddings
            wsi, _ = self._load_global_emb(name)

            # position
            position = torch.LongTensor(self.loc_dict[name])

            # pathway
            pathway = self.pathway_dict[name]

            sid = torch.arange(n_patches)

            if self.mode == "extraction":
                return patches

            if self.mode != "inference":
                exps = torch.FloatTensor(self.exp_dict[name])

            pid = torch.LongTensor([i])

            if self.mode not in ["external_test", "inference"]:
                name = f"{name}+{self.data}"

            if self.mode == 'inference':
                return (
                    patches, sid, wsi, position,
                    neighbors, mask_tb, pathway,
                )
            else:
                return (
                    patches, exps, sid, wsi, position,
                    name, neighbors, mask_tb, pathway,
                )