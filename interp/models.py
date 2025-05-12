import torch
import os
import pandas as pd
import numpy as np
import sklearn.linear_model
import sklearn.svm
import json
import matplotlib.pyplot as plt

from typing import Optional, List, Dict, Tuple
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

from concept_datasets import CONCEPT_DIR
from concept_filters import GEN_AUDIO_FILTER_DICT, GEN_AUDIO_PADDER_DICT, create_concept_filter

class BinaryClassifier():
    def __init__(self, 
                 concept_filter: str, 
                 num_pos_samples: int,
                 num_neg_samples: int,
                 model_name: str,
                 result_json_path: Optional[str] = None,
                 concept_pt_path: Optional[str] = None,
                 dir_path: Optional[str] = None,
                 hparams: Optional[Dict] = None,
                 ) -> None:
        '''
        Binary classifier trained on concept embeddings. We used lightweight classifiers with linear/non-linear
        decision boundaries (i.e. logistic, svm) for ease of interpretability and computation.
        '''
        self.concept_filter = concept_filter
        self.concept_padder = f"{concept_filter}_like"
        self.model_name = model_name

        self.dir_path = dir_path if dir_path is not None else f"{CONCEPT_DIR}/{concept_filter}"  
        self.concept_pt_path = concept_pt_path if concept_pt_path is not None else f"{self.dir_path}/{model_name}_embeddings.pt"
        self.hparams = hparams if hparams is not None else {}

        self.num_pos_samples = num_pos_samples
        self.num_neg_samples = num_neg_samples

        self.result_json_path = result_json_path if result_json_path is not None else f"data/concepts/{self.concept_filter}/concept_results.json"
        
    def train(self, csv_path: Optional[str] = None) -> None:
        'Wrapper for _train_svm, _train_logistic_model'

        csv_path = csv_path if csv_path is not None else f"{self.dir_path}/audio_info.csv"
        
        embeds, labels = self._load_concept_embeds(self.concept_pt_path, csv_path, self.model_name)

        if self.hparams["model_type"] == "logistic":
            tr_acc, te_acc, model = self._train_logistic_model(embeds, labels)
        elif self.hparams["model_type"] == "svm":
            tr_acc, te_acc, model = self._train_svm(embeds, labels)
        else:
            raise NotImplementedError(f"Model type {self.hparams['model_type']} not implemented")

        self.concept_train_accuracy = tr_acc
        self.concept_test_accuracy = te_acc
        self.model = model
    

    def inference(self, diff_step: int, embed_model: str, save: bool = True,) -> Tuple[float, float, float]:
        'Inference given pre-trained classifier'

        pos_tensor, neg_tensor = self._load_gen_audio_embeds(csv_path=None, pt_path=None,
                                                             diff_step=diff_step, embed_model=embed_model)
        labels = np.concatenate([np.ones(pos_tensor.shape[0]), np.zeros(neg_tensor.shape[0])])
        input_tensor = torch.concat([pos_tensor, neg_tensor])

        pred = self.model.predict(input_tensor)
        
        tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()
        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)
        acc = sklearn.metrics.accuracy_score(labels, pred)

        if save:
            new_row = {
                    "concept": self.concept_filter,
                    "diff_step": diff_step,
                    "pos_gen_audio_samples": pos_tensor.shape[0],
                    "neg_gen_audio_samples": neg_tensor.shape[0], 
                    "embed_model": embed_model,
                    "class_model_type": self.hparams["model_type"],
                    "class_model_train_acc": self.concept_train_accuracy,
                    "class_model_test_acc": self.concept_test_accuracy,
                    "acc": acc,
                    "tpr": tpr,
                    "tnr": tnr,
                }
            if os.path.exists(self.result_json_path):
                df = pd.read_json(self.result_json_path)
            else:
                df = pd.DataFrame(columns=new_row.keys())

            # Dumps results into json
            new_df = pd.DataFrame([new_row])
            df = pd.concat([df, new_df], ignore_index=True)
            records = df.to_dict(orient="records")
            with open(self.result_json_path, "w") as fp:
                json.dump(records, fp, indent=2)

        return acc, tpr, tnr
    

    def _train_svm(self, embeds: np.array, labels: np.array):
        svm_kernel = self.hparams.get("svm_kernel", "rbf")
        svm_C = self.hparams.get("svm_C", 1.0)

        model = sklearn.svm.SVC(kernel=svm_kernel, C=svm_C)
        test_size = self.hparams.get("test_size", 0.2)

        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
            embeds, labels, test_size=test_size
        )

        model.fit(x_train, y_train)

        train_pred = model.predict(x_train)
        test_pred  = model.predict(x_test)
        train_acc  = sklearn.metrics.accuracy_score(y_train, train_pred)
        test_acc   = sklearn.metrics.accuracy_score(y_test,  test_pred)

        return train_acc, test_acc, model
    

    def _train_logistic_model(self, embeds: np.array, labels: np.array):
        lin_model = sklearn.linear_model.LogisticRegression()
        test_size = self.hparams["test_size"] if "test_size" in self.hparams else 0.2

        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(embeds, labels, test_size=test_size)   
        lin_model.fit(x_train, y_train) 

        train_pred, test_pred = lin_model.predict(x_train), lin_model.predict(x_test)

        train_acc = sklearn.metrics.accuracy_score(y_train, train_pred)
        test_acc = sklearn.metrics.accuracy_score(y_test, test_pred)

        return train_acc, test_acc, lin_model

    def _load_concept_embeds(self, pt_path: str, csv_path: str, model_name: str,) -> Tuple[np.array, np.array]:
        'Loads concept embeddings to np arrays, returning embeds and labels'
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"Concept embeddings file not found at {pt_path}")

        concept_embeds = torch.load(pt_path)

        batch_size, embed_dim = concept_embeds.shape

        concept_csv = pd.read_csv(csv_path)

        # Tests for corruption / mismatch
        assert len(concept_csv) == batch_size, f"Batch size mismatch between concept embeddings and concept CSV stored at {csv_path}"
        
        if self.model_name == "laion-clap":
            assert embed_dim == 512, f"Embedding dimension mismatch for {model_name} model. Expected 512, got {embed_dim}"

        # Assumes CSV, pt both sorted by positive, negative
        labels = np.concatenate([np.ones(self.num_pos_samples), np.zeros(self.num_neg_samples)])
        concept_embeds_np = concept_embeds.numpy()
        return concept_embeds_np, labels
        
    def _load_gen_audio_embeds(self,
                               diff_step: int, 
                               embed_model: str,
                               pt_path: Optional[None],  
                               csv_path: Optional[str],
                               ) -> Tuple[torch.Tensor]:
        'Gets gen_audio embeds'
        
        # Processing the csv
        csv_path = csv_path if csv_path is not None else "data/generated/audio_info.csv"
        gen_audio_csv = pd.read_csv(csv_path)

        # Filters only use these columns
        refined_csv = gen_audio_csv[["diffusion_step", "prompt_index", "sample_index", "tag.aspects"]]
        
        pt_path = pt_path if pt_path is not None else f"data/generated/diff_step_{diff_step}/{embed_model}_embeddings.pt"

        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"Generated audio embeddings file not found at {pt_path}")
        
        gen_audio_embeds = torch.load(pt_path)
        prompts, batch_size, embed_dim = gen_audio_embeds.shape

        # Check that batch_size matches up
        assert prompts == 1000, f"{prompts} prompts found; 1000 expected."
        assert batch_size == 7, f"Batch size of {batch_size} found; batch size of 7 expected."

        col_dict, get_true, logic = GEN_AUDIO_FILTER_DICT[self.concept_filter]
        concept_filter_func = create_concept_filter(col_dict, get_true, logic)

        col_dict, _, logic = GEN_AUDIO_PADDER_DICT[self.concept_padder]
        concept_padder_func = create_concept_filter(col_dict, False, logic)

        pos_samps = concept_filter_func(refined_csv)
        neg_samps = concept_padder_func(refined_csv)

        pos_samps["prompt_index"] = pd.to_numeric(
            pos_samps["prompt_index"], errors="coerce"
        )
        neg_samps["prompt_index"] = pd.to_numeric(
            neg_samps["prompt_index"], errors="coerce"
        )

        pos_indices = torch.tensor(pos_samps["prompt_index"].dropna().astype(int).unique(), dtype=torch.int64)
        neg_indices = torch.tensor(neg_samps["prompt_index"].dropna().astype(int).unique(), dtype=torch.int64)

        print(f"Positive indices: {pos_indices}")
        print(f"Negative indices: {neg_indices}")
        print(f"Intersection: {list(set(pos_indices.tolist()) & set(neg_indices.tolist()))}")

        num_pos_prompts, num_neg_prompts = len(pos_indices), len(neg_indices)

        pos_tensor = gen_audio_embeds[pos_indices, :, :].reshape(num_pos_prompts * batch_size, embed_dim)
        neg_tensor = gen_audio_embeds[neg_indices, :, :].reshape(num_neg_prompts * batch_size, embed_dim)

        return (pos_tensor, neg_tensor)
    

    def get_pca(self, diff_steps: List[int], embed_model: str, resolution: int = 200,):
        'Produces PCA ims demonstrating embeddings moving throughout diffusion process'

        pca = PCA(n_components=2)
        figs = []
        for diff_step in diff_steps:
            pt_path = f"{self.dir_path}/{embed_model}_embeddings.pt"
            concept_embeds, concept_labels = self._load_concept_embeds(csv_path=f"{self.dir_path}/audio_info.csv", 
                                                                    model_name=embed_model,
                                                                    pt_path=pt_path)
            
            pos_tensor, neg_tensor = self._load_gen_audio_embeds(diff_step=diff_step, embed_model=embed_model, 
                                                                 pt_path=f"data/generated/diff_step_{diff_step}/{embed_model}_embeddings.pt", 
                                                                 csv_path=f"data/generated/audio_info.csv")
            
            gen_audio_embeds = torch.concat([pos_tensor, neg_tensor]).numpy()
            gen_audio_labels = np.concatenate([np.ones(pos_tensor.shape[0]), np.zeros(neg_tensor.shape[0])], axis=0)

            train2d = pca.fit_transform(concept_embeds) if diff_step == 5 else pca.transform(concept_embeds)
            test2d = pca.transform(gen_audio_embeds) 

            x0_min, x0_max = train2d[:,0].min()-0.1, train2d[:,0].max()+0.1
            x1_min, x1_max = train2d[:,1].min()-0.1, train2d[:,1].max()+0.1
            xx, yy = np.meshgrid(
                np.linspace(x0_min, x0_max, resolution),
                np.linspace(x1_min, x1_max, resolution)
            )
            grid2d = np.c_[xx.ravel(), yy.ravel()]

            # Computing predictions
            grid_orig = pca.inverse_transform(grid2d)
            Z = self.model.predict(grid_orig).reshape(xx.shape)

            # Plotting
            fig, ax = plt.subplots(figsize=(8, 8))

            model_type = self.hparams["model_type"]

            ax.contourf(xx, yy, Z, alpha=0.2,)
            ax.scatter(train2d[concept_labels==1,0], train2d[concept_labels==1,1], s=6,
                marker="o", edgecolor="mediumseagreen", facecolor="mediumseagreen", label="Concept +")
            ax.scatter(train2d[concept_labels==0,0], train2d[concept_labels==0,1], s=6,
                marker="x", edgecolor="k", facecolor="tomato", label="Concept -")
            ax.scatter(test2d[gen_audio_labels==1,0], test2d[gen_audio_labels==1,1], s=8,
                marker="o", edgecolor="black", facecolor="black", label="Diffusion Audio +")
            
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title(f"PCA projection of CLAP concept, generated audio embeds with {model_type} decision boundary")
            ax.legend(loc="best")
            plt.tight_layout()
            figs.append(fig)
        return figs