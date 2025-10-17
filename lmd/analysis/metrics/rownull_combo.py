import numpy as np
from typing import Dict, List
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from ..core.metrics_base import MetricBase
from ..core.types import MetricDirection, MetricOutput

class RowNullCombo(MetricBase):
    def __init__(self, model_type: str = "lda", cv_folds: int = 5, seed: int = 42):
        super().__init__(model_type=model_type, cv_folds=cv_folds, seed=seed)
        self.model_type = model_type
        self.cv_folds = cv_folds
        self.seed = seed
    
    @property
    def name(self) -> str:
        return "rownull_combo"
    
    @property
    def requires_lm_head(self) -> bool:
        return True
    
    @property
    def supported_modes(self) -> List[str]:
        return ["state"]
    
    @property
    def output_specs(self) -> Dict[str, MetricDirection]:
        return {"rownull_combo_score": MetricDirection.HIGHER_BETTER}
    
    @property
    def dependencies(self) -> List[str]:
        return ["rownull"]
    
    def compute_state(self, ctx, states: np.ndarray) -> MetricOutput:
        rownull_state = ctx.get_metric_state('rownull')
        if not rownull_state:
            N = len(states)
            return MetricOutput(
                name=self.name,
                scores={"rownull_combo_score": np.zeros(N, dtype=np.float32)},
                directions=self.output_specs
            )
        
        feature_keys = [
            'row_len_frac',
            'tail_row_fraction',
            'row_cohere_len',
            'null_cohere_len',
            'row_null_gap',
            'null_effective',
            'row_frac_gain'
        ]
        
        features = []
        for key in feature_keys:
            if key in rownull_state:
                features.append(rownull_state[key])
        
        if len(features) == 0:
            N = len(states)
            return MetricOutput(
                name=self.name,
                scores={"rownull_combo_score": np.zeros(N, dtype=np.float32)},
                directions=self.output_specs
            )
        
        X = np.column_stack(features).astype(np.float32)
        y = ctx.labels.astype(int)
        N = len(y)
        
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            return MetricOutput(
                name=self.name,
                scores={"rownull_combo_score": np.zeros(N, dtype=np.float32)},
                directions=self.output_specs
            )
        
        n_folds = min(self.cv_folds, len(pos_idx), len(neg_idx))
        
        if n_folds < 2:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            if self.model_type == "lda":
                model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            else:
                model = LogisticRegression(penalty='l2', C=1.0, max_iter=100, random_state=self.seed)
            
            model.fit(X_scaled, y)
            scores = model.decision_function(X_scaled).astype(np.float32)
        else:
            scores = np.zeros(N, dtype=np.float32)
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
            
            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train = y[train_idx]
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                if self.model_type == "lda":
                    model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
                else:
                    model = LogisticRegression(penalty='l2', C=1.0, max_iter=100, random_state=self.seed)
                
                model.fit(X_train_scaled, y_train)
                scores[val_idx] = model.decision_function(X_val_scaled).astype(np.float32)
        
        return MetricOutput(
            name=self.name,
            scores={"rownull_combo_score": scores},
            directions=self.output_specs
        )