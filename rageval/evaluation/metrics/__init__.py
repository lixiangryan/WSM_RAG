<<<<<<< HEAD
from .rag_metrics.generation.rouge_l import ROUGELScore
from .rag_metrics.retrieval.eir_precision import EIR_Precision
from .rag_metrics.retrieval.eir_recall import EIR_Recall
from .rag_metrics.generation.keypoint_metrics import KEYPOINT_METRICS
from .rag_metrics.retrieval.words_precision import Words_Precision
from .rag_metrics.retrieval.words_recall import Words_Recall

METRICS_REGISTRY = {
    "rouge-l": ROUGELScore,
    "words_precision": Words_Precision,
    "words_recall": Words_Recall,
    "sentences_precision": EIR_Precision,
    "sentences_recall": EIR_Recall,
=======
from .rag_metrics.retrieval.precision import Precision
from .rag_metrics.retrieval.recall import Recall
from .rag_metrics.generation.rouge_l import ROUGELScore
from .rag_metrics.retrieval.eir import EIR
from .rag_metrics.generation.keypoint_metrics import KEYPOINT_METRICS

METRICS_REGISTRY = {
    "rouge-l": ROUGELScore,
    "precision": Precision,
    "recall": Recall,
    "eir": EIR,
>>>>>>> wsm/wang
    "keypoint_metrics": KEYPOINT_METRICS,
}


def get_metric(metric_name):
    return METRICS_REGISTRY[metric_name]
