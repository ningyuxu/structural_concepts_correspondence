from dataclasses import asdict, dataclass


@dataclass
class Enumeration:
    def choices(self):
        return list(asdict(self).values())


@dataclass
class Split(Enumeration):
    TRAIN:  str = "train"
    DEV:    str = "dev"
    TEST:   str = "test"


@dataclass
class EvaluationMetric(Enumeration):
    MICRO_ACCURACY:     str = "micro-average accuracy"
    MICRO_F1_SCORE:     str = "micro-average f1-score"
    MACRO_ACCURACY:     str = "macro-average accuracy"
    MACRO_F1_SCORE:     str = "macro-average f1-score"
    MEAN_SQUARED_ERROR: str = "mean squared error"
