import itertools
from tqdm import tqdm
from typing import List, Dict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .template import DataProducerTemplate

import elephant
from elephant.data.corpus import Corpus
from elephant.utils.common_utils import get_signature
from elephant.utils.logging_utils import get_logger
from elephant.utils.huggingface_utils import get_huggingface_tokenizer

logger = get_logger("elephant")


class UDMBertDPDataProducer(DataProducerTemplate):
    def __init__(self, corpus: Corpus, producer_cfg):
        super(UDMBertDPDataProducer, self).__init__(corpus, producer_cfg)

        self.tokenizer = self.get_tokenizer()

    def get_tokenizer(self):
        tokenizer_name = self.producer_cfg.tokenizer.name
        assert tokenizer_name.startswith("huggingface")

        hf_tokenizer_name = tokenizer_name.split('/', 2)[1]
        hf_tokenizer_path = elephant.root_path / "tokenizer" / tokenizer_name
        tokenizer = get_huggingface_tokenizer(hf_tokenizer_name, hf_tokenizer_path)
        return tokenizer

    def prepare_data(self, split: str, **criteria) -> pd.DataFrame:
        """
        Call encoder to 1) prepare dataset for model; 2) cache data to disk.
        """
        producer_info = {
            "corpus": self.corpus.corpus_cfg.name,
            "encoder": self.producer_cfg.encoder.name,
            "header": self.producer_cfg.task.name,
        }
        md5, _ = get_signature(producer_info)
        cache_root = elephant.cache_path / md5

        corpus_fields = self.producer_cfg.corpus_fields
        model_fields = self.producer_cfg.mapping.fields
        field_mapping = dict(zip(corpus_fields, model_fields))

        process_fields = self.producer_cfg.processor.fields
        dataframe = pd.DataFrame([], columns=process_fields)

        langs = criteria["langs"]
        genres = criteria["genres"]
        for lang in langs:
            for genre in genres[lang]:
                cache_path = cache_root / f"lang={lang}" / f"genre={genre}" / f"split={split}"
                if cache_path.exists():
                    filters = [("lang", '=', lang), ("genre", '=', genre), ("split", '=', split)]
                    ds = pq.ParquetDataset(str(cache_root), filters=filters, use_legacy_dataset=False)
                    table = ds.read()
                    df = self._pyarrow_table_to_dataframe(table)
                    logger.info(f"Dataset: [lang='{lang}', genre='{genre}', split='{split}']: {len(df)} sent")
                else:
                    df = self.corpus.load(split=split, lang=lang, genre=genre)
                    df = df.rename(columns=field_mapping)
                    df = self.preprocess(df)
                    table = self._dataframe_to_pyarrow_table(df)
                    pq.write_to_dataset(
                        table, root_path=cache_root, partition_cols=["lang", "genre", "split"]
                    )
                dataframe = pd.concat([dataframe, df.loc[:, process_fields]], ignore_index=True)

        return dataframe

    @staticmethod
    def _dataframe_to_pyarrow_table(df: pd.DataFrame) -> pa.Table:
        df_to_save = df.loc[:, df.columns]
        # df_to_save.loc[:, ["distances"]] = df["distances"].apply(np.ndarray.tolist)
        # df_to_save.loc[:, ["deprel_matrix"]] = df["deprel_matrix"].apply(np.ndarray.tolist)
        table = pa.Table.from_pandas(df_to_save)  # noqa
        return table

    @staticmethod
    def _pyarrow_table_to_dataframe(table: pa.Table) -> pd.DataFrame:
        df_to_load = table.to_pandas()
        return df_to_load

    def collate_data(self, batch: List[Dict]) -> Dict:
        """
        Call encoder to 1) convert datatype to tensor; 2) pad data to the same length. Load data to GPU
        """
        collated_data = self.collate(batch)

        for key, value in collated_data.items():
            if not isinstance(value, torch.Tensor):
                continue
            else:
                collated_data[key] = value.to(elephant.device)

        return collated_data

    def preprocess(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess (filter, augment and process) data
        """
        if hasattr(self.producer_cfg, "filter"):
            dataframe = self._filter(dataframe)
        if hasattr(self.producer_cfg, "augmentor"):
            dataframe = self._augment(dataframe)
        if hasattr(self.producer_cfg, "processor"):
            dataframe = self._process(dataframe)

        return dataframe

    # --------------------------------------------------------------------------------------------
    # filter data
    # --------------------------------------------------------------------------------------------
    def _filter(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        filter_list = ["exceed_max_length", "invalid_tree"]

        progress_bar = tqdm(total=len(dataframe), desc="filtering dataset ...", leave=False)
        valid_indices = []
        for index, row in dataframe.iterrows():
            progress_bar.update()
            is_valid = True
            for f in filter_list:
                is_valid = is_valid and not getattr(self, f"_{f}")(row)
                if not is_valid:
                    break
            valid_indices.append(is_valid)

        filtered_dataset = dataframe[valid_indices]

        return filtered_dataset

    def _exceed_max_length(self, row: pd.Series) -> bool:
        words = row["words"]
        max_len = self.producer_cfg.filter.max_len
        max_len_unit = self.producer_cfg.filter.max_len_unit

        if max_len_unit == "word":
            if len(words) > max_len:
                return True
            else:
                max_len = self.tokenizer.max_len_single_sentence
        else:
            max_len = min(max_len, self.tokenizer.max_len_single_sentence)

        subwords = [self.tokenizer.tokenize(word) for word in words]
        subwords = sum(subwords, [])
        if len(subwords) >= max_len:
            return True

        return False

    def _invalid_tree(self, row: pd.Series) -> bool:  # noqa
        head = [int(h) for h in row["heads"]]

        tree = []

        def find_children(root_idx):
            tree.append(root_idx)
            for idx in range(len(head)):
                if head[idx] - 1 == root_idx:
                    find_children(idx)

        assert 0 in head
        root = head.index(0)
        find_children(root)
        invalid = len(tree) != len(head)

        return invalid

    # --------------------------------------------------------------------------------------------
    # augment data
    # --------------------------------------------------------------------------------------------
    def _augment(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        fields = self.producer_cfg.augmentor.fields
        augmented_dataset = dataframe.loc[:, dataframe.columns]

        tqdm.pandas(desc="augmenting dataset ...", leave=False)
        augmented_dataset[fields] = augmented_dataset.progress_apply(self._augment_row, axis=1)

        return augmented_dataset

    def _augment_row(self, row: pd.Series) -> pd.Series:  # noqa
        return row

    # --------------------------------------------------------------------------------------------
    # process data
    # --------------------------------------------------------------------------------------------
    def _process(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        tqdm.pandas(desc="processing dataset ...", leave=False)
        processed_dataframe = dataframe.progress_apply(self._process_row, axis=1)

        return processed_dataframe

    def _process_row(self, row: pd.Series) -> pd.Series:
        lang = row["lang"]
        genre = row["genre"]
        split = row["split"]
        seqid = row["seqid"]

        words = row["words"]
        subwords_group = [self.tokenizer.tokenize(w) for w in words]
        tokens = list(itertools.chain.from_iterable(subwords_group))
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        token_ids = self.tokenizer.build_inputs_with_special_tokens(token_ids)
        tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]

        attention_mask = [1 if t != self.tokenizer.pad_token_id else 0 for t in token_ids]

        token_type_ids = [0 for _ in token_ids]

        special_tokens_mask = self._get_special_tokens_mask(token_ids)

        heads = row["heads"]
        head_ids = self._update_head_for_subword(heads, subwords_group)
        head_ids = self.tokenizer.build_inputs_with_special_tokens(head_ids)
        head_ids = head_ids * (1 - special_tokens_mask) + elephant.config.pad_head_id * special_tokens_mask

        postags = row["postags"]
        values = self.producer_cfg.processor.upos_values
        postag2ids = {label: i for i, label in enumerate(values)}
        postag_ids = [postag2ids[t] for t in postags]
        postag_ids = self._update_label_for_subword(postag_ids, subwords_group)
        postag_ids = self.tokenizer.build_inputs_with_special_tokens(postag_ids)
        postag_ids = postag_ids * (1 - special_tokens_mask) + elephant.config.pad_label_id * special_tokens_mask

        deprels = row["deprels"]
        values = self.producer_cfg.processor.deprel_values
        deprel2ids = {label: i for i, label in enumerate(values)}
        deprel_ids = [deprel2ids[t] for t in deprels]
        deprel_ids = self._update_label_for_subword(deprel_ids, subwords_group)
        deprel_ids = self.tokenizer.build_inputs_with_special_tokens(deprel_ids)
        deprel_ids = deprel_ids * (1 - special_tokens_mask) + elephant.config.pad_label_id * special_tokens_mask

        length = len(token_ids)

        # distance = self._get_parse_distance_matrix(length, head_ids)

        # deprel_matrix = self._get_dep_rel_matrix(length, head_ids, deprel_ids)

        row = [
            lang, genre, split, seqid, tokens, length, token_ids, attention_mask, token_type_ids,
            postag_ids, head_ids, deprel_ids,  # distance, deprel_matrix
        ]
        processed_row = pd.Series(row, index=self.producer_cfg.processor.fields)

        return processed_row

    def _update_head_for_subword(self, heads: np.array, subwords_group: List) -> List:  # noqa
        word_position = np.cumsum([0, 1] + [len(w) for w in subwords_group])
        new_head = []
        for subwords, head in zip(subwords_group, heads):
            assert head >= 0, "Error: head < 0"
            for i, subword in enumerate(subwords):
                h = word_position[head] if i == 0 else elephant.config.pad_head_id
                new_head.append(h)

        return new_head

    def _update_label_for_subword(self, labels: np.array, subwords_group: List) -> List:  # noqa
        new_label = []
        for subwords, label in zip(subwords_group, labels):
            for i, subword in enumerate(subwords):
                t = label if i == 0 else elephant.config.pad_label_id
                new_label.append(t)

        return new_label

    def _get_special_tokens_mask(self, token_ids: List) -> np.array:
        mask = self.tokenizer.get_special_tokens_mask(token_ids, already_has_special_tokens=True)
        mask = np.array(mask)
        unk_token_id = self.tokenizer.unk_token_id
        unk_mask = [1 if t == unk_token_id else 0 for t in token_ids]
        unk_mask = np.array(unk_mask)
        special_tokens_mask = mask & (~unk_mask)

        return special_tokens_mask

    @staticmethod
    def _get_dep_rel_matrix(sent_length, head_ids, arc_label_ids):
        dep_rel_matrix = np.zeros((sent_length, sent_length))
        for i in range(sent_length):
            for j in range(i, sent_length):
                if head_ids[i] == -1 or head_ids[j] == -1:
                    dep_rel_matrix[i, j] = -1
                    dep_rel_matrix[j, i] = -1
                elif head_ids[j] == i:
                    rel = arc_label_ids[j]
                    dep_rel_matrix[i, j] = rel
                    dep_rel_matrix[j, i] = -1
                elif head_ids[i] == j:
                    rel = arc_label_ids[i]
                    dep_rel_matrix[j, i] = rel
                    dep_rel_matrix[i, j] = -1
                else:
                    dep_rel_matrix[j, i] = -1
                    dep_rel_matrix[i, j] = -1
        np.fill_diagonal(dep_rel_matrix, -1)
        return dep_rel_matrix

    def _get_parse_distance_matrix(self, sent_length, head_ids):
        distances = np.zeros((sent_length, sent_length))
        for i in range(sent_length):
            for j in range(i, sent_length):
                i_j_distance = self._distance_between_pairs(head_ids, i, j)
                distances[i][j] = i_j_distance
                distances[j][i] = i_j_distance

        return distances

    @staticmethod
    def _distance_between_pairs(head_ids, i, j):
        if head_ids[i] == -1 or head_ids[j] == -1:
            return -1
        if i == j:
            return 0
        i_path = [i]
        j_path = [j]
        i_head = i
        j_head = j
        while True:
            if not (i_head == 0 and (i_path == [i] or i_path[-1] == 0)):
                i_head = head_ids[i_head]
                i_path.append(i_head)
                if i_head == -1:
                    print("Appending to i_path:", i_head)
                    print(head_ids)
                    raise AssertionError
            if not (j_head == 0 and (j_path == [j] or j_path[-1] == 0)):
                j_head = head_ids[j_head]
                j_path.append(j_head)
                if j_head == -1:
                    print("Appending to j_path:", j_head)
                    print(head_ids)
                    raise AssertionError
            if i_head in j_path:
                j_path_length = j_path.index(i_head)
                i_path_length = len(i_path) - 1
                break
            elif j_head in i_path:
                i_path_length = i_path.index(j_head)
                j_path_length = len(j_path) - 1
                break
            elif i_head == j_head:
                i_path_length = len(i_path) - 1
                j_path_length = len(j_path) - 1
                break

        total_length = j_path_length + i_path_length

        return total_length

    # --------------------------------------------------------------------------------------------
    # collate data
    # --------------------------------------------------------------------------------------------
    def collate(self, batch: List[Dict]) -> Dict:
        """
        Convert datatype to tensor, and pad batch to the same length
        """
        assert len(batch) > 0
        data = {k: [dic[k] for dic in batch] for k in batch[0]}

        lang = data["lang"]
        genre = data["genre"]
        split = data["split"]
        seqid = data["seqid"]
        tokens = data["tokens"]

        length = torch.tensor(data["length"], dtype=torch.long)

        token_ids = [torch.tensor(d, dtype=torch.long) for d in data["token_ids"]]
        token_ids = torch.nn.utils.rnn.pad_sequence(token_ids, batch_first=True)

        attention_mask = [torch.tensor(d, dtype=torch.long) for d in data["attention_mask"]]
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True)

        token_type_ids = [torch.tensor(d, dtype=torch.long) for d in data["token_type_ids"]]
        token_type_ids = torch.nn.utils.rnn.pad_sequence(
            token_type_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        postag_ids = [torch.tensor(d, dtype=torch.long) for d in data["postag_ids"]]
        postag_ids = torch.nn.utils.rnn.pad_sequence(
            postag_ids, batch_first=True, padding_value=elephant.config.pad_label_id
        )

        head_ids = [torch.tensor(d, dtype=torch.long) for d in data["head_ids"]]
        head_ids = torch.nn.utils.rnn.pad_sequence(
            head_ids, batch_first=True, padding_value=elephant.config.pad_head_id
        )

        deprel_ids = [torch.tensor(d, dtype=torch.long) for d in data["deprel_ids"]]
        deprel_ids = torch.nn.utils.rnn.pad_sequence(
            deprel_ids, batch_first=True, padding_value=elephant.config.pad_label_id
        )

        # size, max_seq_len = token_ids.shape
        # distances = [-torch.ones(*(max_seq_len, max_seq_len)) for _ in range(size)]
        # deprel_matrix = [-torch.ones(*(max_seq_len, max_seq_len), dtype=torch.long) for _ in range(size)]
        # for i in range(size):
        #     true_len = length[i]
        #     _distances = torch.stack([torch.tensor(d, dtype=torch.float64) for d in data["distances"][i]])
        #     distances[i][:true_len, :true_len] = _distances
        #    _deprel_matrix = torch.stack([torch.tensor(d, dtype=torch.long) for d in data["deprel_matrix"][i]])
        #    deprel_matrix[i][:true_len, :true_len] = _deprel_matrix

        # distances = torch.stack(distances)
        # deprel_matrix = torch.stack(deprel_matrix)

        collated_data = {
            "lang": lang,
            "genre": genre,
            "split": split,
            "seqid": seqid,
            "tokens": tokens,
            "length": length,
            "token_ids": token_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "postag_ids": postag_ids,
            "head_ids": head_ids,
            "deprel_ids": deprel_ids,
            # "distances": distances,
            # "deprel_matrix": deprel_matrix,
        }

        return collated_data
