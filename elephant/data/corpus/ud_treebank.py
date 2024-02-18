import copy
import shutil
from pathlib import Path
from tqdm import tqdm
from typing import Optional, List

import pandas as pd

from elephant.data.corpus.template import CorpusTemplate
from elephant.utils import enumeration as enum
from elephant.utils.tqdm_utils import download_with_progress_bar, extract_with_progress_bar
from elephant.utils.logging_utils import get_logger

logger = get_logger("elephant")


class UDTreebankCorpus(CorpusTemplate):
    def __init__(self, corpus_cfg):
        super(UDTreebankCorpus, self).__init__(corpus_cfg)

    def init(self) -> None:
        # download corpus file if not exists
        if not self.corpus_file.is_file():
            logger.info(f"Downloading {self.corpus_cfg.name} corpus ......")
            download_with_progress_bar(url=self.corpus_cfg.url, target_file=str(self.corpus_file))

        # extract corpus files
        shutil.rmtree(self.corpus_path, ignore_errors=True)
        logger.info(f"Extracting {self.corpus_cfg.name} corpus ......")
        extract_with_progress_bar(
            tar=self.corpus_file,
            tar_root_path="ud-treebanks-v2.10",  # the root directory in corpus zip file
            output_path=str(self.corpus_path)
        )

    def load(self, split: str = enum.Split.TRAIN, **criteria) -> pd.DataFrame:
        lang = criteria["lang"]
        genre = criteria["genre"]
        file = self._get_corpus_data_file(lang, genre, split)
        dataset = []
        desc = f"Dataset: [lang='{lang}', genre='{genre}', split='{split}']"
        for seqid, buffer in enumerate(tqdm(self._iter_sentences(str(file)), desc=desc, leave=True, unit=" sent")):
            conllx_lines = []
            # load corpus raw dataset, one conllx_line for one word
            for line in buffer:
                conllx_lines.append(line.strip().split('\t'))
            # word index may be a decimal number for empty nodes, remove these lines.
            conllx_lines = [x for x in conllx_lines if '.' not in x[0]]
            # word index may be a range for multiword tokens, process these lines.
            conllx_lines = self._process_span_lines(conllx_lines)
            # resolves multiple heads ambiguities
            conllx_lines = self._process_multi_heads(conllx_lines)
            # accumulates dataset into dataset
            for line in conllx_lines:
                line[0] = int(line[0])
                line[6] = int(line[6])
                line[7] = line[7].split(':')[0]
            dataset.append([lang, genre, split, seqid, *zip(*conllx_lines)])

        fields = self.corpus_cfg.keys + self.corpus_cfg.fields
        dataframe = pd.DataFrame(dataset, columns=fields)

        return dataframe

    def _get_corpus_data_file(self, lang: str, genre: str, split: str) -> Optional[Path]:
        data_path = self.corpus_path / f"ud_{self.corpus_cfg.langs[lang]}-{genre}"
        data_file = data_path / f"{lang}_{genre}-ud-{split}.conllu"
        assert data_file.exists(), f"Corpus file {data_file} does not exist."

        return data_file

    def _iter_sentences(self, data_file: str) -> List[str]:
        with open(data_file, mode='r', encoding="utf-8") as f:
            buffer = []
            for line in f:
                if line.startswith(self.corpus_cfg.comment_symbol):
                    continue
                if not line.strip():
                    if buffer:
                        yield buffer
                        buffer = []
                    else:
                        continue
                else:
                    buffer.append(line.strip())
            if buffer:
                yield buffer

    def _process_span_lines(self, conllx_lines: List) -> List:
        head_field_id = self.corpus_cfg.fields.index("head")
        copied_lines = copy.deepcopy(conllx_lines)
        index_mappings = {'0': '0'}
        for idx in range(len(copied_lines)):
            if idx >= len(copied_lines):
                break
            line = copied_lines[idx]
            if '-' in line[0]:  # if index field is range, such as 10-12
                left, right = [int(x) for x in line[0].split('-')]
                width = right - left + 1

                # copy to create a new line for the range
                new_line = copied_lines[idx + 1]
                new_line[0] = str(left)  # set new line's id
                new_line[1] = line[1]  # set new line's form

                # list head indices of words in the range
                range_indices = [line[head_field_id] for line in copied_lines[idx+1: idx+1+width]]

                # only keep head indices not in the range
                keep_indices = [x for x in range_indices if not (left <= int(x) <= right)]

                # remove duplicate indices and keep order
                keep_indices = list(dict.fromkeys(keep_indices))
                if len(keep_indices) == 0:
                    raise ValueError("No head for the range.")

                # set new line's head_indices, may have multiple heads, need to be resolved further.
                if len(keep_indices) == 1:
                    new_line[head_field_id] = keep_indices[0]
                else:
                    new_line[head_field_id] = keep_indices

                # set new line for the range and remove original lines
                copied_lines[idx] = new_line
                del copied_lines[idx+1:idx+1+width]

                # record index mapping, map removed indexes to new one
                for i in range(left, right + 1):
                    index_mappings[str(i)] = str(idx + 1)
            else:
                # record index mapping, set idx+1 because index begins with 1
                index_mappings[copied_lines[idx][0]] = str(idx + 1)

            copied_lines[idx][0] = str(idx + 1)

        def mapping(x):
            if isinstance(x, list):
                return [index_mappings[y] for y in x]  # noqa
            else:
                return index_mappings[x]

        for i, line in enumerate(copied_lines):
            line[head_field_id] = mapping(line[head_field_id])

        return copied_lines

    def _process_multi_heads(self, conllx_lines: List) -> List:
        head_field_id = self.corpus_cfg.fields.index("head")
        data = list(zip(*conllx_lines))
        head_indices = list(data[head_field_id])
        for i, indices in enumerate(head_indices, start=1):
            if not isinstance(indices, list):
                continue
            indices = list(dict.fromkeys(indices))
            if len(indices) == 0:
                raise ValueError("No head line found.")
            if '0' in indices:
                head_indices[i - 1] = '0'
                continue
            for idx in indices:
                looped = head_indices[int(idx)-1] == str(i)
                looped = looped or isinstance(head_indices[int(idx)-1], list) and str(i) in head_indices[int(idx)-1]
                if looped:
                    indices.remove(idx)
            if len(indices) == 1:
                head_indices[i - 1] = indices[0]  # noqa
            elif len(indices) == 0:
                raise ValueError("No head line found.")
            else:
                # remaining ambiguity found, choose last indic
                head_indices[i - 1] = indices[-1]  # noqa

        for i, index in enumerate(head_indices):
            assert (isinstance(index, str)), "Multiple head indices found."
            conllx_lines[i][head_field_id] = index

        return conllx_lines
