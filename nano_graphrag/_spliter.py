from typing import List, Optional, Union, Literal

class SeparatorSplitter:
    def __init__(
        self,
        separators: Optional[List[List[int]]] = None,
        keep_separator: Union[bool, Literal["start", "end"]] = "end",
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_function: callable = len,
    ):
        self._separators = separators or [[10], [13, 10]]  # 默认使用换行符作为分隔符
        self._keep_separator = keep_separator
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function

    def split_tokens(self, tokens: List[int]) -> List[List[int]]:
        splits = self._split_tokens_with_separators(tokens)
        return self._merge_splits(splits)

    def _split_tokens_with_separators(self, tokens: List[int]) -> List[List[int]]:
        splits = []
        current_split = []
        i = 0
        while i < len(tokens):
            separator_found = False
            for separator in self._separators:
                if tokens[i:i+len(separator)] == separator:
                    if current_split:
                        if self._keep_separator == "end":
                            current_split.extend(separator)
                            splits.append(current_split)
                            current_split = []
                        elif self._keep_separator == "start":
                            splits.append(current_split)
                            current_split = separator[:]
                        else:
                            splits.append(current_split)
                            current_split = []
                    elif self._keep_separator:
                        current_split.extend(separator)
                    i += len(separator)
                    separator_found = True
                    break
            if not separator_found:
                current_split.append(tokens[i])
                i += 1
        if current_split:
            splits.append(current_split)
        return [s for s in splits if s]

    def _merge_splits(self, splits: List[List[int]]) -> List[List[int]]:
        merged_splits = []
        current_split = []
        current_length = 0
        separator = [] if self._keep_separator is False else self._separators[-1]

        for split in splits:
            if self._length_function(current_split) + self._length_function(split) <= self._chunk_size:
                if current_split and separator:
                    current_split.extend(separator)
                current_split.extend(split)
            else:
                if current_split:
                    merged_splits.append(current_split)
                current_split = split
            if self._length_function(current_split) >= self._chunk_size:
                merged_splits.append(current_split)
                current_split = []
        if current_split:
            merged_splits.append(current_split)

        if self._chunk_overlap > 0:
            return self._enforce_overlap(merged_splits)
        return merged_splits

    def _enforce_overlap(self, chunks: List[List[int]]) -> List[List[int]]:
        new_chunks = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                new_chunks.append(chunk)
            else:
                overlap_tokens = chunks[i-1][-self._chunk_overlap:]
                new_chunk = overlap_tokens + chunk
                if self._length_function(new_chunk) > self._chunk_size:
                    new_chunk = new_chunk[-self._chunk_size:]
                new_chunks.append(new_chunk)
        return new_chunks

# EXAMPLE USAGE
if __name__ == "__main__":
    import tiktoken
    tokenizer = tiktoken.encoding_for_model("gpt-4")

    def tokenize(text: str) -> List[int]:
        return tokenizer.encode(text)

    def detokenize(tokens: List[int]) -> str:
        return tokenizer.decode(tokens)
    
    # 创建splitter实例
    splitter = SeparatorSplitter(
        separators=[tokenize('\n'), tokenize('.')],  # 使用换行符和句号作为分隔符
        chunk_size=5,
        chunk_overlap=0,
        keep_separator="end"
    )

    # 示例文本
    text = "This is a sample text. It contains multiple sentences.\nSome sentences are short. Others are longer."
    tokens = tokenize(text)

    # 分割tokens
    split_tokens = splitter.split_tokens(tokens)

    print("Split tokens:")
    for i, token_chunk in enumerate(split_tokens):
        print(f"Chunk {i + 1}:")
        print(detokenize(token_chunk))
        print("---")