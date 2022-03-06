import re
from typing import List, Tuple, Optional

from hebsafeharbor.common.terms_recognizer import TermsRecognizer


class ContextTermsRecognizer(TermsRecognizer):

    def __init__(self, phrase_list: List[str]):
        """
        Initializes TermsRecognizer
        :param phrase_list: list of terms to recognize
        """
        super().__init__(phrase_list=phrase_list)

    def __call__(self, text: str, word: str, prefixes: Optional[List[str]] = None) -> List[Tuple[int, int]]:
        """
        This method searches for terms in text
        :param text: text
        :return: List of starting offsets of matches and their length
        """
        offsets = []
        for end_index_short, length in self._automaton.iter(text):
            offset = end_index_short - length + 1
            if length == 1 or offset < 0 or offset >= len(text):
                pass
            else:
                prefixes_pattern_raw = "|".join(prefixes)
                pattern = "(\W|^)(" + text[
                                      offset:end_index_short + 1] + ").*\W(" + prefixes_pattern_raw + ")(" + word + ")(\W|$)"
                found_elements = re.search(pattern, text)

                if found_elements:
                    supported_word_position = found_elements[2]
                    supported_word = text[supported_word_position[0]:supported_word_position[1]]
                    preposition_position = found_elements[3]
                    preposition = text[preposition_position[0]:preposition_position[1]]
                    offsets.append((supported_word, supported_word_position, preposition, preposition_position))

        # drop duplicates
        offsets = list(set(offsets))

        return offsets
