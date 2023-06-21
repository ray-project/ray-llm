# Based on code from transformers.generation.streamers
# Up to date with transformers 4.30.2

# Copyright 2018- The Hugging Face team. All rights reserved.

#                                  Apache License
#                            Version 2.0, January 2004
#                         http://www.apache.org/licenses/

#    TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

#    1. Definitions.

#       "License" shall mean the terms and conditions for use, reproduction,
#       and distribution as defined by Sections 1 through 9 of this document.

#       "Licensor" shall mean the copyright owner or entity authorized by
#       the copyright owner that is granting the License.

#       "Legal Entity" shall mean the union of the acting entity and all
#       other entities that control, are controlled by, or are under common
#       control with that entity. For the purposes of this definition,
#       "control" means (i) the power, direct or indirect, to cause the
#       direction or management of such entity, whether by contract or
#       otherwise, or (ii) ownership of fifty percent (50%) or more of the
#       outstanding shares, or (iii) beneficial ownership of such entity.

#       "You" (or "Your") shall mean an individual or Legal Entity
#       exercising permissions granted by this License.

#       "Source" form shall mean the preferred form for making modifications,
#       including but not limited to software source code, documentation
#       source, and configuration files.

#       "Object" form shall mean any form resulting from mechanical
#       transformation or translation of a Source form, including but
#       not limited to compiled object code, generated documentation,
#       and conversions to other media types.

#       "Work" shall mean the work of authorship, whether in Source or
#       Object form, made available under the License, as indicated by a
#       copyright notice that is included in or attached to the work
#       (an example is provided in the Appendix below).

#       "Derivative Works" shall mean any work, whether in Source or Object
#       form, that is based on (or derived from) the Work and for which the
#       editorial revisions, annotations, elaborations, or other modifications
#       represent, as a whole, an original work of authorship. For the purposes
#       of this License, Derivative Works shall not include works that remain
#       separable from, or merely link (or bind by name) to the interfaces of,
#       the Work and Derivative Works thereof.

#       "Contribution" shall mean any work of authorship, including
#       the original version of the Work and any modifications or additions
#       to that Work or Derivative Works thereof, that is intentionally
#       submitted to Licensor for inclusion in the Work by the copyright owner
#       or by an individual or Legal Entity authorized to submit on behalf of
#       the copyright owner. For the purposes of this definition, "submitted"
#       means any form of electronic, verbal, or written communication sent
#       to the Licensor or its representatives, including but not limited to
#       communication on electronic mailing lists, source code control systems,
#       and issue tracking systems that are managed by, or on behalf of, the
#       Licensor for the purpose of discussing and improving the Work, but
#       excluding communication that is conspicuously marked or otherwise
#       designated in writing by the copyright owner as "Not a Contribution."

#       "Contributor" shall mean Licensor and any individual or Legal Entity
#       on behalf of whom a Contribution has been received by Licensor and
#       subsequently incorporated within the Work.

#    2. Grant of Copyright License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       copyright license to reproduce, prepare Derivative Works of,
#       publicly display, publicly perform, sublicense, and distribute the
#       Work and such Derivative Works in Source or Object form.

#    3. Grant of Patent License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       (except as stated in this section) patent license to make, have made,
#       use, offer to sell, sell, import, and otherwise transfer the Work,
#       where such license applies only to those patent claims licensable
#       by such Contributor that are necessarily infringed by their
#       Contribution(s) alone or by combination of their Contribution(s)
#       with the Work to which such Contribution(s) was submitted. If You
#       institute patent litigation against any entity (including a
#       cross-claim or counterclaim in a lawsuit) alleging that the Work
#       or a Contribution incorporated within the Work constitutes direct
#       or contributory patent infringement, then any patent licenses
#       granted to You under this License for that Work shall terminate
#       as of the date such litigation is filed.

#    4. Redistribution. You may reproduce and distribute copies of the
#       Work or Derivative Works thereof in any medium, with or without
#       modifications, and in Source or Object form, provided that You
#       meet the following conditions:

#       (a) You must give any other recipients of the Work or
#           Derivative Works a copy of this License; and

#       (b) You must cause any modified files to carry prominent notices
#           stating that You changed the files; and

#       (c) You must retain, in the Source form of any Derivative Works
#           that You distribute, all copyright, patent, trademark, and
#           attribution notices from the Source form of the Work,
#           excluding those notices that do not pertain to any part of
#           the Derivative Works; and

#       (d) If the Work includes a "NOTICE" text file as part of its
#           distribution, then any Derivative Works that You distribute must
#           include a readable copy of the attribution notices contained
#           within such NOTICE file, excluding those notices that do not
#           pertain to any part of the Derivative Works, in at least one
#           of the following places: within a NOTICE text file distributed
#           as part of the Derivative Works; within the Source form or
#           documentation, if provided along with the Derivative Works; or,
#           within a display generated by the Derivative Works, if and
#           wherever such third-party notices normally appear. The contents
#           of the NOTICE file are for informational purposes only and
#           do not modify the License. You may add Your own attribution
#           notices within Derivative Works that You distribute, alongside
#           or as an addendum to the NOTICE text from the Work, provided
#           that such additional attribution notices cannot be construed
#           as modifying the License.

#       You may add Your own copyright statement to Your modifications and
#       may provide additional or different license terms and conditions
#       for use, reproduction, or distribution of Your modifications, or
#       for any such Derivative Works as a whole, provided Your use,
#       reproduction, and distribution of the Work otherwise complies with
#       the conditions stated in this License.

#    5. Submission of Contributions. Unless You explicitly state otherwise,
#       any Contribution intentionally submitted for inclusion in the Work
#       by You to the Licensor shall be under the terms and conditions of
#       this License, without any additional terms or conditions.
#       Notwithstanding the above, nothing herein shall supersede or modify
#       the terms of any separate license agreement you may have executed
#       with Licensor regarding such Contributions.

#    6. Trademarks. This License does not grant permission to use the trade
#       names, trademarks, service marks, or product names of the Licensor,
#       except as required for reasonable and customary use in describing the
#       origin of the Work and reproducing the content of the NOTICE file.

#    7. Disclaimer of Warranty. Unless required by applicable law or
#       agreed to in writing, Licensor provides the Work (and each
#       Contributor provides its Contributions) on an "AS IS" BASIS,
#       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
#       implied, including, without limitation, any warranties or conditions
#       of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
#       PARTICULAR PURPOSE. You are solely responsible for determining the
#       appropriateness of using or redistributing the Work and assume any
#       risks associated with Your exercise of permissions under this License.

#    8. Limitation of Liability. In no event and under no legal theory,
#       whether in tort (including negligence), contract, or otherwise,
#       unless required by applicable law (such as deliberate and grossly
#       negligent acts) or agreed to in writing, shall any Contributor be
#       liable to You for damages, including any direct, indirect, special,
#       incidental, or consequential damages of any character arising as a
#       result of this License or out of the use or inability to use the
#       Work (including but not limited to damages for loss of goodwill,
#       work stoppage, computer failure or malfunction, or any and all
#       other commercial damages or losses), even if such Contributor
#       has been advised of the possibility of such damages.

#    9. Accepting Warranty or Additional Liability. While redistributing
#       the Work or Derivative Works thereof, You may choose to offer,
#       and charge a fee for, acceptance of support, warranty, indemnity,
#       or other liability obligations and/or rights consistent with this
#       License. However, in accepting such obligations, You may act only
#       on Your own behalf and on Your sole responsibility, not on behalf
#       of any other Contributor, and only if You agree to indemnify,
#       defend, and hold each Contributor harmless for any liability
#       incurred by, or claims asserted against, such Contributor by reason
#       of your accepting any such warranty or additional liability.

#    END OF TERMS AND CONDITIONS

#    APPENDIX: How to apply the Apache License to your work.

#       To apply the Apache License to your work, attach the following
#       boilerplate notice, with the fields enclosed by brackets "[]"
#       replaced with your own identifying information. (Don't include
#       the brackets!)  The text should be enclosed in the appropriate
#       comment syntax for the file format. We also recommend that a
#       file or class name and description of purpose be included on the
#       same "printed page" as the copyright notice for easier
#       identification within third-party archives.

#    Copyright [yyyy] [name of copyright owner]

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import time
from typing import List, Optional, Union

import torch
from transformers import AutoTokenizer

from aviary.backend.server.models import Response


class BatchEntry:
    """A single entry in the batch."""

    def __init__(self, stopping_sequences: Optional[List[List[int]]] = None) -> None:
        self.token_cache: List[int] = []
        self.print_len = 0
        self.stopped = False
        self.stopping_sequences = stopping_sequences or []
        self.stopping_suspects: List[int] = [0] * len(self.stopping_sequences)

    def add_tokens(self, tokens: List[int]):
        if self.stopped:
            return

        for i, stopping_sequence in enumerate(self.stopping_sequences):
            # If the last token is the next token in the stopping sequence, we
            # increment the stopping suspect counter. Otherwise, we reset it.
            if stopping_sequence[self.stopping_suspects[i]] == tokens[-1]:
                self.stopping_suspects[i] += 1
            else:
                self.stopping_suspects[i] = 0
            # If the stopping suspect counter is equal to the length of the
            # stopping sequence, we mark the entry as stopped.
            if self.stopping_suspects[i] >= len(stopping_sequence):
                if len(stopping_sequence) > 1:
                    # Remove suspect tokens from the cache.
                    self.token_cache = self.token_cache[: -len(stopping_sequence) - 1]
                self.stopped = True
                break

        if self.stopped:
            return

        self.token_cache.extend(tokens)

    def get_printable_text(self, text: str):
        if self.stopped:
            return None

        if self.stopping_suspects and min(self.stopping_suspects) > 0:
            # If we have stopping suspects, do not output text.
            return None

        # After the symbol for a new line, we flush the cache.
        if text.endswith("\n"):
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        # If the last token is a CJK character, we print the characters.
        elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
            printable_text = text[self.print_len :]
            self.print_len += len(printable_text)
        # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
        # which may change with the subsequent token -- there are probably smarter ways to do this!)
        else:
            printable_text = text[self.print_len : text.rfind(" ") + 1]
            self.print_len += len(printable_text)

        return printable_text.replace("\u200b", "")

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  #
            return True

        return False


class TokenBatchPostprocessor:
    """Post-processes batches of tokens. Based on transformer's Streamer API"""

    def __init__(
        self,
        tokenizer: "AutoTokenizer",
        skip_prompt: bool = False,
        stopping_sequences: Optional[List[Union[int, List[int]]]] = None,
        **decode_kwargs,
    ):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.decode_kwargs = decode_kwargs

        stopping_sequences = stopping_sequences or []
        stopping_sequences = [
            [sequence] if isinstance(sequence, int) else sequence
            for sequence in stopping_sequences
        ]
        self.stopping_sequences = stopping_sequences

        # variables used in the streaming process
        self.entries = None
        self.next_tokens_are_prompt = True
        self.num_input_tokens = []

    def _handle_input(self, value: torch.LongTensor):
        self.num_input_tokens = [
            len([token for token in entry if token != self.tokenizer.pad_token_id])
            for entry in value
        ]
        self.next_tokens_are_prompt = False

    def process(self, value: torch.LongTensor) -> List[str]:
        """
        Recives tokens, decodes them, and returns them as they form entire words.
        """
        if self.skip_prompt and self.next_tokens_are_prompt:
            self._handle_input(value)
            return [""] * len(value)

        value = value.unsqueeze(0).T

        if self.entries is None:
            self.entries = [
                BatchEntry(self.stopping_sequences) for i in range(value.shape[0])
            ]

        # Add the new token to the cache and decodes the entire thing.

        for i, entry in enumerate(value):
            self.entries[i].add_tokens(entry.tolist())

        # Do batch_decode as it is more performant.
        texts = self.tokenizer.batch_decode(
            [entry.token_cache for entry in self.entries], **self.decode_kwargs
        )

        printable_batch = []
        for i, text in enumerate(texts):
            printable_text = self.entries[i].get_printable_text(text)
            printable_batch.append(printable_text)
        return printable_batch

    def end(self) -> List[str]:
        """Flushes any remaining cache."""

        # Flush the cache, if it exists
        printable_batch = []
        for entry in self.entries:
            if len(entry.token_cache) > 0:
                text = self.tokenizer.decode(entry.token_cache, **self.decode_kwargs)
                printable_text = text[entry.print_len :]
            else:
                printable_text = None
            printable_batch.append(printable_text)

        self.next_tokens_are_prompt = True
        self.entries = None
        return printable_batch


class ResponseTokenBatchPostprocessor(TokenBatchPostprocessor):
    def __init__(
        self,
        tokenizer: "AutoTokenizer",
        skip_prompt: bool = False,
        stopping_sequences: Optional[List[Union[int, List[int]]]] = None,
        preprocessing_time: Optional[float] = None,
        **decode_kwargs,
    ):
        super().__init__(
            tokenizer,
            skip_prompt,
            stopping_sequences,
            **decode_kwargs,
        )
        self.preprocessing_time = preprocessing_time
        self.start_time = time.monotonic()

    def process(self, value: torch.LongTensor) -> List[Response]:
        if self.skip_prompt and self.next_tokens_are_prompt:
            self._handle_input(value)
            return self._convert_to_responses([None] * len(value))

        return self._convert_to_responses(super().process(value))

    def end(self) -> List[Response]:
        return self._convert_to_responses(super().end(), stream_end=True)

    def _convert_to_responses(
        self, texts: List[str], stream_end: bool = False
    ) -> List[Response]:
        responses = []
        num_input_tokens_batch = sum(self.num_input_tokens)
        num_generated_tokens_batch = 0
        for i, text in enumerate(texts):
            num_generated_tokens = 1 if text is not None and not stream_end else 0
            num_generated_tokens_batch += num_generated_tokens
            responses.append(
                Response(
                    generated_text=text or "",
                    preprocessing_time=self.preprocessing_time,
                    num_generated_tokens=num_generated_tokens,
                    num_input_tokens=self.num_input_tokens[i],
                    num_input_tokens_batch=num_input_tokens_batch,
                    generation_time=time.monotonic() - self.start_time,
                )
            )

        for response in responses:
            response.num_generated_tokens_batch = num_generated_tokens_batch
        self.start_time = time.monotonic()
        return responses
