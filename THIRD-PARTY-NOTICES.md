# Third-party notices

SwiftCFD contains code derived from DeepCFD and uses the DeepCFD dataset. This file says which files derive from
DeepCFD and carries the notices that DeepCFD's licences require.

## DeepCFD code (MIT License)

Source: https://github.com/mdribeiro/DeepCFD, read at commit f9cc152fe1190895cdeb18b102c2640bc5587deb.
Licence: https://github.com/mdribeiro/DeepCFD/blob/f9cc152fe1190895cdeb18b102c2640bc5587deb/LICENSE

Files in this repository that derive from it:

| File | Derived part | From DeepCFD |
|---|---|---|
| `app.py` | `create_layer` to the end of `class UNetEx` (marked by comments) | `create_layer`: copied from `src/deepcfd/models/AutoEncoder.py`; `create_encoder_block`, `create_decoder_block`, `create_encoder`, `create_decoder`, `class UNetEx`: reformatted from `src/deepcfd/models/UNetEx.py` |
| `service/model.py` | the same block (marked by comments) | copied from `app.py`, so derived from the same DeepCFD files |

Each of these files also carries the notice below in a comment above the derived part, so that the notice travels with
the file (the Docker image, for example, copies only `service/`).

`train_pass1.py` and `train_finetune.py` import the `deepcfd` package, installed from DeepCFD's repository; they do
not copy its code.

DeepCFD's copyright and permission notice:

```text
MIT License

Copyright (c) 2020 Mateus Dias Ribeiro

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## DeepCFD dataset (CC BY 4.0)

The dataset is not stored in this repository. `app.py` downloads `dataX.pkl` and `dataY.pkl` from
https://huggingface.co/vamsigudipati/deepcfd-model, where they are unchanged copies of the files in `DeepCFD.zip`.

- Title: DeepCFD
- Creator: Mateus Dias Ribeiro
- Source: https://zenodo.org/records/3666056, DOI [10.5281/zenodo.3666056](https://doi.org/10.5281/zenodo.3666056)
- Licence: Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/
- Changes: none; `dataX.pkl` and `dataY.pkl` are byte-identical to the files of the same names in `DeepCFD.zip`.
- The creator provides the dataset as is, without warranties (CC BY 4.0, Section 5).
