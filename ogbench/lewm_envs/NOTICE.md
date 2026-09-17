# LeWM evaluation environment provenance

The environment and variation-space implementations in this directory were
migrated from [`pickxiguapi/stable-worldmodel`](https://github.com/pickxiguapi/stable-worldmodel)
at commit `b874c7ef9cc96f099407b7bfb4e20c4c6e0b1f8f`.

The source project's package metadata declares the MIT license and names Lucas
Maes, Quentin Le Lidec, and Randall Balestriero as authors. The source checkout
does not contain a standalone license or copyright file, so the standard MIT
license text is included below with an attribution to the project contributors.
It applies to the migrated files in this directory.

Local changes include:

- package-local imports and `ogbench-lewm/...` Gymnasium IDs;
- a Torch-free NumPy implementation of TwoRoom;
- removal of unused Shapely helpers;
- a NumPy/JAX-only dataset-goal evaluator in `evaluation.py`.

No runtime import, checkout path, Python executable, or package dependency on
`stable_worldmodel` remains.

## MIT license for the migrated Stable World Model code

Copyright (c) 2025 Stable World Model contributors

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
