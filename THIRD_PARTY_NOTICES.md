# Third-party notices

QuantX is released under the MIT License ([LICENSE](LICENSE)). It builds on the
open-source software listed here. This file ships in every wheel, beside the QuantX
license.

## Compiled into the native engine (in every wheel)

The library (`libquantx.so`, `libquantx.dylib`, `quantx.dll`) contains code from:

| Component | Version | License | Copyright |
|---|---|---|---|
| [Eigen](https://eigen.tuxfamily.org) | 3.4.0 | MPL-2.0 | Eigen contributors |
| [nlohmann/json](https://github.com/nlohmann/json) | 3.11.3 | MIT | Copyright (c) 2013-2022 Niels Lohmann |
| [spdlog](https://github.com/gabime/spdlog) | 1.17.0 | MIT | Copyright (c) 2016 - present, Gabi Melman and spdlog contributors |
| [{fmt}](https://github.com/fmtlib/fmt), bundled with spdlog | (spdlog's) | MIT | Copyright (c) 2012 - present, Victor Zverovich and {fmt} contributors |

**Eigen** is used unmodified, as header-only templates, and is compiled with
`EIGEN_MPL2_ONLY`, so no Eigen code under another license is included. Under the
Mozilla Public License 2.0, its source code is available from
https://gitlab.com/libeigen/eigen/-/tree/3.4.0, and the license text is at
https://www.mozilla.org/MPL/2.0/.

**nlohmann/json, spdlog and {fmt}** are used under the MIT License. The copyright
notices are in the table above; the license text is:

> Permission is hereby granted, free of charge, to any person obtaining a copy of this
> software and associated documentation files (the "Software"), to deal in the Software
> without restriction, including without limitation the rights to use, copy, modify,
> merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
> permit persons to whom the Software is furnished to do so, subject to the following
> conditions:
>
> The above copyright notice and this permission notice shall be included in all copies
> or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
> INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
> PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
> HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
> CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
> OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Bundled with the wheels by the packaging tools

| Wheel | Library | License |
|---|---|---|
| Linux (`manylinux_2_28`) | `libgomp`, the GNU OpenMP runtime, from GCC | GPL-3.0-or-later WITH GCC-exception-3.1 |
| Windows | `vcomp140.dll`, the Microsoft Visual C++ OpenMP runtime | Microsoft Visual C++ redistributable terms |

The GCC Runtime Library Exception permits distributing `libgomp` with programs under
any license. Its source is part of GCC (https://gcc.gnu.org), and the exception's text
is at https://www.gnu.org/licenses/gcc-exception-3.1.html. `vcomp140.dll` is one of
the files Microsoft licenses for redistribution with applications built by Visual
Studio. The macOS wheels bundle no runtime library, and the Windows engine links the
C++ runtime statically.

## In the repository only (not in the wheels)

| Component | Where | License | Copyright |
|---|---|---|---|
| [CLI11](https://github.com/CLIUtils/CLI11) 2.4.2 | the `qx_cli` executable of a source build | BSD-3-Clause | Copyright (c) 2017-2024 University of Cincinnati, developed by Henry Schreiner under NSF AWARD 1414736 |
| [three.js](https://threejs.org) and its `OrbitControls` | `scripts/vendor/`, for the visualisation scripts | MIT | Copyright 2010-2021 Three.js Authors |

The CMake build downloads nlohmann/json, spdlog, CLI11 and Eigen at configure time
unless they are installed already (`CMakeLists.txt`). Each distribution includes its
own license file.
