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
| [{fmt}](https://github.com/fmtlib/fmt), bundled with spdlog | 12.1.0 | MIT | Copyright (c) 2012 - present, Victor Zverovich and {fmt} contributors |

**Eigen** is used unmodified, as header-only templates. Most of it is under the Mozilla
Public License 2.0: its source code is available from
https://gitlab.com/libeigen/eigen/-/tree/3.4.0, and the license text is at
https://www.mozilla.org/MPL/2.0/. The engine is compiled with `EIGEN_MPL2_ONLY`, which
keeps out Eigen's LGPL code.

A few headers that `Eigen/Core` always includes carry, or derive from, permissive
licenses:

- `Eigen/src/Core/arch/Default/BFloat16.h`: Apache License 2.0, Copyright 2017 The
  TensorFlow Authors (text: https://www.apache.org/licenses/LICENSE-2.0);
- `Eigen/src/Core/util/MKL_support.h`: BSD 3-Clause, Copyright (c) 2011, Intel
  Corporation (below);
- `Eigen/src/Core/arch/Default/Half.h`: its half-precision conversion routines are
  Copyright (c) Fabian Giesen, 2016, whose license permits redistribution and use in
  source and binary forms, with or without modification, without warranty;
- the vectorised sin and cos (`arch/SSE`, `arch/AVX`, `arch/Default`) derive from
  Julien Pommier's sse_mathfun, under the zlib license.

The BSD 3-Clause terms for Intel's file:

> Redistribution and use in source and binary forms, with or without modification, are
> permitted provided that the following conditions are met: (1) Redistributions of
> source code must retain the above copyright notice, this list of conditions and the
> following disclaimer. (2) Redistributions in binary form must reproduce the above
> copyright notice, this list of conditions and the following disclaimer in the
> documentation and/or other materials provided with the distribution. (3) Neither the
> name of the copyright holder nor the names of its contributors may be used to endorse
> or promote products derived from this software without specific prior written
> permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
> EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
> MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
> THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
> SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
> PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
> INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
> STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
> THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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

`libgomp` is free software, licensed under the GNU General Public License version 3
or later with the GCC Runtime Library Exception 3.1. Both texts ship in every wheel, in
the `licenses` folder beside this file (`GPL-3.0.txt`, `GCC-exception-3.1.txt`; in the
repository, `python/licenses/`). The exception lets the engine, which only links to
the library, keep its own license; the library itself stays under the GPL.

The Linux wheel's `quantx_engine.libs/libgomp-*.so.1` is the `libgomp.so.1` of the
`manylinux_2_28` build image (AlmaLinux 8) that cibuildwheel uses. auditwheel copies it
into the wheel, renaming it and rewriting its `soname`, and changes nothing else. Its
source is GCC's: in the image, `rpm -qf` on that file names the package that provides
it, and AlmaLinux publishes the source RPM of each of its packages at
https://vault.almalinux.org. Upstream, libgomp is in the GCC repository,
https://gcc.gnu.org/git/?p=gcc.git;a=tree;f=libgomp.

`vcomp140.dll` is one of the files Microsoft licenses for redistribution with
applications built by Visual Studio. The macOS wheels bundle no runtime library, and
the Windows engine links the C++ runtime statically.

## In the repository only (not in the wheels)

| Component | Where | License | Copyright |
|---|---|---|---|
| [CLI11](https://github.com/CLIUtils/CLI11) 2.4.2 | the `qx_cli` executable of a source build | BSD-3-Clause | Copyright (c) 2017-2024 University of Cincinnati, developed by Henry Schreiner under NSF AWARD 1414736 |
| [three.js](https://threejs.org) and its `OrbitControls` | `scripts/vendor/`, for the visualisation scripts | MIT | Copyright 2010-2021 Three.js Authors |

The CMake build downloads Eigen at configure time, and nlohmann/json, spdlog and CLI11
too unless they are installed already (`CMakeLists.txt`). Each distribution includes its
own license file.
