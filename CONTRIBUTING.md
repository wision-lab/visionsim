# Contributing to VisionSIM

Thanks for your interest in VisionSIM! We welcome contributions of all kinds — bug reports,
documentation fixes, and code — and we are grateful for the time you spend on them.

## Opening Issues

We use GitHub issues to track bugs and feature requests. When opening an issue, please be as
specific as you can:

- A clear, descriptive title.
- The version of VisionSIM and your Python version.
- Your operating system and hardware, including GPU model if relevant.
- Exact steps to reproduce the problem, and the output you observed versus what you expected.
- A minimal reproduction where possible, plus any relevant logs or tracebacks.

## Pull Requests

- Opening an issue before starting work is encouraged, especially for larger changes — it lets
  us agree on the approach before you invest time in it.
- Maintainers may decline changes that don't fit the project's design goals, so discussing the
  change early avoids wasted effort.

## Development Setup & Tests

For instructions on setting up an editable install, running the test suite (`inv test`), linting
(`inv lint`), and configuring pre-commit hooks, see
[docs/source/development.rst](docs/source/development.rst).

## AI Use Policy

Generative AI tools can be useful. We don't ban them. We do ask that they be used in a way that
respects everyone's time. This policy is adapted from
[NumPy's AI policy](https://numpy.org/devdocs/dev/ai_policy.html).

**Responsibility.** You are responsible for any code you submit to VisionSIM, whether you wrote
it yourself or generated it with AI. You must understand the code you submit and the existing
related code well enough to explain it. Don't submit a patch you cannot explain yourself. When
explaining your contribution, write the comments, pull request descriptions, and issue
descriptions yourself rather than having AI generate them.

**Disclosure.** Tell us in the pull request whether AI was used, which tool(s), how, and what
code or text is AI-generated. We may reject pull requests that don't include this disclosure.

**Code quality.** Contributions must meet VisionSIM's standards. We will reject pull requests we
consider AI-generated slop. Please don't waste maintainer time with code that is fully or mostly
AI-generated and doesn't meet the bar.

**Copyright.** VisionSIM's code is released under GPL-3.0-or-later, and contributions are
accepted under the MIT License (see [Licensing](#licensing) below). You must own the copyright
of any code you submit, or include the license(s) for any third-party code in your patch.
AI-generated code may infringe on copyright; it is your responsibility not to infringe. We
reserve the right to reject any pull request, AI-generated or not, whose copyright status is in
question.

**Communication.** On issues, discussions, and pull requests, don't let AI speak for you. The
only exceptions are translation and grammar editing. If people wanted to talk to a chatbot,
they'd do it themselves; human-to-human communication is what makes an open source community
work.

**AI agents.** An AI agent that writes code and autonomously opens a pull request is not
permitted. A human must review any generated code and open the pull request themselves, in line
with the "Responsibility" section above.


## Licensing

VisionSIM is released under the **GNU General Public License, version 3 or later
(GPL-3.0-or-later)**. A commercial license is also available for those who cannot comply
with the GPL; if you are interested, contact us.

All contributions are accepted under the MIT License. By opening a pull request you agree
that your contribution is licensed to the project under the MIT License, and you accept these
terms. No contributor license agreement (CLA) is required, and you keep the copyright to
your contribution.

What this means in practice:

- Your pull request's code is licensed under the MIT License and you retain copyright in it.
- Once merged, it is distributed as part of VisionSIM under GPL-3.0-or-later.
- Because you license your contribution to the project under MIT, the maintainers may
  relicense the combined work without needing a CLA or a copyright assignment from you.
- The MIT License requires that your copyright notice be maintained, so it's included below:


```
MIT License

Copyright (2026) Code Contributor

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