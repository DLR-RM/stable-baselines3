---
name: "\U0001F41B Bug Report"
about: Submit a bug report to help us improve Stable-Baselines3
labels: bug
title: "[Bug] bug title"
---

**Important Note: We do not do technical support, nor consulting** and don't answer personal questions per email.
Please post your question on the [RL Discord](https://discord.com/invite/xhfNqQv), [Reddit](https://www.reddit.com/r/reinforcementlearning/) or [Stack Overflow](https://stackoverflow.com/) in that case.


If your issue is related to a **custom gym environment**, please use the custom gym env template.

### 🐛 Bug

A clear and concise description of what the bug is.


### To Reproduce

Steps to reproduce the behavior.

Please try to provide a minimal example to reproduce the bug. Error messages and stack traces are also helpful.

Please use the [markdown code blocks](https://help.github.com/en/articles/creating-and-highlighting-code-blocks)
for both code and stack traces.

```python
from stable_baselines3 import ...

```

```bash
Traceback (most recent call last): File ...

```

### Expected behavior

A clear and concise description of what you expected to happen.


### System Info

Describe the characteristic of your environment:
 * Describe how the library was installed (pip, docker, source, ...)
 * GPU models and configuration
 * Python version
 * PyTorch version
 * Gym version
 * Versions of any other relevant libraries

You can use `sb3.get_system_info()` to print relevant packages info:
```python
import stable_baselines3 as sb3
sb3.get_system_info()
```

### Additional context
Add any other context about the problem here.

### Checklist

- [ ] I have checked that there is no similar [issue](https://github.com/DLR-RM/stable-baselines3/issues) in the repo (**required**)
- [ ] I have read the [documentation](https://stable-baselines3.readthedocs.io/en/master/) (**required**)
- [ ] I have provided a minimal working example to reproduce the bug (**required**)
