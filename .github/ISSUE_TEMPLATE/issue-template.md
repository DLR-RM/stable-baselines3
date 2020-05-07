---
name: Issue Template
about: How to create an issue for this repository

---

**Important Note: We do not do technical support, nor consulting** and don't answer personal questions per email.
Please post your question on [reddit](https://www.reddit.com/r/reinforcementlearning/) or [stack overflow](https://stackoverflow.com/) in that case.

If you have any questions, feel free to create an issue with the tag [question].  
If you wish to suggest an enhancement or feature request, add the tag [feature request].  
If you are submitting a bug report, please fill in the following details.

If your issue is related to a custom gym environment, please check it first using:

```python
from stable_baselines3.common.env_checker import check_env

env = CustomEnv(arg1, ...)
# It will check your custom environment and output additional warnings if needed
check_env(env)
```

**Describe the bug**
A clear and concise description of what the bug is.

**Code example**
Please try to provide a minimal example to reproduce the bug. Error messages and stack traces are also helpful.

Please use the [markdown code blocks](https://help.github.com/en/articles/creating-and-highlighting-code-blocks)
for both code and stack traces.

```python
from stable_baselines3 import ...

```

```bash
Traceback (most recent call last): File ...

```

**System Info**
Describe the characteristic of your environment:
 * Describe how the library was installed (pip, docker, source, ...)
 * GPU models and configuration
 * Python version
 * PyTorch version
 * Gym version
 * Versions of any other relevant libraries

**Additional context**
Add any other context about the problem here.
