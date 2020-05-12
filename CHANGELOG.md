Changelog
==========

Support for MultiBinary / MultiDiscrete spaces
---------------------------

New Features:
^^^^^^^^^^^^^
- Implemented MultiCategorical & Bernoulli distributions in `common/distributions.py`
- Added support for MultiCategorial & Bernoulli observation / action spaces in `preprocessing.py`, `ppo/policies.py`
- Merged the Categorical and MultiCategorical tests, added Bernoulli test in `test_distributions.py`

