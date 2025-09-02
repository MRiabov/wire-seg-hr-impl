---
trigger: always_on
---

When executing python files, use python3 instead of python because that adheres to project's venv. Additionally, if you haven't activated venv yet, you have to activate it or else the execution will fail with module not found exception.