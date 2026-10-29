"""Code-to-pseudocode step for the self-consistency loop."""

from .code_summarization import CodeSummarizationTask
from .factory import register_task


PSEUDOCODE_PROMPT = """Convert the following Python code into structured pseudocode.

Describe its actual behavior.
Preserve function names, parameters, return values, conditions,
loops, and edge cases.
Use FUNCTION, INPUTS, OUTPUT, STEPS, and END FUNCTION.
Use plain English for the steps.
Do not copy Python statements or add explanations outside the pseudocode.

```python
{code}
```
"""


@register_task("code_to_pseudocode")
class CodeToPseudocodeTask(CodeSummarizationTask):
    """Use pseudocode in place of prose, with the standard assertion wrapper.

    Inherited extraction strips surrounding whitespace without repairing the
    model's response. Inherited postprocessing carries the original assertion
    forward, exactly as in the standard loop.
    """

    def create_prompt(self, input_text: str) -> str:
        template = self.task_config.get("prompt_template", PSEUDOCODE_PROMPT)
        return template.format(code=input_text)
