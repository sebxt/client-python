import unittest

from mistralai.client.models import (
    FunctionCallEntry,
    FunctionResultEntry,
    MessageInputEntry,
    MessageOutputEntry,
)

from ..run.context import RunContext, _validate_run


class TestValidateRunInputs(unittest.IsolatedAsyncioTestCase):
    async def _input_entries(self, inputs):
        run_ctx = RunContext(model="mistral-medium-latest")
        _, run_result, input_entries = await _validate_run(
            beta_client=None,  # type: ignore[arg-type]
            run_ctx=run_ctx,
            inputs=inputs,
        )
        self.assertEqual(run_result.input_entries, input_entries)
        return input_entries

    async def test_string_input(self):
        entries = await self._input_entries("Hello")
        self.assertEqual(entries, [MessageInputEntry(role="user", content="Hello")])

    async def test_dict_input(self):
        entries = await self._input_entries([{"role": "user", "content": "Hello"}])
        self.assertEqual(entries, [MessageInputEntry(role="user", content="Hello")])

    async def test_typed_input_entries_are_kept(self):
        inputs = [
            MessageInputEntry(role="user", content="What is the weather?"),
            FunctionCallEntry(
                tool_call_id="call_1", name="get_weather", arguments="{}"
            ),
            FunctionResultEntry(tool_call_id="call_1", result="sunny"),
            MessageOutputEntry(content="It is sunny."),
            MessageInputEntry(role="user", content="Thanks"),
        ]
        entries = await self._input_entries(inputs)
        self.assertEqual(entries, inputs)

    async def test_mixed_typed_and_dict_inputs_keep_order(self):
        typed = MessageInputEntry(role="user", content="first")
        entries = await self._input_entries(
            [typed, {"role": "user", "content": "second"}]
        )
        self.assertEqual(
            entries, [typed, MessageInputEntry(role="user", content="second")]
        )


if __name__ == "__main__":
    unittest.main()
